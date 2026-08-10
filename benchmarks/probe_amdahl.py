"""What fraction of a real decode step is RoPE + KV-cache append, and what can a kernel win?

Substitution ablation, not profiler attribution: under torch.compile the operation fuses into its
neighbours and there is no kernel left to attribute. The instrument is to remove the operation and
measure the step. `op_removed` is the upper bound on a fused kernel's win -- it removes the
operation's launches as well as its compute, which is what a fused kernel also does.
"""
import argparse
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import decode_loop as dl
from benchmarks.benchmark_utils import (
    REPO_ROOT,
    env_metadata,
    record_gpu_state_end,
    time_amortized_call,
    time_synchronized_call,
    write_csv,
    write_json,
)
from benchmarks.rung_patches import (
    FULL,
    RUNGS,
    apply_rung,
    assert_no_roll,
    assert_patch_target_reachable,
    resolve_rungs,
)

# b=32 sweeps ctx downward: the operator's own cost is context-independent, but attention's KV
# read grows with context, so the operator's share is largest at small ctx -- the gate calibration.
PROBE_CONFIGS = [
    dl.DecodeConfig(dl.DEFAULT_MODEL, 32, 128),
    dl.DecodeConfig(dl.DEFAULT_MODEL, 32, 256),
    dl.DecodeConfig(dl.DEFAULT_MODEL, 32, 512),
    dl.DecodeConfig(dl.DEFAULT_MODEL, 32, 1024),
    dl.DecodeConfig(dl.DEFAULT_MODEL, 8, 1024),
    dl.DecodeConfig(dl.DEFAULT_MODEL, 1, 1024),
]


def _reset_compile_counters():
    """torch._dynamo.reset() does not clear these, and counters['frames'] is not populated on
    this path in torch 2.11 -- stats.unique_graphs is the one that moves."""
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()


def _compiled_evidence():
    from torch._dynamo.utils import counters
    return counters["stats"]["unique_graphs"], counters["stats"]["calls_captured"]


def _cudagraph_manager_exists():
    try:
        from torch._inductor.cudagraph_trees import get_manager
        return get_manager(0, create_if_none_exists=False) is not None
    except Exception:
        return False


def headroom_slots(args):
    """op_doubled advances the cache counter twice per step. Sized uniformly across rungs so
    every rung allocates the same cache and attention reads the same length."""
    return 2 * (args.warmup + args.iters) + 8


def repeat_stats(samples):
    """Spread of repeated timings of one already-compiled callable.

    This is run-to-run timing noise, not compile-to-compile variance: the repeats reuse one
    compiled instance. Rungs are separately compiled, so a rung-vs-rung delta carries variance
    this does not capture -- see LIMITATIONS.
    """
    s = sorted(samples)
    return {
        "amortized_step_ms": st.median(s),
        "amortized_min_ms": s[0],
        "amortized_max_ms": s[-1],
        "amortized_stdev_ms": st.pstdev(s) if len(s) > 1 else 0.0,
        "amortized_spread_pct": (s[-1] - s[0]) / st.median(s) * 100.0 if st.median(s) else 0.0,
        "repeats": len(s),
    }


def measure(model, cfg, mode, rung, args, arch, device="cuda"):
    max_cache_len = cfg.ctx + headroom_slots(args)
    row = {
        **cfg.as_row(), "mode": mode, "rung": rung.label,
        "description": rung.description,
        "numerically_valid": rung.numerically_valid,
        "layer_types_forced": True,
        "max_cache_len": max_cache_len,
        "amortized_step_ms": "", "synchronized_step_ms": "", "error": "",
    }
    cache = None
    try:
        _reset_compile_counters()
        cache = dl.build_cache(model, cfg, max_cache_len, device)
        dl.prefill(model, cache, cfg, device)
        dl.assert_static_addresses(cache)
        with apply_rung(rung.label, arch):
            callable_ = dl.build_callable(model, mode)
            step = dl.make_step(callable_, cache, cfg, cfg.ctx, device)
            graphs_before, _ = _compiled_evidence()
            samples = []
            with torch.inference_mode():
                for _ in range(args.repeats):
                    dl.reset_counters(cache, cfg.ctx)
                    samples.append(time_amortized_call(step, args.warmup, args.iters))
                    assert_no_roll(cache)
                dl.reset_counters(cache, cfg.ctx)
                synchronized = time_synchronized_call(
                    step, args.warmup, max(20, args.iters // 5))
                assert_no_roll(cache)
            graphs, calls_captured = _compiled_evidence()
            graphs -= graphs_before
            if mode != dl.HF_EAGER and graphs == 0:
                raise RuntimeError(
                    f"{mode}/{rung.label} captured no graph -- dynamo silently fell back to "
                    "eager and this row would corrupt the gate")
            row.update({
                **repeat_stats(samples),
                "synchronized_step_ms": synchronized,
                "unique_graphs": graphs,
                "calls_captured": calls_captured,
                "cudagraph_manager": _cudagraph_manager_exists(),
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                "error": "",
            })
    except Exception as e:
        detail = " ".join(str(e).split())[:200]
        row.update({"amortized_step_ms": "", "synchronized_step_ms": "",
                    "error": f"{type(e).__name__}: {detail}"})
        print(f"    ERROR {type(e).__name__}: {detail}", flush=True)
    dl.release(cache)
    return row


def env_path(out):
    return os.path.splitext(out)[0] + ".env.json"


def _write(args, rows, meta, complete=False):
    """CSV and its provenance move together.

    The CSV was already written per config so a late crash could not cost the earlier ones,
    but env.json was written only at the end. A run killed in between therefore left rows
    from this run beside an env.json describing the previous one -- and nothing in the file
    said so. `complete` is what tells a reader which case they are holding.
    """
    write_csv(args.out, rows)
    write_json(env_path(args.out),
               {"environment": meta,
                "complete": complete,
                "rows_written": len(rows),
                "rungs": [{"rung": r.label, "description": r.description,
                           "numerically_valid": r.numerically_valid} for r in RUNGS]})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3,
                    help="re-times the same compiled callable; the gate turns on differences of "
                         "under 1%% of a step, which is not obviously above run-to-run noise")
    ap.add_argument("--model", default=dl.DEFAULT_MODEL)
    ap.add_argument("--modes", nargs="+", default=list(dl.MODES))
    ap.add_argument("--rungs", nargs="+", default=[r.label for r in RUNGS])
    ap.add_argument("--batches", type=int, nargs="+", default=None)
    ap.add_argument("--ctxs", type=int, nargs="+", default=None)
    # Calibrated against the observed OOM: b=32 ctx=1024 had 20.10 GB allocated and 23.17 GB in
    # use when it died, so workspace + fragmentation is ~3.0 GB on top of weights + KV.
    ap.add_argument("--mem-budget-gb", type=float, default=22.5)
    ap.add_argument("--activation-reserve-gb", type=float, default=3.0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results/raw/amdahl_probe.csv"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")

    configs = PROBE_CONFIGS
    if args.batches or args.ctxs:
        batches = args.batches or [32]
        ctxs = args.ctxs or [1024]
        configs = [dl.DecodeConfig(args.model, b, c) for b in batches for c in ctxs]
    rungs = resolve_rungs(args.rungs)

    print(f"loading {args.model} ...", flush=True)
    model = dl.load_model(dl.DecodeConfig(args.model, 1, 1), device="cuda")
    arch = assert_patch_target_reachable(model)
    text = model.config.get_text_config(decoder=True)
    print(f"  {text.num_hidden_layers} layers, sliding_window={text.sliding_window}, "
          f"weights {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    print(f"  arch {arch.model_type} -> {arch.module.__name__}, "
          f"op_replaced_ref {'supported' if arch.supports_ref_replacement else 'UNSUPPORTED'}",
          flush=True)

    weights_gb = torch.cuda.memory_allocated() / 1e9
    meta = env_metadata(0, cli_args=vars(args), extra={
        "model_id": args.model,
        "sliding_window_disabled": True,
        "arch_module": arch.module.__name__,
    })
    rows = []
    for cfg in configs:
        print(f"\n=== {cfg.label()}", flush=True)
        max_cache_len = cfg.ctx + headroom_slots(args)
        kv_gb = dl.footprint_bytes(model.config, cfg, max_cache_len) / 1e9
        need = weights_gb + kv_gb + args.activation_reserve_gb
        if need > args.mem_budget_gb:
            reason = (f"weights {weights_gb:.1f} + KV {kv_gb:.1f} + reserve "
                      f"{args.activation_reserve_gb:.1f} = {need:.1f} GB > "
                      f"{args.mem_budget_gb:.1f} GB budget")
            print(f"  SKIPPED: {reason}", flush=True)
            rows.append({**cfg.as_row(), "mode": "", "rung": "skipped",
                         "description": "not run", "numerically_valid": False,
                         "layer_types_forced": True, "max_cache_len": max_cache_len,
                         "amortized_step_ms": "", "synchronized_step_ms": "",
                         "error": reason})
            _write(args, rows, meta)
            continue
        for mode in args.modes:
            baseline = None
            for rung in rungs:
                row = measure(model, cfg, mode, rung, args, arch)
                rows.append(row)
                ms = row.get("amortized_step_ms")
                if not ms:
                    continue
                if rung.label == FULL:
                    baseline = ms
                saving = "" if not baseline else f"{(baseline - ms) / baseline * 100:+6.2f}%"
                row["saving_vs_full_pct"] = (
                    "" if not baseline else (baseline - ms) / baseline * 100)
                row["per_layer_us"] = ms * 1000.0 / text.num_hidden_layers
                print(f"  {mode:16s} {rung.label:16s} {ms:8.3f} ms/step  {saving}"
                      f"  (spread {row['amortized_spread_pct']:.2f}%)", flush=True)
        # written per config: a crash in a later config must not cost the earlier ones
        _write(args, rows, meta)

    record_gpu_state_end(meta, 0)
    _write(args, rows, meta, complete=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
