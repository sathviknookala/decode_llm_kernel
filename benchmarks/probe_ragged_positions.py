import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import positions as pos
from benchmarks.benchmark_utils import (
    REPO_ROOT,
    RESUME_HELP,
    ResumableRun,
    bracketed,
    cache_footprint_bytes,
    env_metadata,
    ordering_drift,
    summarize_device_samples,
    time_amortized_call,
    time_device_events,
)
from benchmarks.impls import resolve_impls
from benchmarks.workload import Config, build_op_args

SHARED = "shared"
DISTINCT = "distinct"
TENSOR_MODES = (SHARED, DISTINCT)

SPREAD_LADDER = (1, 2, 8, 64, 512, 2048)

PROBE_CONFIGS = [
    Config("mha", 32, 32, 128, 1, 2048, "fp32", pos.RAGGED),
    Config("mha", 32, 32, 128, 1, 2048, "bf16", pos.RAGGED),
    Config("mha", 32, 32, 128, 32, 2048, "fp32", pos.RAGGED),
    Config("gqa", 32, 8, 128, 1, 2048, "bf16", pos.RAGGED),
]


def spread_position_sets(num_requests, cache_alloc_len, spread, n_sets, seed, device,
                         tensor_mode=DISTINCT):
    """Position sets drawn from a window of `spread` slots ending at the last valid slot.

    spread=1 with tensor_mode=shared is exactly what the sweep calls uniform mode: one tensor
    object, repeated, every request at cache_alloc_len-1. spread=1 with distinct tensors holds
    those same values in n_sets separate allocations, which is the control the sweep lacks --
    identical memory traffic, identical values, different objects crossing the call boundary.
    """
    if not 1 <= spread <= cache_alloc_len:
        raise ValueError(f"spread {spread} outside [1, {cache_alloc_len}]")
    if tensor_mode not in TENSOR_MODES:
        raise ValueError(f"unknown tensor mode {tensor_mode!r}, expected one of {TENSOR_MODES}")
    if tensor_mode == SHARED and spread != 1:
        raise ValueError("a shared tensor only makes sense at spread=1; above it the sets "
                         "would differ and could not be one object")
    hi = cache_alloc_len - 1
    if tensor_mode == SHARED:
        p = torch.full((num_requests,), hi, dtype=torch.long, device=device)
        return [p] * n_sets
    g = torch.Generator().manual_seed(seed)
    lo = cache_alloc_len - spread
    sets = []
    for _ in range(n_sets):
        if spread == 1:
            vals = torch.full((num_requests,), hi, dtype=torch.long)
        else:
            vals = lo + torch.randint(0, spread, (num_requests,), generator=g)
        sets.append(vals.to(device=device, dtype=torch.long))
    return sets


def rungs(cache_alloc_len):
    """(spread, tensor_mode) pairs. The shared rung is the sweep's uniform baseline; every
    other rung is distinct-tensor, so tensor identity is held fixed across the spread ladder.

    The shared rung is measured twice, first and last. Measured once it is always the first
    rung of the config, and "the first rung is slower" would be indistinguishable from any
    effect of the tensor being shared -- so without the repeat every ratio here has an
    ordering confound sitting underneath it.
    """
    return bracketed([(1, SHARED)]
                     + [(s, DISTINCT) for s in SPREAD_LADDER if s <= cache_alloc_len])


def probe(cfg, specs, device, seed, warmup, iters, n_sets, fresh_args=False):
    """fresh_args reallocates every tensor per rung. The sweep timed uniform and ragged as
    separate configs, so they got separate compiles and separate allocations; holding those
    fixed is the tighter experiment, but if the 1.41x lives there rather than in the positions,
    only the reallocating arm can show it."""
    args = None if fresh_args else build_op_args(cfg, device, seed, None)
    rows = []
    for rung_index, (spread, tensor_mode) in enumerate(rungs(cfg.cache_alloc_len)):
        position_sets = spread_position_sets(cfg.num_requests, cfg.cache_alloc_len, spread,
                                             n_sets, seed, device, tensor_mode)
        for spec in specs:
            if fresh_args:
                args = build_op_args(cfg, device, seed, None)
            runner = spec.build()
            with torch.inference_mode():
                thunk = runner.make_thunk(args._replace(positions=position_sets[0]),
                                          position_sets)
                stats = summarize_device_samples(time_device_events(thunk, warmup, iters))
                amortized = time_amortized_call(thunk, warmup, iters)
            rows.append({
                **cfg.as_row(), "impl": spec.label, "spread": spread,
                "tensor_mode": tensor_mode, "rung_index": rung_index,
                "is_baseline_rung": tensor_mode == SHARED,
                "position_sets": n_sets, "fresh_args": fresh_args,
                "distinct_slots_per_request": min(spread, n_sets),
                "device_median_ms": stats["device_median_ms"],
                "device_min_ms": stats["device_min_ms"],
                "amortized_call_ms": amortized,
            })
            if hasattr(runner, "release"):
                runner.release()
            del runner
            gc.collect()
            if fresh_args:
                torch.cuda.empty_cache()
    del args
    gc.collect()
    torch.cuda.empty_cache()
    return with_ratios(rows)


def with_ratios(rows):
    """Every row against its own impl's opening shared-tensor rung, which is the quantity the
    sweep reported as ragged/uniform.

    ordering_drift is the closing shared rung against the opening one. Both hold the tensor
    and the values fixed, so anything other than 1.0 there is drift over the config's run --
    and it is the amount of any other ratio in the config that ordering alone can explain.
    """
    def opening(impl):
        shared = [r for r in rows if r["impl"] == impl and r["is_baseline_rung"]]
        return min(shared, key=lambda r: r["rung_index"]) if shared else None

    for r in rows:
        b = opening(r["impl"])
        for col in ("device_median_ms", "amortized_call_ms"):
            r[f"{col.replace('_ms', '')}_ratio"] = (
                r[col] / b[col] if b and b[col] else "")
    return ordering_drift(rows, "amortized_call_ms")


def main():
    ap = argparse.ArgumentParser(
        description="Is compile's 1.41x ragged/uniform at b=1 about position values or about "
                    "the tensor carrying them?")
    ap.add_argument("--impls", nargs="+", default=["eager", "compile", "graph_compile"])
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--position-sets", type=int, default=8,
                    help="sets cycled in the timed loop; 8 is what the baseline sweep uses")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fresh-args", action="store_true",
                    help="reallocate every tensor per rung, as separate sweep configs do")
    ap.add_argument("--footprint-budget-gb", type=float, default=12.0)
    ap.add_argument("--out",
                    default=os.path.join(REPO_ROOT, "results/raw/ragged_positions_probe.csv"))
    ap.add_argument("--resume", action="store_true", help=RESUME_HELP)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    specs = resolve_impls(args.impls)
    device = "cuda"
    print(f"# impls {[s.label for s in specs]} | {args.position_sets} position sets cycled")

    run = ResumableRun(args.out, env_metadata(0, cli_args=vars(args)), resume=args.resume,
                       sidecar={"spread_ladder": list(SPREAD_LADDER),
                                "baseline_rung": {"spread": 1, "tensor_mode": SHARED}})
    # the two arms are separate experiments over the same configs, so the arm is part of the key
    arm = "fresh_args" if args.fresh_args else "held_args"
    for cfg in PROBE_CONFIGS:
        fp = cache_footprint_bytes(cfg.num_requests, cfg.cache_alloc_len, cfg.num_kv_heads,
                                   cfg.head_dim, cfg.dtype)
        if fp / 1e9 > args.footprint_budget_gb:
            print(f"SKIPPED {cfg.label()}: one cache set is {fp/1e9:.1f} GB", flush=True)
            continue
        key = f"{cfg.label()}|{arm}"
        if run.done(key):
            print(f"\n=== {cfg.label()}  already measured, skipping", flush=True)
            continue
        print(f"\n=== {cfg.label()}", flush=True)
        # the whole ladder is one unit: ordering_drift reads its closing shared rung against
        # its opening one, and a restart between them would measure the restart
        for r in run.add(key, probe(cfg, specs, device, args.seed, args.warmup, args.iters,
                                    args.position_sets, args.fresh_args)):
            print(f"  {r['impl']:14s} spread={r['spread']:>5} {r['tensor_mode']:8s} "
                  f"median={r['device_median_ms']*1000:>7.2f} us "
                  f"amort={r['amortized_call_ms']*1000:>7.2f} us  "
                  f"ratio={r['amortized_call_ratio']:>6.3f} (amortized)", flush=True)

    run.finish()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
