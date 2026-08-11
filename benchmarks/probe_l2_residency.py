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
    logical_bytes,
    logical_eff_gbps,
    ordering_drift,
    summarize_device_samples,
    time_device_events,
)
from benchmarks.impls import resolve_impls
from benchmarks.workload import Config, build_op_args

SLOT_LADDER = (1, 8, 32, 128, 512)

PROBE_CONFIGS = [
    Config("mha", 32, 32, 128, 128, 2048, "fp32", pos.RAGGED),
    Config("mha", 32, 32, 128, 128, 2048, "bf16", pos.RAGGED),
    Config("mha96", 32, 32, 96, 128, 2048, "bf16", pos.RAGGED),
    Config("gqa", 32, 8, 128, 128, 2048, "bf16", pos.RAGGED),
    Config("mqa", 71, 1, 64, 128, 2048, "bf16", pos.RAGGED),
]


def disjoint_slot_sets(num_requests, cache_alloc_len, n_sets, seed, device):
    """n_sets position tensors drawn from pairwise-disjoint slot ranges.

    The baseline sweep cycles 8 sets, so the timed loop rewrites the same slots forever and
    they stay L2-resident. Disjoint ranges make the write working set num_requests * n_sets
    slots, which is the quantity this probe sweeps across the L2 boundary.
    """
    if n_sets > cache_alloc_len:
        raise ValueError(f"{n_sets} sets needs cache_alloc_len >= {n_sets}")
    chunk = cache_alloc_len // n_sets
    g = torch.Generator().manual_seed(seed)
    sets = []
    for i in range(n_sets):
        offset = torch.randint(0, chunk, (num_requests,), generator=g)
        sets.append((i * chunk + offset).to(device=device, dtype=torch.long))
    return sets


def write_working_set_bytes(cfg, n_sets):
    """Distinct K+V cache bytes the timed loop touches. Reads are the same tensors every
    invocation, so only the scattered writes grow with n_sets."""
    return (2 * cfg.num_requests * n_sets * cfg.num_kv_heads * cfg.head_dim
            * torch.empty(0, dtype=cfg.dtype).element_size())


def probe(cfg, specs, device, seed, warmup, iters, l2_bytes):
    args = build_op_args(cfg, device, seed, None)
    lb = logical_bytes(args.q, args.k, args.v, args.cos, args.sin,
                       args.k_cache, args.v_cache)
    rows = []
    ladder = bracketed([n for n in SLOT_LADDER if n <= cfg.cache_alloc_len])
    for rung_index, n_sets in enumerate(ladder):
        if n_sets > cfg.cache_alloc_len:
            continue
        position_sets = disjoint_slot_sets(cfg.num_requests, cfg.cache_alloc_len,
                                           n_sets, seed, device)
        ws = write_working_set_bytes(cfg, n_sets)
        for spec in specs:
            runner = spec.build()
            with torch.inference_mode():
                thunk = runner.make_thunk(args._replace(positions=position_sets[0]),
                                          position_sets)
                stats = summarize_device_samples(time_device_events(thunk, warmup, iters))
            rows.append({
                **cfg.as_row(), "impl": spec.label, "slot_sets": n_sets,
                "rung_index": rung_index,
                "is_baseline_rung": rung_index in (0, len(ladder) - 1),
                "write_working_set_bytes": ws,
                "write_working_set_mb": round(ws / 1e6, 2),
                "l2_multiple": round(ws / l2_bytes, 3),
                "device_median_ms": stats["device_median_ms"],
                "device_min_ms": stats["device_min_ms"],
                "logical_total_bytes": lb["total_bytes"],
                "logical_eff_gbps": logical_eff_gbps(lb["total_bytes"],
                                                     stats["device_median_ms"]),
            })
            if hasattr(runner, "release"):
                runner.release()
            del runner
            gc.collect()
    del args
    gc.collect()
    torch.cuda.empty_cache()
    return ordering_drift(rows, "device_median_ms")


def main():
    ap = argparse.ArgumentParser(
        description="Does latency follow cache-write traffic once the writes leave L2?")
    ap.add_argument("--impls", nargs="+", default=["compile", "graph_compile"])
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--footprint-budget-gb", type=float, default=12.0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results/raw/l2_residency_probe.csv"))
    ap.add_argument("--resume", action="store_true", help=RESUME_HELP)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    specs = resolve_impls(args.impls)
    device = "cuda"
    l2 = torch.cuda.get_device_properties(device).L2_cache_size
    print(f"# L2 = {l2/1e6:.1f} MB | impls {[s.label for s in specs]}")

    run = ResumableRun(args.out, env_metadata(0, cli_args=vars(args)), resume=args.resume,
                       sidecar={"l2_cache_bytes": l2, "slot_ladder": list(SLOT_LADDER)})
    for cfg in PROBE_CONFIGS:
        fp = cache_footprint_bytes(cfg.num_requests, cfg.cache_alloc_len, cfg.num_kv_heads,
                                   cfg.head_dim, cfg.dtype)
        if fp / 1e9 > args.footprint_budget_gb:
            print(f"SKIPPED {cfg.label()}: one cache set is {fp/1e9:.1f} GB", flush=True)
            continue
        if run.done(cfg.label()):
            print(f"\n=== {cfg.label()}  already measured, skipping", flush=True)
            continue
        print(f"\n=== {cfg.label()}  (one cache set {fp/1e9:.2f} GB)", flush=True)
        # the whole ladder is one unit: ordering_drift reads its closing rung against its
        # opening one, and a restart between them would measure the restart
        for r in run.add(cfg.label(),
                         probe(cfg, specs, device, args.seed, args.warmup, args.iters, l2)):
            print(f"  {r['impl']:14s} sets={r['slot_sets']:>4}  "
                  f"writes={r['write_working_set_mb']:>8.2f} MB ({r['l2_multiple']:>6.2f}x L2)  "
                  f"median={r['device_median_ms']*1000:>7.2f} us  "
                  f"{r['logical_eff_gbps']:>7.1f} GB/s", flush=True)

    run.finish()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
