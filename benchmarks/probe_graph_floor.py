import argparse
import gc
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import positions as pos
from benchmarks.benchmark_utils import (
    REPO_ROOT,
    cache_footprint_bytes,
    env_metadata,
    summarize_device_samples,
    time_amortized_call,
    time_device_events,
    time_synchronized_call,
    write_csv,
    write_json,
)
from benchmarks.impls import GraphRunner, base_callable
from benchmarks.workload import Config, build_op_args, build_position_sets

PROBE_CONFIGS = [
    Config("mha", 32, 32, 128, 1, 2048, "bf16", pos.RAGGED),
    Config("mha", 32, 32, 128, 32, 2048, "bf16", pos.RAGGED),
    Config("mha", 32, 32, 128, 128, 2048, "bf16", pos.RAGGED),
    Config("mha", 32, 32, 128, 128, 2048, "fp32", pos.RAGGED),
    Config("mqa", 71, 1, 64, 1, 2048, "bf16", pos.RAGGED),
]

# What each stage adds to the one above it. graph_compile is the last row; everything above is
# the cost it carries before any operator work happens.
STAGES = (
    ("harness_noop", "CUDA-event bracketing around a Python call that launches nothing"),
    ("positions_copy", "the static-position copy_ alone, one launch"),
    ("minimal_replay", "replay of a graph holding one trivial kernel"),
    ("copy_plus_minimal_replay", "both launches, no operator work"),
    ("real_replay_no_copy", "replay of the operator graph, positions frozen -- NOT SERVABLE"),
    ("graph_compile", "the timed impl: position copy then operator replay"),
)


def _capture(call, warmup=3):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph


def build_stages(cfg, device, seed):
    args = build_op_args(cfg, device, seed, None)
    position_sets = build_position_sets(cfg, seed, device)

    runner = GraphRunner(base_callable("compile"))
    runner(*args._replace(positions=position_sets[0]))
    # Deliberately reaching past the runner's surface: replaying without the position copy is a
    # decomposition, not a workload anyone can serve, so it is not exposed as a public accessor.
    real_graph = runner._graph
    static_positions = runner._static_positions

    scratch = torch.zeros(1, device=device)
    minimal_graph = _capture(lambda: scratch.add_(1.0))

    cyc = {name: itertools.cycle(position_sets) for name, _ in STAGES}
    thunks = {
        "harness_noop": lambda: None,
        "positions_copy": lambda: static_positions.copy_(next(cyc["positions_copy"])),
        "minimal_replay": minimal_graph.replay,
        "real_replay_no_copy": real_graph.replay,
    }

    def copy_plus_minimal():
        static_positions.copy_(next(cyc["copy_plus_minimal_replay"]))
        minimal_graph.replay()

    def full():
        static_positions.copy_(next(cyc["graph_compile"]))
        real_graph.replay()

    thunks["copy_plus_minimal_replay"] = copy_plus_minimal
    thunks["graph_compile"] = full
    return thunks, (args, runner, minimal_graph, scratch)


def probe(cfg, device, seed, warmup, iters):
    thunks, keepalive = build_stages(cfg, device, seed)
    rows = []
    with torch.inference_mode():
        for name, description in STAGES:
            fn = thunks[name]
            stats = summarize_device_samples(time_device_events(fn, warmup, iters))
            rows.append({
                **cfg.as_row(), "stage": name, "description": description,
                "device_median_us": stats["device_median_ms"] * 1000.0,
                "device_min_us": stats["device_min_ms"] * 1000.0,
                "amortized_call_us": time_amortized_call(fn, warmup, iters) * 1000.0,
                "synchronized_call_us": time_synchronized_call(
                    fn, warmup, max(20, iters // 5)) * 1000.0,
            })
    del thunks, keepalive
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="How much of the CUDA-graph floor is work, and how much is measuring it")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--footprint-budget-gb", type=float, default=12.0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results/raw/graph_floor_probe.csv"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    device = "cuda"

    rows = []
    for cfg in PROBE_CONFIGS:
        fp = cache_footprint_bytes(cfg.num_requests, cfg.cache_alloc_len, cfg.num_kv_heads,
                                   cfg.head_dim, cfg.dtype)
        if fp / 1e9 > args.footprint_budget_gb:
            print(f"SKIPPED {cfg.label()}: one cache set is {fp/1e9:.1f} GB", flush=True)
            continue
        print(f"\n=== {cfg.label()}", flush=True)
        config_rows = probe(cfg, device, args.seed, args.warmup, args.iters)
        base = None
        for r in config_rows:
            rows.append(r)
            delta = "" if base is None else f"  (+{r['device_median_us'] - base:6.2f})"
            base = base if base is not None else r["device_median_us"]
            print(f"  {r['stage']:26s} event={r['device_median_us']:6.2f} us"
                  f"  amort={r['amortized_call_us']:6.2f}  "
                  f"sync={r['synchronized_call_us']:6.2f}{delta}", flush=True)
        by = {r["stage"]: r for r in config_rows}
        work = by["graph_compile"]["device_median_us"] - by["copy_plus_minimal_replay"]["device_median_us"]
        harness = by["harness_noop"]["device_median_us"]
        print(f"  -> operator work {work:.2f} us of a "
              f"{by['graph_compile']['device_median_us']:.2f} us call; "
              f"harness floor {harness:.2f} us", flush=True)

    write_csv(args.out, rows)
    write_json(os.path.splitext(args.out)[0] + ".env.json",
               {"environment": env_metadata(0, cli_args=vars(args)),
                "stages": [{"stage": n, "description": d} for n, d in STAGES]})
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
