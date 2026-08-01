import argparse
import json
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import positions as pos
from benchmarks.benchmark_utils import REPO_ROOT, env_metadata, sync, write_json
from benchmarks.impls import IMPL_LABELS, resolve_impls
from benchmarks.validation import validate_candidate
from benchmarks.workload import (
    Config,
    build_op_args,
    build_position_sets,
)

PROFILE_DIR = os.path.join(REPO_ROOT, "results/profiling")

# Representative decode points: single-request and a batched step, MHA and GQA.
REPRESENTATIVE = [
    Config("mha", 32, 32, 128, 1, 2048, "bf16", pos.RAGGED),
    Config("mha", 32, 32, 128, 32, 2048, "bf16", pos.RAGGED),
    Config("gqa", 32, 8, 128, 1, 2048, "bf16", pos.RAGGED),
    Config("gqa", 32, 8, 128, 32, 2048, "bf16", pos.RAGGED),
]

DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")


def summarize_trace(trace_path, iters):
    """Kernel-level launch structure straight from the exported trace."""
    with open(trace_path) as f:
        trace = json.load(f)
    events = [e for e in trace.get("traceEvents", [])
              if e.get("cat") in DEVICE_CATEGORIES and e.get("ph") == "X"]
    per_name = {}
    for e in events:
        rec = per_name.setdefault(e.get("name", "?"), {"count": 0, "total_us": 0.0})
        rec["count"] += 1
        rec["total_us"] += float(e.get("dur", 0.0))
    total_us = sum(r["total_us"] for r in per_name.values())
    kernels = [e for e in events if e.get("cat") == "kernel"]
    return {
        "device_events_total": len(events),
        "kernel_events_total": len(kernels),
        "kernels_per_invocation": len(kernels) / iters if iters else 0,
        "device_time_total_us": total_us,
        "device_time_per_invocation_us": total_us / iters if iters else 0,
        "distinct_kernels": len(per_name),
        "by_kernel": sorted(
            [{"name": n, "count": r["count"], "total_us": r["total_us"],
              "per_invocation": r["count"] / iters if iters else 0}
             for n, r in per_name.items()],
            key=lambda d: -d["total_us"]),
    }


def _op_table(prof, top=8):
    rows = []
    for evt in prof.key_averages():
        self_dev = getattr(evt, "self_device_time_total", None)
        if self_dev is None:
            self_dev = getattr(evt, "self_cuda_time_total", 0)
        total_dev = getattr(evt, "device_time_total", None)
        if total_dev is None:
            total_dev = getattr(evt, "cuda_time_total", 0)
        if self_dev or total_dev:
            rows.append({"op": evt.key, "count": evt.count,
                         "self_device_us": float(self_dev),
                         "total_device_us": float(total_dev)})
    rows.sort(key=lambda r: -r["self_device_us"])
    return rows[:top]


def profile_one(cfg, spec, device, seed, warmup, iters, outdir,
                compile_backend="inductor", compile_mode=None):
    impl_label = spec.label
    runner = spec.build(compile_mode, compile_backend)
    report = validate_candidate(runner, cfg, device, seed)
    if not report["ok"]:
        raise SystemExit(f"refusing to profile unvalidated config {cfg.label()} "
                         f"[{impl_label}]: {'; '.join(report['failures'])}")

    position_sets = build_position_sets(cfg, seed, device)
    args = build_op_args(cfg, device, seed, position_sets[0])
    thunk = runner.make_thunk(args, position_sets)

    for _ in range(warmup):
        thunk()
    sync()

    name = f"{impl_label}_{cfg.label()}"
    trace_path = os.path.join(outdir, f"trace_{name}.json")
    os.makedirs(outdir, exist_ok=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False, acc_events=True) as prof:
        for _ in range(iters):
            thunk()
        sync()
    prof.export_chrome_trace(trace_path)

    summary = summarize_trace(trace_path, iters)
    summary.update({
        "impl": impl_label, "config": cfg.label(), **cfg.as_row(),
        "profiled_iters": iters, "warmup": warmup,
        "trace": os.path.relpath(trace_path, REPO_ROOT),
        "top_ops_by_self_device_time": _op_table(prof),
    })
    return summary


def main():
    ap = argparse.ArgumentParser(description="Profiler evidence for eager vs compiled operator")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--outdir", default=PROFILE_DIR)
    ap.add_argument("--compile-backend", default="inductor")
    ap.add_argument("--compile-mode", default=None)
    ap.add_argument("--impls", nargs="+", default=["eager", "compile"],
                    help=f"subset of {list(IMPL_LABELS)}")
    ap.add_argument("--single", metavar="HEAD_LABEL",
                    help="profile one config only (for wrapping under nsys): mha|gqa")
    ap.add_argument("--single-batch", type=int, default=32)
    ap.add_argument("--no-trace", action="store_true",
                    help="run the workload without the PyTorch profiler (nsys wraps it instead)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    device = "cuda"

    try:
        specs = resolve_impls(args.impls)
    except ValueError as e:
        raise SystemExit(str(e))

    if args.single:
        cfgs = [c for c in REPRESENTATIVE
                if c.head_label == args.single and c.num_requests == args.single_batch]
        if not cfgs:
            raise SystemExit(f"no representative config for {args.single} b={args.single_batch}")
    else:
        cfgs = REPRESENTATIVE

    if args.no_trace:
        for cfg in cfgs:
            for spec in specs:
                runner = spec.build(args.compile_mode, args.compile_backend)
                position_sets = build_position_sets(cfg, args.seed, device)
                op_args = build_op_args(cfg, device, args.seed, position_sets[0])
                thunk = runner.make_thunk(op_args, position_sets)
                for _ in range(args.warmup):
                    thunk()
                sync()
                torch.cuda.nvtx.range_push(f"{spec.label}:{cfg.label()}")
                for _ in range(args.iters):
                    thunk()
                sync()
                torch.cuda.nvtx.range_pop()
                print(f"ran {spec.label} {cfg.label()} x{args.iters}")
        return

    summaries = []
    for cfg in cfgs:
        for spec in specs:
            impl = spec.label
            s = profile_one(cfg, spec, device, args.seed, args.warmup, args.iters,
                            args.outdir, args.compile_backend, args.compile_mode)
            summaries.append(s)
            print(f"{impl:8s} {cfg.label():34s} "
                  f"kernels/invocation={s['kernels_per_invocation']:.2f} "
                  f"distinct={s['distinct_kernels']} "
                  f"device_us/invocation={s['device_time_per_invocation_us']:.1f}")

    out = os.path.join(args.outdir, "profile_summary.json")
    write_json(out, {"environment": env_metadata(0, cli_args=vars(args)),
                     "summaries": summaries})

    print("\nlaunch structure (kernels per invocation):")
    for s in summaries:
        print(f"  {s['impl']:8s} {s['config']:34s} {s['kernels_per_invocation']:.2f}")
    print(f"\nwrote {out}")
    print(f"traces in {args.outdir}/trace_*.json (open in chrome://tracing or perfetto.dev)")


if __name__ == "__main__":
    main()
