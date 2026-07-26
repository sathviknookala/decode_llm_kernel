import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import positions as pos
from benchmarks.bandwidth_reference import measure_bandwidth_reference
from benchmarks.benchmark_utils import (
    REPO_ROOT,
    cache_footprint_bytes,
    env_metadata,
    logical_bytes,
    logical_eff_gbps,
    summarize_device_samples,
    time_amortized_call,
    time_device_events,
    time_synchronized_call,
    write_csv,
    write_json,
)
from benchmarks.validation import validate_candidate
from benchmarks.workload import (
    build_matrix,
    build_op_args,
    build_position_sets,
    compile_impl,
    eager_impl,
    make_thunk,
)

DEFAULT_FOOTPRINT_BUDGET_GB = 12.0

# Validation briefly holds the oracle's and the candidate's caches at the same time.
PEAK_CACHE_SETS = 2


def bench_impl(fn, args, position_sets, warmup, iters, bw_ref_gbps):
    thunk = make_thunk(fn, args, position_sets)
    stats = summarize_device_samples(time_device_events(thunk, warmup, iters))
    amortized = time_amortized_call(thunk, warmup, iters)
    synchronized = time_synchronized_call(thunk, warmup, max(10, iters // 5))
    lb = logical_bytes(args.q, args.k, args.v, args.cos, args.sin,
                       args.k_cache, args.v_cache)
    gbps = logical_eff_gbps(lb["total_bytes"], stats["device_median_ms"])
    row = dict(stats)
    row["amortized_call_ms"] = amortized
    row["synchronized_call_ms"] = synchronized
    row["logical_read_bytes"] = lb["read_bytes"]
    row["logical_write_bytes"] = lb["write_bytes"]
    row["logical_total_bytes"] = lb["total_bytes"]
    row["logical_eff_gbps"] = gbps
    row["pct_of_empirical_bw"] = (100.0 * gbps / bw_ref_gbps) if bw_ref_gbps else ""
    return row


def _blank_metrics():
    return {k: "" for k in (
        "device_median_ms", "device_p95_ms", "device_min_ms", "device_std_ms",
        "amortized_call_ms", "synchronized_call_ms", "logical_read_bytes",
        "logical_write_bytes", "logical_total_bytes", "logical_eff_gbps",
        "pct_of_empirical_bw")}


def run_config(cfg, device, args_ns, bw_ref_gbps):
    """Validate every impl against the oracle, then time only the impls that passed."""
    rows = []
    seed = args_ns.seed
    position_sets = build_position_sets(cfg, seed, device)

    impls = [("eager", eager_impl())]
    compile_error = None
    try:
        impls.append(("compile", compile_impl(mode=args_ns.compile_mode,
                                              backend=args_ns.compile_backend)))
    except Exception as e:  # noqa: BLE001
        compile_error = f"{type(e).__name__}: {e}"

    if compile_error:
        rows.append({"impl": "compile", **cfg.as_row(), "seed": seed,
                     "validation": "ERROR", "validation_detail": compile_error,
                     **_blank_metrics()})

    for label, fn in impls:
        report = validate_candidate(fn, cfg, device, seed)
        base = {"impl": label, **cfg.as_row(), "seed": seed}
        if not report["ok"]:
            detail = "; ".join(report["failures"])
            print(f"  VALIDATION FAILED  {label:8s} {cfg.label()}: {detail}", flush=True)
            rows.append({**base, "validation": "FAIL", "validation_detail": detail,
                         **_blank_metrics()})
            continue
        op_args = build_op_args(cfg, device, seed, position_sets[0])
        timed = bench_impl(fn, op_args, position_sets, args_ns.warmup,
                           args_ns.iters, bw_ref_gbps)
        del op_args
        rows.append({**base, "validation": "pass",
                     "validation_detail": (f"q_maxdiff={report['max_abs_diff_q']:.2e} "
                                           f"k_maxdiff={report['max_abs_diff_k_cache']:.2e} "
                                           f"cases={report['num_cases']}"),
                     **timed})
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Operator benchmark: eager vs torch.compile fused RoPE + KV-append")
    ap.add_argument("--quick", action="store_true", help="reduced matrix for a smoke run")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--position-mode", choices=["uniform", "ragged", "both"], default="both")
    ap.add_argument("--compile-backend", default="inductor")
    ap.add_argument("--compile-mode", default=None)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--footprint-budget-gb", type=float, default=DEFAULT_FOOTPRINT_BUDGET_GB)
    ap.add_argument("--bandwidth-buffer-mib", type=int, default=512)
    ap.add_argument("--skip-bandwidth-ref", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results/raw/operator_baseline_v2.csv"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for benchmarking")
    torch.cuda.set_device(args.device_index)
    device = f"cuda:{args.device_index}"

    modes = (pos.UNIFORM, pos.RAGGED) if args.position_mode == "both" else (args.position_mode,)

    bw_ref = None
    bw_ref_gbps = None
    if not args.skip_bandwidth_ref:
        bw_ref = measure_bandwidth_reference(device, args.bandwidth_buffer_mib)
        bw_ref_gbps = bw_ref["reference_gbps"]
        print(f"# empirical bandwidth reference: {bw_ref_gbps:.1f} GB/s "
              f"({bw_ref['copy']['byte_convention']})")

    meta = env_metadata(args.device_index, cli_args=vars(args), extra={
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "position_modes": list(modes),
        "compile_backend": args.compile_backend,
        "compile_mode": args.compile_mode,
        "footprint_budget_gb": args.footprint_budget_gb,
        "bandwidth_reference": bw_ref,
    })
    print(f"# {meta['gpu_name']} cc{meta['gpu_compute_capability']} | "
          f"torch {meta['torch_version']} | driver {meta['nvidia_driver_version']} | "
          f"nvcc {meta['cuda_toolkit_version_nvcc']} | git {meta['git_sha']}"
          f"{' (dirty)' if meta['git_dirty'] else ''}")

    all_rows = []
    n_fail = 0
    n_skip = 0
    for cfg in build_matrix(args.quick, position_modes=modes):
        fp_gb = cache_footprint_bytes(cfg.num_requests, cfg.cache_alloc_len,
                                      cfg.num_kv_heads, cfg.head_dim, cfg.dtype) / 1e9
        peak_gb = PEAK_CACHE_SETS * fp_gb
        if peak_gb > args.footprint_budget_gb:
            n_skip += 1
            print(f"SKIPPED {cfg.label()}: peak contiguous cache {peak_gb:.1f} GB "
                  f"({fp_gb:.1f} GB/set x {PEAK_CACHE_SETS}) > "
                  f"{args.footprint_budget_gb} GB budget", flush=True)
            all_rows.append({"impl": "skipped", **cfg.as_row(), "seed": args.seed,
                             "validation": "not-run",
                             "validation_detail": f"peak cache {peak_gb:.1f}GB > budget",
                             **_blank_metrics()})
            continue
        for r in run_config(cfg, device, args, bw_ref_gbps):
            all_rows.append(r)
            if r["validation"] in ("FAIL", "ERROR"):
                n_fail += 1
            elif r["device_median_ms"] != "":
                print(f"{r['impl']:8s} {cfg.label():34s} "
                      f"dev_median={r['device_median_ms']:.4f}ms "
                      f"p95={r['device_p95_ms']:.4f}ms "
                      f"amort={r['amortized_call_ms']:.4f}ms "
                      f"sync={r['synchronized_call_ms']:.4f}ms "
                      f"logical={r['logical_eff_gbps']:.1f}GB/s", flush=True)

    write_csv(args.out, all_rows)
    meta_path = os.path.splitext(args.out)[0] + ".env.json"
    n_timed = sum(1 for r in all_rows if r["validation"] == "pass")
    write_json(meta_path, {"environment": meta, "summary": {
        "rows_total": len(all_rows), "rows_timed": n_timed,
        "validation_failures": n_fail, "configs_skipped": n_skip}})

    print(f"\ntimed rows: {n_timed} | validation failures: {n_fail} | skipped configs: {n_skip}")
    print(f"wrote {args.out}\nwrote {meta_path}")
    if n_fail:
        print(f"\nFAILED: {n_fail} configuration(s) did not validate; their timings were "
              f"not recorded.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
