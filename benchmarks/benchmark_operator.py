import argparse
import gc
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
    gpu_clock_lock,
    logical_bytes,
    logical_eff_gbps,
    record_gpu_state_end,
    summarize_device_samples,
    time_amortized_call,
    time_device_events,
    time_synchronized_call,
    write_csv,
    write_json,
)
from benchmarks.impls import (
    DEFAULT_IMPLS,
    IMPL_LABELS,
    inductor_cudagraph_skips,
    resolve_impls,
)
from benchmarks.validation import validate_candidate
from benchmarks.workload import (
    Config,
    IDENTITY,
    PACKED,
    SERVING_VARIANT_BATCHES,
    build_matrix,
    build_op_args,
    build_position_sets,
)

DEFAULT_FOOTPRINT_BUDGET_GB = 12.0

# Validation briefly holds the oracle's and the candidate's caches at the same time.
PEAK_CACHE_SETS = 2


def bench_impl(runner, args, position_sets, warmup, iters, bw_ref_gbps,
               scattered_ref_gbps=None):
    # Decode serving runs under inference mode; leaving autograd on would charge every impl
    # for version-counter bookkeeping on the in-place cache write.
    with torch.inference_mode():
        thunk = runner.make_thunk(args, position_sets)
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
    row["pct_of_scattered_write_bw"] = (
        100.0 * gbps / scattered_ref_gbps if scattered_ref_gbps else "")
    return row


def _blank_metrics():
    return {k: "" for k in (
        "device_median_ms", "device_p95_ms", "device_min_ms", "device_std_ms",
        "amortized_call_ms", "synchronized_call_ms", "logical_read_bytes",
        "logical_write_bytes", "logical_total_bytes", "logical_eff_gbps",
        "pct_of_empirical_bw", "pct_of_scattered_write_bw")}


NUMERICS_FIELDS = ("max_abs_diff_q", "max_abs_diff_k_cache", "v_byte_exact",
                   "unaddressed_slots_intact", "tolerance_atol", "tolerance_rtol")


def _blank_numerics():
    return {k: "" for k in NUMERICS_FIELDS}


def numerics_row(report):
    """The gate's measured deltas as columns. The locked dtype policy requires reporting
    numerical deltas, not just pass/fail, and a string in validation_detail is not a column
    anything can aggregate over."""
    return {
        "max_abs_diff_q": report["max_abs_diff_q"],
        "max_abs_diff_k_cache": report["max_abs_diff_k_cache"],
        "v_byte_exact": report["v_byte_exact"],
        "unaddressed_slots_intact": report["unaddressed_slots_intact"],
        "tolerance_atol": report["tolerance"]["atol"],
        "tolerance_rtol": report["tolerance"]["rtol"],
    }


def graph_savings(rows):
    config_keys = tuple(Config.__dataclass_fields__)
    timed = {(r["impl"], tuple(r.get(k) for k in config_keys)): r for r in rows
             if r.get("validation") == "pass" and r.get("device_median_ms") != ""}
    savings = []
    for (impl, key), direct in timed.items():
        if impl not in ("eager", "compile"):
            continue
        graph_impl = f"graph_{impl}"
        graph = timed.get((graph_impl, key))
        if graph is None:
            continue
        direct_ms = float(direct["device_median_ms"])
        graph_ms = float(graph["device_median_ms"])
        savings.append({
            "direct_impl": impl,
            "graph_impl": graph_impl,
            "config": dict(zip(config_keys, key)),
            "direct_device_median_ms": direct_ms,
            "graph_device_median_ms": graph_ms,
            "latency_removed_pct": 100.0 * (1.0 - graph_ms / direct_ms),
        })
    return savings


def run_config(cfg, device, args_ns, bw_ref_gbps, specs, scattered_ref_gbps=None):
    """Validate every impl against the oracle, then time only the impls that passed."""
    rows = []
    seed = args_ns.seed
    position_sets = build_position_sets(cfg, seed, device)

    for spec in specs:
        # In-row, not just env.json: with mode variants in one sweep a single run-level setting
        # no longer describes every row, which is how validation_cases went wrong in v2.
        base = {"impl": spec.label, **cfg.as_row(), "seed": seed,
                "compile_mode": "", "compile_backend": "", "inductor_cudagraph_skips": ""}
        runner = None
        try:
            spec_mode, spec_backend = spec.resolve(args_ns.compile_mode, args_ns.compile_backend)
            base.update(compile_mode=spec_mode or "", compile_backend=spec_backend or "")
            skips_before = inductor_cudagraph_skips()
            runner = spec.build(args_ns.compile_mode, args_ns.compile_backend)
            report = validate_candidate(runner, cfg, device, seed)
            if spec_backend is not None:
                base["inductor_cudagraph_skips"] = inductor_cudagraph_skips() - skips_before
        except Exception as e:  # noqa: BLE001 -- a build or capture failure is a result, not a crash
            print(f"  ERROR  {spec.label:14s} {cfg.label()}: {type(e).__name__}: {e}",
                  flush=True)
            rows.append({**base, "validation": "ERROR",
                         "validation_detail": f"{type(e).__name__}: {e}",
                         **_blank_numerics(), **_blank_metrics()})
            _release(runner)
            continue
        if not report["ok"]:
            detail = "; ".join(report["failures"])
            print(f"  VALIDATION FAILED  {spec.label:14s} {cfg.label()}: {detail}", flush=True)
            rows.append({**base, "validation": "FAIL", "validation_detail": detail,
                         "validation_cases": "|".join(report["cases"]),
                         **numerics_row(report), **_blank_metrics()})
            _release(runner)
            continue
        op_args = build_op_args(cfg, device, seed, position_sets[0])
        timed = bench_impl(runner, op_args, position_sets, args_ns.warmup,
                           args_ns.iters, bw_ref_gbps, scattered_ref_gbps)
        del op_args
        rows.append({**base, "validation": "pass",
                     "validation_detail": (f"q_maxdiff={report['max_abs_diff_q']:.2e} "
                                           f"k_maxdiff={report['max_abs_diff_k_cache']:.2e} "
                                           f"cases={report['num_cases']}"),
                     "validation_cases": "|".join(report["cases"]),
                     **numerics_row(report), **timed})
        _release(runner)
    return rows


def _release(runner):
    """A captured graph pins its private memory pool until the runner is collected."""
    if runner is not None and hasattr(runner, "release"):
        runner.release()
    del runner
    gc.collect()
    torch.cuda.empty_cache()


def _run_benchmark(args, device, specs, modes, clock_status):
    meta = env_metadata(args.device_index, cli_args=vars(args), extra={
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "position_modes": list(modes),
        "impls": [s.label for s in specs],
        "compile_backend": args.compile_backend,
        "compile_mode": args.compile_mode,
        "footprint_budget_gb": args.footprint_budget_gb,
        "clock_lock": clock_status,
        "timing_matrix": {
            "main": {"layout": PACKED, "request_mapping": IDENTITY},
            "serving_variant_batches": list(SERVING_VARIANT_BATCHES),
        },
    })

    bw_ref = None
    bw_ref_gbps = None
    scattered_ref_gbps = None
    if not args.skip_bandwidth_ref:
        bw_ref = measure_bandwidth_reference(device, args.bandwidth_buffer_mib)
        bw_ref_gbps = bw_ref["reference_gbps"]
        scattered_ref_gbps = bw_ref["scattered_write_reference_gbps"]
        print(f"# empirical bandwidth reference: {bw_ref_gbps:.1f} GB/s "
              f"({bw_ref['copy']['byte_convention']})")
        if scattered_ref_gbps:
            print(f"# scattered-write reference: {scattered_ref_gbps:.1f} GB/s "
                  f"({bw_ref['scattered_write']['byte_convention']})")
        else:
            print(f"WARNING: scattered-write reference unavailable: "
                  f"{bw_ref['scattered_write']['error']}", file=sys.stderr)
    meta["bandwidth_reference"] = bw_ref

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
                             "compile_mode": "", "compile_backend": "",
                             "validation": "not-run",
                             "inductor_cudagraph_skips": "",
                             "validation_detail": f"peak cache {peak_gb:.1f}GB > budget",
                             **_blank_numerics(), **_blank_metrics()})
            continue
        for r in run_config(cfg, device, args, bw_ref_gbps, specs, scattered_ref_gbps):
            all_rows.append(r)
            if r["validation"] in ("FAIL", "ERROR"):
                n_fail += 1
            elif r["device_median_ms"] != "":
                print(f"{r['impl']:14s} {cfg.label():34s} "
                      f"dev_median={r['device_median_ms']:.4f}ms "
                      f"p95={r['device_p95_ms']:.4f}ms "
                      f"amort={r['amortized_call_ms']:.4f}ms "
                      f"sync={r['synchronized_call_ms']:.4f}ms "
                      f"logical={r['logical_eff_gbps']:.1f}GB/s", flush=True)

    write_csv(args.out, all_rows)
    meta_path = os.path.splitext(args.out)[0] + ".env.json"
    n_timed = sum(1 for r in all_rows if r["validation"] == "pass")
    savings = graph_savings(all_rows)
    record_gpu_state_end(meta, args.device_index)
    write_json(meta_path, {"environment": meta, "summary": {
        "rows_total": len(all_rows), "rows_timed": n_timed,
        "validation_failures": n_fail, "configs_skipped": n_skip,
        # Which gate cleared these rows, by name. A count alone let the gate change under a
        # committed CSV once already.
        "validation_case_sets": sorted({r["validation_cases"] for r in all_rows
                                        if r.get("validation_cases")}),
        "graph_savings": savings}})

    print(f"\ntimed rows: {n_timed} | validation failures: {n_fail} | skipped configs: {n_skip}")
    if savings:
        print("CUDA graph device-median latency removed:")
        for s in savings:
            cfg = s["config"]
            variant = ""
            if (cfg["layout"], cfg["request_mapping"]) != (PACKED, IDENTITY):
                variant = f" {cfg['layout']}/{cfg['request_mapping']}"
            print(f"  {s['graph_impl']} vs {s['direct_impl']}: "
                  f"{s['latency_removed_pct']:.1f}% at {cfg['head_label']} "
                  f"b={cfg['num_requests']} alloc={cfg['cache_alloc_len']} "
                  f"{cfg['dtype_label']} {cfg['position_mode']}{variant}")
    print(f"wrote {args.out}\nwrote {meta_path}")
    if n_fail:
        print(f"\nFAILED: {n_fail} configuration(s) did not validate; their timings were "
              f"not recorded.", file=sys.stderr)
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Operator benchmark: eager vs torch.compile fused RoPE + KV-append")
    ap.add_argument("--quick", action="store_true", help="reduced matrix for a smoke run")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--position-mode", choices=["uniform", "ragged", "both"], default="both")
    ap.add_argument("--impls", nargs="+", default=list(DEFAULT_IMPLS),
                    help=f"subset of {list(IMPL_LABELS)}")
    ap.add_argument("--compile-backend", default="inductor")
    ap.add_argument("--compile-mode", default=None)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--lock-clocks", action="store_true",
                    help="attempt to lock the selected GPU at its maximum SM clock")
    ap.add_argument("--footprint-budget-gb", type=float, default=DEFAULT_FOOTPRINT_BUDGET_GB)
    ap.add_argument("--bandwidth-buffer-mib", type=int, default=512)
    ap.add_argument("--skip-bandwidth-ref", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results/raw/operator_baseline_v2.csv"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for benchmarking")
    torch.cuda.set_device(args.device_index)
    device = f"cuda:{args.device_index}"

    try:
        specs = resolve_impls(args.impls)
    except ValueError as e:
        raise SystemExit(str(e))

    modes = (pos.UNIFORM, pos.RAGGED) if args.position_mode == "both" else (args.position_mode,)
    with gpu_clock_lock(args.device_index, args.lock_clocks) as clock_status:
        _run_benchmark(args, device, specs, modes, clock_status)


if __name__ == "__main__":
    main()
