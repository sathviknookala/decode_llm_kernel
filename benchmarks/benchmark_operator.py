import argparse
import itertools
import sys
from dataclasses import dataclass, asdict

import torch
import torch._dynamo

sys.path.insert(0, __file__.rsplit("/benchmarks/", 1)[0])

from decode_kernels.reference import build_rope_tables, fused_rope_kv_append_ref
from benchmarks.benchmark_utils import (
    cache_footprint_bytes,
    eff_gbps,
    env_metadata,
    logical_bytes,
    summarize,
    time_cuda_events,
    time_op_call,
    write_csv,
)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# Llama-2-7B anchor (MHA) plus a GQA variant; head_dim fixed at the anchor's 128.
HEAD_CONFIGS = [
    ("mha", 32, 32, 128),
    ("gqa", 32, 8, 128),
]

FOOTPRINT_BUDGET_GB = 6.0


@dataclass
class Config:
    head_label: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_requests: int   # active decode requests = num_tokens
    context_len: int    # decode position; also the contiguous-cache max_seq
    dtype_label: str

    @property
    def dtype(self):
        return DTYPES[self.dtype_label]


def build_inputs(cfg, device):
    T, Hq, Hkv, D = cfg.num_requests, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim
    dt = cfg.dtype
    max_seq = cfg.context_len
    q = torch.randn(T, Hq, D, dtype=dt, device=device)
    k = torch.randn(T, Hkv, D, dtype=dt, device=device)
    v = torch.randn(T, Hkv, D, dtype=dt, device=device)
    cos, sin = build_rope_tables(max_seq, D, theta=10000.0, device=device,
                                 dtype=torch.float32)
    positions = torch.full((T,), max_seq - 1, dtype=torch.long, device=device)
    request_indices = torch.arange(T, device=device)
    k_cache = torch.zeros(T, max_seq, Hkv, D, dtype=dt, device=device)
    v_cache = torch.zeros(T, max_seq, Hkv, D, dtype=dt, device=device)
    return (q, k, v, positions, cos, sin, k_cache, v_cache, request_indices)


def make_thunk(fn, args):
    def run():
        fn(*args)
    return run


def bench_one(cfg, device, warmup, iters):
    args = build_inputs(cfg, device)
    rows = []
    impls = [("eager", fused_rope_kv_append_ref)]
    try:
        torch._dynamo.reset()  # isolate each config's compile; avoid recompile-limit fallback
        compiled = torch.compile(fused_rope_kv_append_ref, dynamic=False)
        compiled(*args)  # trigger compilation now (excluded from timing)
        torch.cuda.synchronize()
        impls.append(("compile", compiled))
    except Exception as e:  # noqa: BLE001 - record, don't crash the sweep
        rows.append(_row(cfg, "compile", None, None, note=f"compile-failed: {type(e).__name__}"))

    lb = logical_bytes(cfg.num_requests, cfg.num_q_heads, cfg.num_kv_heads,
                       cfg.head_dim, cfg.dtype)
    for label, fn in impls:
        thunk = make_thunk(fn, args)
        stats = summarize(time_cuda_events(thunk, warmup, iters))
        op_ms = time_op_call(thunk, warmup, iters)
        rows.append(_row(cfg, label, stats, op_ms, lb=lb))
    return rows


def _row(cfg, impl, stats, op_ms, lb=None, note=""):
    row = {"impl": impl, **{k: v for k, v in asdict(cfg).items()}}
    if stats is None:
        row.update({"median_ms": "", "p50_ms": "", "p95_ms": "", "min_ms": "",
                    "std_ms": "", "op_call_ms": "", "eff_gbps": "", "note": note})
    else:
        row.update(stats)
        row["op_call_ms"] = op_ms
        row["eff_gbps"] = eff_gbps(lb, stats["median_ms"])
        row["note"] = note
    return row


def build_matrix(quick):
    batches = [1, 8, 32] if quick else [1, 8, 32, 128]
    contexts = [2048] if quick else [128, 2048]
    dtypes = ["bf16", "fp16", "fp32"]
    heads = HEAD_CONFIGS[:1] if quick else HEAD_CONFIGS
    for (hl, hq, hkv, hd), b, ctx, dl in itertools.product(heads, batches, contexts, dtypes):
        yield Config(hl, hq, hkv, hd, b, ctx, dl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small matrix for a smoke run")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for benchmarking")
    device = "cuda"
    meta = env_metadata()
    print(f"# {meta}")

    all_rows = []
    for cfg in build_matrix(args.quick):
        fp_gb = cache_footprint_bytes(cfg.num_requests, cfg.context_len,
                                      cfg.num_kv_heads, cfg.head_dim, cfg.dtype) / 1e9
        if fp_gb > FOOTPRINT_BUDGET_GB:
            print(f"SKIP {cfg.head_label} b={cfg.num_requests} ctx={cfg.context_len} "
                  f"{cfg.dtype_label}: contiguous cache {fp_gb:.1f}GB > {FOOTPRINT_BUDGET_GB}GB budget")
            all_rows.append(_row(cfg, "skipped", None, None,
                                 note=f"cache {fp_gb:.1f}GB > budget"))
            continue
        rows = bench_one(cfg, device, args.warmup, args.iters)
        for r in rows:
            r.update({"gpu": meta["gpu"], "torch": meta["torch"]})
            if r.get("median_ms") not in ("", None):
                print(f"{r['impl']:8s} {cfg.head_label} b={cfg.num_requests:<3d} "
                      f"ctx={cfg.context_len:<4d} {cfg.dtype_label}: "
                      f"median={r['median_ms']:.4f}ms p95={r['p95_ms']:.4f}ms "
                      f"eff={r['eff_gbps']:.1f}GB/s")
        all_rows.extend(rows)

    out = args.out or (__file__.rsplit("/benchmarks/", 1)[0] +
                       "/results/raw/operator_baseline" + ("_quick" if args.quick else "") + ".csv")
    write_csv(out, all_rows)
    print(f"\nwrote {len([r for r in all_rows if r['impl'] not in ('skipped',)])} rows -> {out}")


if __name__ == "__main__":
    main()
