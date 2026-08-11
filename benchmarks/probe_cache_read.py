import argparse
import functools
import gc
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark_utils import (
    REPO_ROOT,
    RESUME_HELP,
    ResumableRun,
    bracketed,
    dtype_bytes,
    env_metadata,
    ordering_drift,
    summarize_device_samples,
    time_amortized_call,
    time_device_events,
)
from benchmarks.workload import DTYPES, HEAD_CONFIGS

HEAD_MAJOR = "head_major"
TOKEN_MAJOR_VIEW = "token_major_view"
TOKEN_MAJOR_COPY = "token_major_copy"
READ_FLOOR = "read_floor"
ARMS = (HEAD_MAJOR, TOKEN_MAJOR_VIEW, TOKEN_MAJOR_COPY, READ_FLOOR)

CTX_LADDER = (128, 512, 2048, 8192)
DEFAULT_BATCHES = (1, 32)


@functools.lru_cache(maxsize=1)
def supports_enable_gqa():
    """SDPA is a builtin with no introspectable signature, so ask it rather than read it."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.zeros(1, 2, 1, 8, device=device)
    kv = torch.zeros(1, 1, 4, 8, device=device)
    try:
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
        return True
    except (TypeError, RuntimeError):
        return False


def cache_shape(arm, batch, num_kv_heads, ctx, head_dim):
    """Head-major is [B, Hkv, S, D], which is what SDPA and HF's cache use. This project's
    locked layout is token-major [B, S, Hkv, D], so its consumer needs a transpose."""
    if arm == HEAD_MAJOR:
        return (batch, num_kv_heads, ctx, head_dim)
    return (batch, ctx, num_kv_heads, head_dim)


def cache_read_bytes(batch, num_kv_heads, ctx, head_dim, dtype):
    """K and V are both read in full for one decode step. Q is one token and rounds away.

    read_gbps divides this by latency, so it carries the same caveat as the sweep's
    pct_of_empirical_bw: a cache small enough to sit in L2 is served from L2, and the column
    then exceeds the streaming reference without anything moving that fast off DRAM. Read the
    wide end of the ctx ladder for a bandwidth claim.
    """
    return 2 * batch * num_kv_heads * ctx * head_dim * dtype_bytes(dtype)


def build_arm(arm, batch, num_q_heads, num_kv_heads, ctx, head_dim, dtype, device, seed):
    """Returns (thunk, gqa_path). The copy arm keeps `.contiguous()` inside the thunk: that
    conversion is the cost a consumer pays for the write-side layout, so hoisting it out would
    price the layout at zero."""
    g = torch.Generator(device=device).manual_seed(seed)
    shape = cache_shape(arm, batch, num_kv_heads, ctx, head_dim)
    k = torch.randn(shape, generator=g, dtype=dtype, device=device)
    v = torch.randn(shape, generator=g, dtype=dtype, device=device)
    q = torch.randn((batch, num_q_heads, 1, head_dim), generator=g, dtype=dtype, device=device)

    if arm == READ_FLOOR:
        def floor():
            k.sum()
            v.sum()
        return floor, "n/a", (k, v, q)

    gqa = num_q_heads != num_kv_heads
    if gqa and not supports_enable_gqa():
        raise RuntimeError(
            "this torch has no enable_gqa; materializing the KV heads instead would change the "
            "bytes read and make the arms incomparable")
    kwargs = {"enable_gqa": True} if gqa else {}
    gqa_path = "enable_gqa" if gqa else "none"

    if arm == HEAD_MAJOR:
        def run():
            F.scaled_dot_product_attention(q, k, v, **kwargs)
    elif arm == TOKEN_MAJOR_VIEW:
        kt, vt = k.transpose(1, 2), v.transpose(1, 2)

        def run():
            F.scaled_dot_product_attention(q, kt, vt, **kwargs)
    elif arm == TOKEN_MAJOR_COPY:
        def run():
            F.scaled_dot_product_attention(
                q, k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), **kwargs)
    else:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    return run, gqa_path, (k, v, q)


def probe_one(head_cfg, batch, ctx, dtype_label, device, seed, warmup, iters, arms):
    label, num_q_heads, num_kv_heads, head_dim = head_cfg
    dtype = DTYPES[dtype_label]
    rows = []
    ladder = bracketed(list(arms))
    for rung_index, arm in enumerate(ladder):
        base = {"head_label": label, "num_q_heads": num_q_heads, "num_kv_heads": num_kv_heads,
                "head_dim": head_dim, "num_requests": batch, "ctx": ctx,
                "dtype_label": dtype_label, "arm": arm, "rung_index": rung_index,
                "is_baseline_rung": rung_index in (0, len(ladder) - 1),
                "cache_read_bytes": cache_read_bytes(batch, num_kv_heads, ctx, head_dim, dtype)}
        try:
            thunk, gqa_path, held = build_arm(arm, batch, num_q_heads, num_kv_heads, ctx,
                                              head_dim, dtype, device, seed)
            with torch.inference_mode():
                stats = summarize_device_samples(time_device_events(thunk, warmup, iters))
                amortized = time_amortized_call(thunk, warmup, iters)
        except Exception as e:  # noqa: BLE001 -- an unrunnable arm is a result, not a crash
            print(f"  ERROR  {arm:18s}: {type(e).__name__}: {e}", flush=True)
            rows.append({**base, "gqa_path": "", "device_median_ms": "",
                         "amortized_call_ms": "", "read_gbps": "",
                         "error": f"{type(e).__name__}: {e}"})
            continue
        rows.append({**base, "gqa_path": gqa_path,
                     "device_median_ms": stats["device_median_ms"],
                     "amortized_call_ms": amortized,
                     "read_gbps": base["cache_read_bytes"] / (amortized / 1e3) / 1e9,
                     "error": ""})
        del thunk, held
        gc.collect()
        torch.cuda.empty_cache()
    return with_layout_ratio(rows)


def with_layout_ratio(rows):
    """Every arm against the opening head-major rung, which is what a consumer would get for
    free if the cache were written head-major. This ratio is the write-side layout's bill on
    the read side.

    head-major is both the baseline and the first arm measured, so the ladder is bracketed
    with a second head-major rung at the end: without it a drifting run and a real layout cost
    are the same number, and the effects here are a few percent.
    """
    base = next((r for r in sorted(rows, key=lambda r: r["rung_index"])
                 if r["arm"] == HEAD_MAJOR and r["amortized_call_ms"] != ""), None)
    for r in rows:
        r["vs_head_major"] = (r["amortized_call_ms"] / base["amortized_call_ms"]
                              if base and r["amortized_call_ms"] != "" else "")
    return ordering_drift(rows, "amortized_call_ms", group_col="head_label")


def unit_key(head_label, batch, ctx, dtype_label):
    return f"{head_label}_b{batch}_ctx{ctx}_{dtype_label}"


def ladder_ran(rows):
    """An unrunnable arm is recorded as a row on purpose, but a ladder where every arm failed
    is a failure of the config, not a measurement of it."""
    return any(not r.get("error") for r in rows)


def footprint_gb(batch, num_kv_heads, ctx, head_dim, dtype):
    """The copy arm holds the source and its contiguous copy at once."""
    return 2 * cache_read_bytes(batch, num_kv_heads, ctx, head_dim, dtype) / 1e9


def main():
    ap = argparse.ArgumentParser(
        description="What one decode step's attention pays to read the KV cache, per layout")
    ap.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    ap.add_argument("--ctxs", type=int, nargs="+", default=list(CTX_LADDER))
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--head-labels", nargs="+", default=None)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--footprint-budget-gb", type=float, default=12.0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results/raw/cache_read_probe.csv"))
    ap.add_argument("--resume", action="store_true", help=RESUME_HELP)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}, expected any of {list(ARMS)}")
    device = "cuda"
    heads = [h for h in HEAD_CONFIGS
             if not args.head_labels or h[0] in args.head_labels]
    print(f"# arms {args.arms} | enable_gqa available: {supports_enable_gqa()}")

    run = ResumableRun(args.out, env_metadata(0, cli_args=vars(args)), resume=args.resume,
                       unit_ok=ladder_ran,
                       sidecar={"arms": list(args.arms), "ctx_ladder": args.ctxs,
                                "enable_gqa_available": supports_enable_gqa()})
    for head_cfg in heads:
        for batch in args.batches:
            for ctx in args.ctxs:
                fp = footprint_gb(batch, head_cfg[2], ctx, head_cfg[3], DTYPES[args.dtype])
                if fp > args.footprint_budget_gb:
                    print(f"SKIPPED {head_cfg[0]} b={batch} ctx={ctx}: peak {fp:.1f} GB > "
                          f"{args.footprint_budget_gb} GB budget", flush=True)
                    continue
                key = unit_key(head_cfg[0], batch, ctx, args.dtype)
                if run.done(key):
                    print(f"\n=== {key}  already measured, skipping", flush=True)
                    continue
                print(f"\n=== {head_cfg[0]} b={batch} ctx={ctx} {args.dtype} "
                      f"({fp:.2f} GB peak)", flush=True)
                # the whole ladder is one unit: ordering_drift reads its closing head-major
                # rung against its opening one, and a restart between them would measure the
                # restart
                for r in run.add(key, probe_one(head_cfg, batch, ctx, args.dtype, device,
                                                args.seed, args.warmup, args.iters, args.arms)):
                    if r["error"]:
                        continue
                    print(f"  {r['arm']:18s} amort={r['amortized_call_ms']*1000:>8.2f} us  "
                          f"{r['read_gbps']:>7.1f} GB/s  vs head-major "
                          f"{r['vs_head_major']:.3f}", flush=True)

    run.finish()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
