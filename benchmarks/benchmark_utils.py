import csv
import math
import os
import statistics
import time

import torch


def sync():
    torch.cuda.synchronize()


def time_cuda_events(fn, warmup=25, iters=100):
    # device-timeline latency per call (ms), via CUDA events
    for _ in range(warmup):
        fn()
    sync()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    sync()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def time_op_call(fn, warmup=25, iters=100):
    # complete synchronized operator-call latency (ms): dispatch + launch + device
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e3


def _percentile(sorted_samples, p):
    n = len(sorted_samples)
    k = min(n - 1, max(0, int(math.ceil(p / 100.0 * n)) - 1))
    return sorted_samples[k]


def summarize(samples):
    s = sorted(samples)
    return {
        "median_ms": statistics.median(s),
        "p50_ms": _percentile(s, 50),
        "p95_ms": _percentile(s, 95),
        "min_ms": s[0],
        "std_ms": statistics.pstdev(s),
    }


def dtype_bytes(dtype):
    return torch.tensor([], dtype=dtype).element_size()


def logical_bytes(num_tokens, num_q_heads, num_kv_heads, head_dim, dtype):
    # minimal traffic the fused op must move: read q/k/v + cos/sin rows,
    # write q_rot + rotated k + raw v. cos/sin are stored FP32.
    e = dtype_bytes(dtype)
    q = num_tokens * num_q_heads * head_dim * e
    kv = num_tokens * num_kv_heads * head_dim * e
    cossin = num_tokens * head_dim * 4 * 2
    reads = q + kv + kv + cossin
    writes = q + kv + kv
    return reads + writes


def eff_gbps(total_bytes, median_ms):
    return total_bytes / (median_ms / 1e3) / 1e9


def cache_footprint_bytes(batch, max_seq, num_kv_heads, head_dim, dtype):
    return 2 * batch * max_seq * num_kv_heads * head_dim * dtype_bytes(dtype)


def env_metadata():
    cc = torch.cuda.get_device_capability(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_cap": f"{cc[0]}.{cc[1]}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
