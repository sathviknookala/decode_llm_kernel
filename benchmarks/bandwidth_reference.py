import argparse
import statistics
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark_utils import (
    REPO_ROOT,
    env_metadata,
    gpu_clock_lock,
    record_gpu_state_end,
    time_device_events,
    write_json,
)

DEFAULT_BUFFER_MIB = 512


def _gbps(nbytes, ms):
    return nbytes / (ms / 1e3) / 1e9


def measure_copy(device, buffer_mib=DEFAULT_BUFFER_MIB, warmup=10, iters=50,
                 dtype=torch.float32):
    """device-to-device copy. Byte convention: read src + write dst = 2 * buffer bytes."""
    n = (buffer_mib * 1024 ** 2) // dtype.itemsize
    src = torch.empty(n, dtype=dtype, device=device).uniform_(-1, 1)
    dst = torch.empty_like(src)
    buf_bytes = n * dtype.itemsize
    samples = time_device_events(lambda: dst.copy_(src), warmup, iters)
    med = statistics.median(samples)
    return {
        "kernel": "copy_ (d2d)",
        "byte_convention": "read src + write dst = 2 * buffer_bytes",
        "buffer_bytes": buf_bytes,
        "buffer_mib": buffer_mib,
        "bytes_moved_per_iter": 2 * buf_bytes,
        "median_ms": med,
        "min_ms": min(samples),
        "gbps_median": _gbps(2 * buf_bytes, med),
        "gbps_best": _gbps(2 * buf_bytes, min(samples)),
    }


def measure_triad(device, buffer_mib=DEFAULT_BUFFER_MIB, warmup=10, iters=50,
                  dtype=torch.float32):
    """y += a*x. Byte convention: read x + read y + write y = 3 * buffer bytes."""
    n = (buffer_mib * 1024 ** 2) // dtype.itemsize
    x = torch.empty(n, dtype=dtype, device=device).uniform_(-1, 1)
    y = torch.empty(n, dtype=dtype, device=device).uniform_(-1, 1)
    buf_bytes = n * dtype.itemsize
    samples = time_device_events(lambda: y.add_(x, alpha=2.0), warmup, iters)
    med = statistics.median(samples)
    return {
        "kernel": "add_ (y += a*x)",
        "byte_convention": "read x + read y + write y = 3 * buffer_bytes",
        "buffer_bytes": buf_bytes,
        "buffer_mib": buffer_mib,
        "bytes_moved_per_iter": 3 * buf_bytes,
        "median_ms": med,
        "min_ms": min(samples),
        "gbps_median": _gbps(3 * buf_bytes, med),
        "gbps_best": _gbps(3 * buf_bytes, min(samples)),
    }


def measure_bandwidth_reference(device="cuda", buffer_mib=DEFAULT_BUFFER_MIB,
                                warmup=10, iters=50):
    """Empirical achievable bandwidth on this device. Buffers far exceed L2 so the
    measurement is DRAM-bound rather than cache-resident."""
    props = torch.cuda.get_device_properties(device)
    copy = measure_copy(device, buffer_mib, warmup, iters)
    triad = measure_triad(device, buffer_mib, warmup, iters)
    return {
        "reference_gbps": copy["gbps_median"],
        "reference_source": "copy_ median",
        "l2_cache_bytes": getattr(props, "L2_cache_size", None),
        "buffer_mib": buffer_mib,
        "warmup": warmup,
        "iters": iters,
        "copy": copy,
        "triad": triad,
    }


def main():
    ap = argparse.ArgumentParser(description="Empirical device memory-bandwidth reference")
    ap.add_argument("--buffer-mib", type=int, default=DEFAULT_BUFFER_MIB)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--lock-clocks", action="store_true",
                    help="attempt to lock the selected GPU at its maximum SM clock")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results/raw/bandwidth_reference.json"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")

    torch.cuda.set_device(args.device_index)
    device = f"cuda:{args.device_index}"
    with gpu_clock_lock(args.device_index, args.lock_clocks) as clock_status:
        meta = env_metadata(args.device_index, cli_args=vars(args),
                            extra={"clock_lock": clock_status})
        ref = measure_bandwidth_reference(device, args.buffer_mib, args.warmup, args.iters)
        record_gpu_state_end(meta, args.device_index)
        payload = {"environment": meta, "bandwidth_reference": ref}
        write_json(args.out, payload)

    print(f"buffer            : {ref['buffer_mib']} MiB   (L2 = {ref['l2_cache_bytes']} bytes)")
    for key in ("copy", "triad"):
        m = ref[key]
        print(f"{m['kernel']:<18}: median {m['median_ms']:.3f} ms -> "
              f"{m['gbps_median']:.1f} GB/s (best {m['gbps_best']:.1f})   [{m['byte_convention']}]")
    print(f"\nreference_gbps    : {ref['reference_gbps']:.1f} GB/s ({ref['reference_source']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
