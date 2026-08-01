import argparse
import gc
import itertools
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
DEFAULT_SIZE_SWEEP_MIB = (1, 4, 16, 64, 256, 512)
DEFAULT_SCATTER_SPEC = {
    "num_requests": 128,
    "cache_alloc_len": 2048,
    "num_kv_heads": 32,
    "head_dim": 128,
    "dtype": torch.float32,
}
SCATTER_POSITION_SETS = 8
SCATTER_SEED = 1234
SCATTER_FOOTPRINT_BUDGET_BYTES = int(12e9)


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


def measure_scattered_write(device, *, num_requests, cache_alloc_len, num_kv_heads,
                            head_dim, dtype=torch.float32, warmup=10, iters=50,
                            num_position_sets=SCATTER_POSITION_SETS, seed=SCATTER_SEED):
    shape = (num_requests, cache_alloc_len, num_kv_heads, head_dim)
    cache_bytes = (2 * num_requests * cache_alloc_len * num_kv_heads * head_dim
                   * dtype.itemsize)
    if cache_bytes > SCATTER_FOOTPRINT_BUDGET_BYTES:
        raise ValueError(f"scattered cache footprint {cache_bytes} exceeds "
                         f"{SCATTER_FOOTPRINT_BUDGET_BYTES}-byte budget")

    k_cache = torch.empty(shape, dtype=dtype, device=device)
    v_cache = torch.empty_like(k_cache)
    k = torch.randn(num_requests, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn_like(k)
    g = torch.Generator().manual_seed(seed)
    address_sets = []
    for _ in range(num_position_sets):
        request_indices = torch.randperm(num_requests, generator=g).to(device)
        positions = torch.randint(cache_alloc_len, (num_requests,), generator=g).to(device)
        address_sets.append((request_indices, positions))
    addresses = itertools.cycle(address_sets)

    def scatter(kc, vc, k_src, v_src, request_indices, positions):
        kc[request_indices, positions] = k_src
        vc[request_indices, positions] = v_src

    torch._dynamo.reset()
    compiled_scatter = torch.compile(scatter, dynamic=False)

    def write_slots():
        request_indices, positions = next(addresses)
        compiled_scatter(k_cache, v_cache, k, v, request_indices, positions)

    with torch.inference_mode():
        samples = time_device_events(write_slots, warmup, iters)
    median_ms = statistics.median(samples)
    slot_bytes = num_kv_heads * head_dim * dtype.itemsize
    bytes_moved = 4 * num_requests * slot_bytes
    return {
        "kernel": "torch.compile scattered K/V slot writes",
        "byte_convention": "read K/V sources + write K/V cache slots = 4 * slot_bytes",
        "num_requests": num_requests,
        "cache_alloc_len": cache_alloc_len,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": str(dtype).removeprefix("torch."),
        "slot_bytes_per_cache": slot_bytes,
        "cache_footprint_bytes": cache_bytes,
        "position_sets": num_position_sets,
        "seed": seed,
        "bytes_moved_per_iter": bytes_moved,
        "median_ms": median_ms,
        "min_ms": min(samples),
        "gbps_median": _gbps(bytes_moved, median_ms),
        "gbps_best": _gbps(bytes_moved, min(samples)),
    }


def measure_bandwidth_reference(device="cuda", buffer_mib=DEFAULT_BUFFER_MIB,
                                warmup=10, iters=50,
                                size_sweep_mib=DEFAULT_SIZE_SWEEP_MIB,
                                scatter_spec=DEFAULT_SCATTER_SPEC):
    """Empirical achievable bandwidth on this device. Buffers far exceed L2 so the
    measurement is DRAM-bound rather than cache-resident."""
    props = torch.cuda.get_device_properties(device)
    copy = measure_copy(device, buffer_mib, warmup, iters)
    triad = measure_triad(device, buffer_mib, warmup, iters)
    copy_size_sweep = []
    for size_mib in size_sweep_mib:
        result = copy if size_mib == buffer_mib else measure_copy(
            device, size_mib, warmup, iters)
        copy_size_sweep.append(result)
        gc.collect()
        torch.cuda.empty_cache()

    scattered_write = None
    if scatter_spec:
        try:
            scattered_write = measure_scattered_write(
                device, **scatter_spec, warmup=warmup, iters=iters)
        except Exception as e:  # noqa: BLE001 -- a missing reference is recorded, not hidden
            scattered_write = {"error": f"{type(e).__name__}: {e}"}
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    return {
        "reference_gbps": copy["gbps_median"],
        "reference_source": "copy_ median",
        "scattered_write_reference_gbps": (
            scattered_write.get("gbps_median") if scattered_write else None),
        "scattered_write_reference_source": (
            scattered_write.get("kernel") if scattered_write else None),
        "l2_cache_bytes": getattr(props, "L2_cache_size", None),
        "buffer_mib": buffer_mib,
        "warmup": warmup,
        "iters": iters,
        "copy": copy,
        "triad": triad,
        "copy_size_sweep": copy_size_sweep,
        "scattered_write": scattered_write,
    }


def main():
    ap = argparse.ArgumentParser(description="Empirical device memory-bandwidth reference")
    ap.add_argument("--buffer-mib", type=int, default=DEFAULT_BUFFER_MIB)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--size-sweep-mib", type=int, nargs="+",
                    default=list(DEFAULT_SIZE_SWEEP_MIB))
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
        ref = measure_bandwidth_reference(
            device, args.buffer_mib, args.warmup, args.iters, args.size_sweep_mib)
        record_gpu_state_end(meta, args.device_index)
        payload = {"environment": meta, "bandwidth_reference": ref}
        write_json(args.out, payload)

    print(f"buffer            : {ref['buffer_mib']} MiB   (L2 = {ref['l2_cache_bytes']} bytes)")
    for key in ("copy", "triad"):
        m = ref[key]
        print(f"{m['kernel']:<18}: median {m['median_ms']:.3f} ms -> "
              f"{m['gbps_median']:.1f} GB/s (best {m['gbps_best']:.1f})   [{m['byte_convention']}]")
    print("\ncopy size sweep:")
    for m in ref["copy_size_sweep"]:
        print(f"  {m['buffer_mib']:4d} MiB: {m['gbps_median']:.1f} GB/s median")
    scattered = ref["scattered_write"]
    if "error" in scattered:
        print(f"\nscattered write   : ERROR {scattered['error']}")
    else:
        print(f"\nscattered write   : median {scattered['median_ms']:.3f} ms -> "
              f"{scattered['gbps_median']:.1f} GB/s   [{scattered['byte_convention']}]")
    print(f"\nreference_gbps    : {ref['reference_gbps']:.1f} GB/s ({ref['reference_source']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
