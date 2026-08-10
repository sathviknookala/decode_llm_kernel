import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_STATE_FIELDS = (
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
    "power.draw",
    "clocks_throttle_reasons.active",
)


def sync():
    torch.cuda.synchronize()


def dtype_bytes(dtype):
    return torch.empty(0, dtype=dtype).element_size()


def logical_bytes(q, k, v, cos, sin, k_cache, v_cache):
    """Logical tensor traffic (bytes) for one invocation, from real element_size().

    reads  = q + k + v + cos rows + sin rows
    writes = q_rot + rotated-k cache slots + raw-v cache slots
    Split-half RoPE duplicates each table row, so a position contributes
    head_dim/2 unique cos and head_dim/2 unique sin scalars.
    This is a logical floor, not measured DRAM traffic: it excludes intermediates
    an implementation may materialize and cache bytes never touched.
    """
    num_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    half = head_dim // 2
    reads = {
        "q": num_tokens * num_q_heads * head_dim * q.element_size(),
        "k": num_tokens * num_kv_heads * head_dim * k.element_size(),
        "v": num_tokens * num_kv_heads * head_dim * v.element_size(),
        "cos": num_tokens * half * cos.element_size(),
        "sin": num_tokens * half * sin.element_size(),
    }
    writes = {
        "q_rot": num_tokens * num_q_heads * head_dim * q.element_size(),
        "k_cache": num_tokens * num_kv_heads * head_dim * k_cache.element_size(),
        "v_cache": num_tokens * num_kv_heads * head_dim * v_cache.element_size(),
    }
    read_bytes = sum(reads.values())
    write_bytes = sum(writes.values())
    return {
        "reads": reads,
        "writes": writes,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "total_bytes": read_bytes + write_bytes,
    }


def logical_eff_gbps(total_bytes, latency_ms):
    """Logical effective bandwidth: logical bytes / latency. Not physical DRAM traffic."""
    return total_bytes / (latency_ms / 1e3) / 1e9


def cache_footprint_bytes(batch, cache_alloc_len, num_kv_heads, head_dim, dtype):
    return 2 * batch * cache_alloc_len * num_kv_heads * head_dim * dtype_bytes(dtype)


def time_device_events(fn, warmup, iters):
    """Per-invocation device-timeline ms: CUDA events bracketing each individual call."""
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


def time_amortized_call(fn, warmup, iters):
    """Total wall time of an unsynchronized loop, divided by iters (ms).

    Steady-state cost per invocation with launches pipelined; NOT a per-call
    latency, because only the loop as a whole is synchronized.
    """
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e3


def time_synchronized_call(fn, warmup, iters):
    """Median wall time of individually synchronized single calls (ms).

    Each sample syncs before and after one invocation, so it includes full
    launch + drain latency and cannot pipeline across calls.
    """
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _percentile(sorted_samples, p):
    n = len(sorted_samples)
    k = min(n - 1, max(0, int(math.ceil(p / 100.0 * n)) - 1))
    return sorted_samples[k]


def summarize_device_samples(samples):
    """Distribution of per-invocation device times. No p50 column: it duplicates median."""
    s = sorted(samples)
    return {
        "device_median_ms": statistics.median(s),
        "device_p95_ms": _percentile(s, 95),
        "device_min_ms": s[0],
        "device_std_ms": statistics.pstdev(s),
    }


def _capture(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def gpu_state_metadata(device_index):
    query = ",".join(GPU_STATE_FIELDS)
    out = _capture([
        "nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits",
        f"--id={device_index}",
    ])
    if not out:
        return {"error": "nvidia-smi state query failed"}
    values = [v.strip() for v in out.splitlines()[0].split(",")]
    if len(values) != len(GPU_STATE_FIELDS):
        return {"error": "unexpected nvidia-smi state response", "raw": out}
    return dict(zip(GPU_STATE_FIELDS, values))


def record_gpu_state_end(meta, device_index):
    meta["gpu_state_end"] = gpu_state_metadata(device_index)


def _run_nvidia_smi(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20)
    except Exception as e:
        return subprocess.CompletedProcess(args, 1, "", f"{type(e).__name__}: {e}")


def _command_error(result):
    return (result.stderr or result.stdout or f"exit {result.returncode}").strip()


@contextmanager
def gpu_clock_lock(device_index, enabled):
    status = {"requested": bool(enabled), "locked": False, "target_sm_mhz": None}
    if enabled:
        target = _capture([
            "nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits",
            f"--id={device_index}",
        ])
        if target:
            target = target.splitlines()[0].strip()
            result = _run_nvidia_smi([
                "nvidia-smi", f"--id={device_index}", "-lgc", f"{target},{target}",
            ])
            if result.returncode == 0:
                status.update({"locked": True, "target_sm_mhz": target})
                print(f"# locked GPU {device_index} SM clock at {target} MHz")
            else:
                status["lock_error"] = _command_error(result)
        else:
            status["lock_error"] = "could not query clocks.max.sm"
        if not status["locked"]:
            print(f"WARNING: --lock-clocks failed for GPU {device_index}: "
                  f"{status['lock_error']}; continuing unlocked", file=sys.stderr)
    try:
        yield status
    finally:
        if status["locked"]:
            result = _run_nvidia_smi(["nvidia-smi", f"--id={device_index}", "-rgc"])
            if result.returncode == 0:
                print(f"# restored GPU {device_index} SM clock policy")
            else:
                print(f"WARNING: failed to restore GPU {device_index} clocks: "
                      f"{_command_error(result)}", file=sys.stderr)


def _nvcc_version():
    out = _capture(["nvcc", "--version"])
    if not out:
        return None
    for tok in out.split():
        if tok.startswith("V") and tok[1:2].isdigit():
            return tok[1:]
    return None


def _git_metadata():
    sha = _capture(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"])
    porcelain = _capture(["git", "-C", REPO_ROOT, "status", "--porcelain"])
    branch = _capture(["git", "-C", REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD"])
    dirty_files = [l.strip() for l in (porcelain or "").splitlines() if l.strip()]
    return {
        "git_sha": sha,
        "git_branch": branch,
        "git_dirty": bool(dirty_files),
        "git_dirty_files": dirty_files,
    }


def _dynamo_inductor_settings():
    out = {}
    try:
        import torch._dynamo.config as dc
        for name in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit",
                     "assume_static_by_default", "automatic_dynamic_shapes", "suppress_errors"):
            if hasattr(dc, name):
                out[f"dynamo.{name}"] = getattr(dc, name)
    except Exception:
        pass
    try:
        import torch._inductor.config as ic
        for name in ("max_autotune", "coordinate_descent_tuning", "freezing", "debug"):
            if hasattr(ic, name):
                out[f"inductor.{name}"] = getattr(ic, name)
        if hasattr(ic, "triton") and hasattr(ic.triton, "cudagraphs"):
            out["inductor.triton.cudagraphs"] = ic.triton.cudagraphs
    except Exception:
        pass
    return out


def env_metadata(device_index=0, *, cli_args=None, extra=None):
    props = torch.cuda.get_device_properties(device_index)
    cc = torch.cuda.get_device_capability(device_index)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(device_index),
        "gpu_vram_bytes": props.total_memory,
        "gpu_vram_gib": round(props.total_memory / 1024 ** 3, 2),
        "gpu_compute_capability": f"{cc[0]}.{cc[1]}",
        "gpu_multi_processor_count": props.multi_processor_count,
        "device_index": device_index,
        "nvidia_driver_version": _capture(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
             f"--id={device_index}"]),
        "cuda_runtime_version_torch": torch.version.cuda,
        "cuda_toolkit_version_nvcc": _nvcc_version(),
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_version": torch.__version__,
        "triton_version": _triton_version(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os_platform": platform.platform(),
        "os_system": platform.system(),
        "kernel_release": platform.release(),
        "cli_args": cli_args,
        "gpu_state_start": gpu_state_metadata(device_index),
    }
    meta.update(_git_metadata())
    meta.update(_dynamo_inductor_settings())
    if extra:
        meta.update(extra)
    return meta


def _triton_version():
    try:
        import triton
        return triton.__version__
    except Exception:
        return None


def write_csv(path, rows):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    if not rows:
        return
    fields = []
    for r in rows:
        for key in r:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_env_doc(path):
    """The environment block from a probe's .env.json, tolerant of both shapes.

    Probes write `{"environment": {...}, ...}`; `amdahl_probe.env.json` as committed is flat,
    because the code was left as it ran rather than edited after the measurement. Detect by
    looking for a key only the metadata carries.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        doc = json.load(f)
    if "environment" in doc:
        return doc["environment"]
    return doc if "timestamp_utc" in doc else {}


def read_run_completeness(path):
    """Whether the run that produced this env.json reached its end.

    Returns None when the file predates the flag, which is not the same as False: the
    committed artifacts were written by code that only wrote provenance on success, so their
    absence of a flag says nothing either way. Only an explicit False means a partial run.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    return doc.get("complete")


def write_json(path, obj):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)
