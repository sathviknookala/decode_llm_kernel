import csv
import hashlib
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


def _tree_digest(sha, porcelain):
    """Identity of the working tree, not just of the commit it sits on.

    Covers HEAD, the uncommitted diff against it, and the names of untracked files. What it
    misses is the *content* of untracked files -- hashing those means walking build artifacts
    and multi-GB result CSVs, so an untracked new module can still change behaviour without
    moving the digest.
    """
    if sha is None:
        return None
    h = hashlib.sha256()
    for part in (sha, porcelain or "", _capture(["git", "-C", REPO_ROOT, "diff", "HEAD"]) or ""):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


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
        "git_tree_digest": _tree_digest(sha, porcelain),
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


def _fsync_dir(dirname):
    fd = os.open(dirname or ".", os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass  # not every filesystem allows fsync on a directory
    finally:
        os.close(fd)


@contextmanager
def _atomic(path, **open_kwargs):
    """Write to a sibling temp file, flush it to the platter, and rename over the target.

    Checkpointing rewrites the whole file once per unit rather than once per run, so a crash
    inside `open(path, "w")` would destroy every row already measured -- the exact loss resume
    exists to prevent. os.replace is atomic within a filesystem, so a reader sees the old file
    or the new one and never a half-written one.

    os.replace orders the rename against the data only if the data is already durable. Without
    the fsyncs a host crash can land the rename while the blocks behind it are still in the page
    cache, which yields a zero-length or torn CSV -- the whole run lost, from the mechanism built
    to lose one unit. Killing the process is survivable without this; losing the machine is not.
    """
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", **open_kwargs) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(dirname)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for key in r:
            if key not in fields:
                fields.append(key)
    with _atomic(path, newline="") as f:
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


def bracketed(rungs):
    """A ladder with its first rung repeated at the end.

    Measured once, the baseline rung is always the first thing a configuration times, so
    "the first rung is slow" and "the baseline rung is special" produce the same number and
    the ladder cannot tell them apart. Measured on this rig that ambiguity was worth up to
    25% -- larger than most effects these ladders are built to detect.
    """
    rungs = list(rungs)
    return rungs + rungs[:1] if rungs else rungs


def ordering_drift(rows, value_col, *, group_col="impl", rung_col="rung_index"):
    """Closing baseline rung against the opening one, per group, written onto every row.

    Both hold everything except position in the run fixed, so anything off 1.0 is drift over
    the configuration -- and it bounds how much of any other ratio ordering alone explains.
    """
    for group in {r[group_col] for r in rows}:
        members = sorted((r for r in rows if r[group_col] == group),
                         key=lambda r: r[rung_col])
        baselines = [r for r in members if r.get("is_baseline_rung")]
        drift = ""
        if len(baselines) > 1 and baselines[0][value_col]:
            drift = baselines[-1][value_col] / baselines[0][value_col]
        for r in members:
            r["ordering_drift"] = drift
    return rows


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
    with _atomic(path) as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)


UNIT_KEY = "unit_key"
RUN_SEGMENT = "run_segment"
RESUME_HELP = ("keep whole units already measured in --out and fill in the rest; refuses if "
               "those rows came from a different tree")


def env_path(out):
    return os.path.splitext(out)[0] + ".env.json"


def read_rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# Everything else is assumed to change what a row means, so a resume refuses on it. Default-deny
# is the safe direction here: a new flag is guarded the day it is added, and a spurious refusal
# is loud, while a missed one puts two measurement settings in one CSV silently.
RESUME_IGNORED_ARGS = ("out", "resume")


def _measurement_args(cli_args):
    return {k: v for k, v in (cli_args or {}).items() if k not in RESUME_IGNORED_ARGS}


def resume_is_safe(path, meta, *, rows=None):
    """Refuse to append onto rows that were not measured the same way.

    Two trees' timings in one CSV, or two settings of --iters, would be compared against each
    other by every consumer of the file with nothing marking the boundary. The tree is checked
    by sha and then by working-tree digest, since a sha alone says nothing about uncommitted
    edits; everything else is checked by the arguments the run was given.
    """
    prior = read_env_doc(env_path(path))
    # An absent sidecar is only harmless when there is nothing to inherit. Unit-keyed rows with
    # no provenance beside them came from a tree and a set of arguments neither check can see,
    # so every guard below silently passes on a file it knows nothing about.
    if not prior:
        rows = read_rows(path) if rows is None else rows
        if any(r.get(UNIT_KEY) for r in rows):
            return False, (f"{os.path.basename(path)} holds finished units but "
                           f"{os.path.basename(env_path(path))} is missing, so the tree and "
                           "settings behind those rows cannot be checked")
    prior_sha = prior.get("git_sha")
    if prior_sha and meta.get("git_sha") and prior_sha != meta["git_sha"]:
        return False, (f"existing rows were measured at {prior_sha[:12]}, this tree is "
                       f"{meta['git_sha'][:12]}; resuming would mix two trees in one CSV")
    was_digest, now_digest = prior.get("git_tree_digest"), meta.get("git_tree_digest")
    if was_digest and now_digest and was_digest != now_digest:
        return False, ("existing rows were measured at the same commit but a different working "
                       f"tree ({was_digest[:12]} -> {now_digest[:12]}); resuming would mix two "
                       "trees in one CSV")
    # A sha identifies the tree only when nothing is uncommitted. Rows measured dirty by code
    # that recorded no digest cannot be shown to have come from this tree, so they are refused
    # rather than assumed -- committing (or --out to a fresh path) is the way through.
    if prior_sha and prior.get("git_dirty") and not was_digest:
        return False, (f"existing rows were measured at {prior_sha[:12]} with uncommitted "
                       "changes and no tree digest, so the tree behind them cannot be "
                       "identified; resuming would mix two trees in one CSV")
    was, now = _measurement_args(prior.get("cli_args")), _measurement_args(meta.get("cli_args"))
    if was and now:
        changed = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))
        if changed:
            detail = "; ".join(f"{k} {was.get(k)!r} -> {now.get(k)!r}" for k in changed)
            return False, (f"existing rows were measured with different settings ({detail}); "
                           f"resuming would mix two measurement settings in one CSV")
    return (True, "same tree") if prior_sha else (True, "no prior provenance")


class ResumableRun:
    """Rows on disk at every unit boundary, and a --resume that trusts only whole units.

    The unit is the smallest amount of work a probe can restart without changing what it
    measures. For a per-point ablation that is one row; for a bracketed ladder it is the whole
    ladder, because ordering_drift compares the closing baseline rung against the opening one
    and a restart between them would put a cold process inside the control.

    Completeness is read from the CSV's own unit_key column rather than from the sidecar, so
    the two files may be written in either order and a crash between them costs nothing: rows
    only ever reach disk a whole unit at a time.
    """

    def __init__(self, out, meta, *, resume=False, unit_ok=None, sidecar=None):
        self.out = out
        self.meta = dict(meta)
        self.sidecar = dict(sidecar or {})
        self.unit_ok = unit_ok or (lambda rows: bool(rows))
        self.rows = []
        self.units = set()
        # what the run that measured the inherited rows recorded about itself; a resumed run
        # that re-derives a reference the old rows were scored against would mix two of them
        self.prior_meta = {}
        self.segment = 0
        if resume:
            self._inherit()
        self._record_segment()

    def _inherit(self):
        prior_rows = read_rows(self.out)
        safe, why = resume_is_safe(self.out, self.meta, rows=prior_rows)
        if not safe:
            raise SystemExit(f"--resume refused: {why}")
        self.prior_meta = read_env_doc(env_path(self.out))
        groups = {}
        for r in prior_rows:
            groups.setdefault(r.get(UNIT_KEY, ""), []).append(r)
        # A CSV predating this column keys every row to "" and is re-measured whole, which is
        # the safe direction: no committed artifact carries unit boundaries.
        for key, rows in groups.items():
            if key and self.unit_ok(rows):
                self.units.add(key)
                self.rows.extend(rows)
        self.meta["resumed_onto_rows"] = len(self.rows)
        self.meta["resumed_units"] = len(self.units)
        print(f"# resuming {self.out}: {len(self.units)} units, {len(self.rows)} rows already "
              f"measured ({why})", flush=True)

    def _record_segment(self):
        """One entry per process that wrote to this CSV, carried forward across every resume.

        resumed_onto_rows describes only the most recent inheritance, so a file resumed three
        times reported the third and lost the first two -- and these CSVs are the deliverable.
        """
        chain = list(self.prior_meta.get("resume_chain") or [])
        self.segment = (chain[-1].get("segment", 0) + 1) if chain else 0
        chain.append({
            "segment": self.segment,
            "timestamp_utc": self.meta.get("timestamp_utc"),
            "git_sha": self.meta.get("git_sha"),
            "git_tree_digest": self.meta.get("git_tree_digest"),
            "inherited_rows": len(self.rows),
            "inherited_units": len(self.units),
        })
        self.meta["resume_chain"] = chain

    def done(self, unit_key):
        return unit_key in self.units

    def add(self, unit_key, rows):
        """One unit's rows, stamped and checkpointed. Nothing else may write to self.rows."""
        # Two configurations keyed the same is a probe bug with no loud symptom: fresh, their
        # rows merge under one key; resumed, done() reports the second already measured and it
        # never runs. Either way the CSV looks finished and a configuration is missing from it.
        if not unit_key:
            raise ValueError("a unit needs a key; rows keyed '' are dropped by every resume")
        if unit_key in self.units:
            raise ValueError(f"unit {unit_key!r} was already measured in this run; two "
                             f"configurations share a unit key and one of them will be lost")
        for r in rows:
            r[UNIT_KEY] = unit_key
            r[RUN_SEGMENT] = self.segment
        self.rows.extend(rows)
        self.units.add(unit_key)
        self._checkpoint(False)
        return rows

    def finish(self, sidecar_extra=None):
        self.sidecar.update(sidecar_extra or {})
        self._checkpoint(True)

    def _checkpoint(self, complete):
        write_csv(self.out, self.rows)
        write_json(env_path(self.out),
                   {"environment": self.meta, "complete": complete,
                    "rows_written": len(self.rows), "units_written": len(self.units),
                    **self.sidecar})
