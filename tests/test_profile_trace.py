import json

import pytest

from benchmarks.profile_operator import summarize_trace


def trace(tmp_path, events):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    return str(path)


def ev(cat, name, dur, ph="X"):
    return {"cat": cat, "name": name, "dur": dur, "ph": ph}


def test_a_memcpy_is_not_a_distinct_kernel(tmp_path):
    """The committed artifact reported 3 distinct kernels for a compiled path launching 2 per
    invocation: per_name was keyed over every device category, so Memcpy DtoD counted as one."""
    path = trace(tmp_path, [ev("kernel", "k1", 1.0), ev("kernel", "k2", 2.0),
                            ev("gpu_memcpy", "Memcpy DtoD (Device -> Device)", 0.5)])
    got = summarize_trace(path, iters=1)
    assert got["distinct_kernels"] == 2
    assert [b["name"] for b in got["by_kernel"]] == ["k2", "k1"]
    assert got["kernel_events_total"] == 2


def test_device_totals_still_span_every_device_category(tmp_path):
    """distinct_kernels narrowing must not narrow the fields that say "device"."""
    path = trace(tmp_path, [ev("kernel", "k1", 1.0),
                            ev("gpu_memcpy", "Memcpy DtoD", 0.5),
                            ev("gpu_memset", "Memset", 0.25)])
    got = summarize_trace(path, iters=1)
    assert got["device_events_total"] == 3
    assert got["device_time_total_us"] == pytest.approx(1.75)
    assert got["kernel_events_total"] == 1


def test_launch_counts_are_per_invocation(tmp_path):
    path = trace(tmp_path, [ev("kernel", "k", 1.0) for _ in range(36)])
    got = summarize_trace(path, iters=2)
    assert got["kernels_per_invocation"] == 18
    assert got["by_kernel"][0]["per_invocation"] == 18


def test_non_device_and_non_complete_events_are_ignored(tmp_path):
    path = trace(tmp_path, [ev("kernel", "k", 1.0), ev("cpu_op", "aten::add", 9.0),
                            ev("kernel", "flow", 5.0, ph="f")])
    got = summarize_trace(path, iters=1)
    assert got["device_events_total"] == 1
    assert got["device_time_total_us"] == pytest.approx(1.0)


def test_zero_iters_does_not_divide(tmp_path):
    path = trace(tmp_path, [ev("kernel", "k", 1.0)])
    got = summarize_trace(path, iters=0)
    assert got["kernels_per_invocation"] == 0
    assert got["device_time_per_invocation_us"] == 0


def test_an_empty_trace_summarizes_to_zeros(tmp_path):
    got = summarize_trace(trace(tmp_path, []), iters=20)
    assert (got["device_events_total"], got["distinct_kernels"]) == (0, 0)
    assert got["by_kernel"] == []


def test_by_kernel_is_ordered_by_total_device_time(tmp_path):
    path = trace(tmp_path, [ev("kernel", "slow", 10.0), ev("kernel", "fast", 1.0),
                            ev("kernel", "fast", 1.0), ev("kernel", "mid", 5.0)])
    got = summarize_trace(path, iters=1)
    assert [b["name"] for b in got["by_kernel"]] == ["slow", "mid", "fast"]
    assert got["by_kernel"][-1]["count"] == 2
