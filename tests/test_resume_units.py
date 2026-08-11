"""What each script calls one unit of work.

The unit is the smallest amount a probe can restart without changing what it measures. For a
bracketed ladder that is the whole ladder: ordering_drift reads its closing baseline rung
against its opening one, so a restart between the two would measure the restart. These tests
pin that granularity, because keying a ladder per row would still resume -- and would silently
destroy the control the ladder exists to carry.
"""
import importlib
import inspect
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from benchmarks import probe_cache_read, probe_l2_residency, probe_ragged_positions
from benchmarks.benchmark_operator import config_ran
from benchmarks.benchmark_utils import UNIT_KEY, read_rows

BENCHMARKS = ("benchmark_operator", "probe_amdahl", "probe_cache_read", "probe_graph_floor",
              "probe_l2_residency", "probe_ragged_positions")


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda *a, **k: SimpleNamespace(L2_cache_size=50 * 1024 * 1024))


def stub_env(module, monkeypatch):
    monkeypatch.setattr(module, "env_metadata", lambda *a, **k: {"git_sha": "a" * 40})


def run_main(module, monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["probe"] + argv)
    module.main()


@pytest.mark.parametrize("name", BENCHMARKS)
def test_every_benchmark_checkpoints_and_resumes(name):
    """Five of these held every row in a list until the last config, so a kill lost the whole
    run rather than its tail."""
    src = inspect.getsource(importlib.import_module(f"benchmarks.{name}"))
    assert '"--resume"' in src, f"{name} has no --resume"
    assert "ResumableRun" in src, f"{name} does not checkpoint"
    assert "write_csv(" not in src, f"{name} still writes its CSV outside the checkpoint"


def test_the_l2_ladder_is_one_unit(tmp_path, monkeypatch, fake_gpu):
    stub_env(probe_l2_residency, monkeypatch)
    monkeypatch.setattr(probe_l2_residency, "probe", lambda cfg, *a, **k: [
        {"impl": "compile", "slot_sets": n, "write_working_set_mb": 1.0, "l2_multiple": 0.1,
         "device_median_ms": 0.01, "logical_eff_gbps": 100.0}
        for n in probe_l2_residency.bracketed(list(probe_l2_residency.SLOT_LADDER))])
    out = str(tmp_path / "l2.csv")
    run_main(probe_l2_residency, monkeypatch, ["--out", out])

    per_unit = Counter(r[UNIT_KEY] for r in read_rows(out))
    assert len(per_unit) == len(probe_l2_residency.PROBE_CONFIGS)
    assert set(per_unit.values()) == {len(probe_l2_residency.SLOT_LADDER) + 1}


def test_the_spread_ladder_is_one_unit_and_the_arm_is_part_of_its_key(
        tmp_path, monkeypatch, fake_gpu):
    """The two arms are separate experiments over the same configs; keyed by config alone, a
    --fresh-args run would inherit the held-args rows and measure nothing."""
    stub_env(probe_ragged_positions, monkeypatch)
    monkeypatch.setattr(probe_ragged_positions, "probe", lambda cfg, *a, **k: [
        {"impl": "compile", "spread": s, "tensor_mode": m, "device_median_ms": 0.01,
         "amortized_call_ms": 0.01, "amortized_call_ratio": 1.0}
        for s, m in probe_ragged_positions.rungs(cfg.cache_alloc_len)])
    out = str(tmp_path / "ragged.csv")
    run_main(probe_ragged_positions, monkeypatch, ["--out", out])
    held = {r[UNIT_KEY] for r in read_rows(out)}

    run_main(probe_ragged_positions, monkeypatch, ["--out", out, "--fresh-args", "--resume"])
    both = {r[UNIT_KEY] for r in read_rows(out)}
    assert len(both) == 2 * len(held)
    assert all(k.endswith("|held_args") or k.endswith("|fresh_args") for k in both)


def test_the_read_side_ladder_is_one_unit(tmp_path, monkeypatch, fake_gpu):
    stub_env(probe_cache_read, monkeypatch)
    monkeypatch.setattr(probe_cache_read, "supports_enable_gqa", lambda: True)
    monkeypatch.setattr(probe_cache_read, "probe_one", lambda *a, **k: [
        {"arm": arm, "amortized_call_ms": 0.01, "read_gbps": 100.0, "vs_head_major": 1.0,
         "error": ""}
        for arm in probe_cache_read.bracketed(list(probe_cache_read.ARMS))])
    out = str(tmp_path / "read.csv")
    run_main(probe_cache_read, monkeypatch,
             ["--out", out, "--batches", "1", "--ctxs", "128", "--head-labels", "mha"])

    per_unit = Counter(r[UNIT_KEY] for r in read_rows(out))
    assert per_unit == Counter({"mha_b1_ctx128_bf16": len(probe_cache_read.ARMS) + 1})


def test_a_resumed_ladder_is_not_re_measured(tmp_path, monkeypatch, fake_gpu):
    stub_env(probe_l2_residency, monkeypatch)
    calls = []

    def counting_probe(cfg, *a, **k):
        calls.append(cfg.label())
        return [{"impl": "compile", "slot_sets": 1, "write_working_set_mb": 1.0,
                 "l2_multiple": 0.1, "device_median_ms": 0.01, "logical_eff_gbps": 100.0}]

    monkeypatch.setattr(probe_l2_residency, "probe", counting_probe)
    out = str(tmp_path / "l2.csv")
    run_main(probe_l2_residency, monkeypatch, ["--out", out])
    first = list(calls)
    calls.clear()

    run_main(probe_l2_residency, monkeypatch, ["--out", out, "--resume"])
    assert calls == []
    assert len(read_rows(out)) == len(first)


def test_the_read_side_keys_a_ladder_by_everything_that_changes_its_bytes():
    key = probe_cache_read.unit_key
    base = key("mha", 1, 128, "bf16")
    assert base != key("gqa", 1, 128, "bf16")
    assert base != key("mha", 32, 128, "bf16")
    assert base != key("mha", 1, 8192, "bf16")
    assert base != key("mha", 1, 128, "fp32")


def test_a_ladder_where_every_arm_failed_is_retried():
    """An unrunnable arm is recorded as a row on purpose, but a ladder of nothing but failures
    is not a measurement of the config."""
    assert probe_cache_read.ladder_ran([{"error": ""}, {"error": "RuntimeError"}])
    assert not probe_cache_read.ladder_ran([{"error": "RuntimeError"}])


def test_the_sweep_inherits_recorded_failures_but_retries_a_dead_config():
    assert config_ran([{"validation": "pass"}, {"validation": "ERROR"}])
    assert config_ran([{"validation": "FAIL"}])
    assert config_ran([{"validation": "not-run"}])
    assert not config_ran([{"validation": "ERROR"}, {"validation": "ERROR"}])
