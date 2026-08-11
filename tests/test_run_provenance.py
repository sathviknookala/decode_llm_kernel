import json

import pytest

from benchmarks.benchmark_utils import (
    ResumableRun,
    env_path,
    read_run_completeness,
    write_json,
)
from benchmarks.probe_amdahl import rung_sidecar

ROWS = [{"batch": 32, "ctx": 128, "mode": "hf_eager", "rung": "full",
         "amortized_step_ms": 61.9}]


@pytest.fixture
def out(tmp_path):
    return str(tmp_path / "probe.csv")


def amdahl_run(out):
    return ResumableRun(out, {"gpu_name": "test"}, sidecar=rung_sidecar())


def test_provenance_lands_with_the_first_rows_not_only_at_the_end(out):
    """A run killed mid-flight used to leave this run's rows beside the previous run's
    env.json, with nothing in either file saying so."""
    amdahl_run(out).add("u1", list(ROWS))
    assert read_run_completeness(env_path(out)) is False
    doc = json.load(open(env_path(out)))
    assert doc["environment"]["gpu_name"] == "test"
    assert doc["rows_written"] == 1


def test_the_final_write_marks_the_run_complete(out):
    run = amdahl_run(out)
    run.add("u1", list(ROWS))
    run.add("u2", list(ROWS))
    run.finish()
    assert read_run_completeness(env_path(out)) is True
    assert json.load(open(env_path(out)))["rows_written"] == 2


def test_a_file_predating_the_flag_is_unknown_rather_than_incomplete(tmp_path):
    """The committed artifacts were written by code that only wrote provenance on success, so
    a missing flag says nothing -- reporting them as partial would be a false alarm."""
    path = str(tmp_path / "old.env.json")
    write_json(path, {"environment": {"gpu_name": "test"}})
    assert read_run_completeness(path) is None


def test_a_missing_env_file_is_unknown(tmp_path):
    assert read_run_completeness(str(tmp_path / "absent.env.json")) is None


def test_the_rung_registry_travels_with_every_write(out):
    amdahl_run(out).add("u1", list(ROWS))
    rungs = json.load(open(env_path(out)))["rungs"]
    assert {"op_removed", "op_doubled"}.issubset({r["rung"] for r in rungs})
    assert all("numerically_valid" in r for r in rungs)
