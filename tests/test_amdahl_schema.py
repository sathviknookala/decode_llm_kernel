"""The gate reads the CSV by column name. A rename anywhere upstream turns a derivation into a
silent zero, which is the failure mode this repo archives claims over."""
import csv
import os

import pytest

from benchmarks.benchmark_utils import REPO_ROOT
from benchmarks.rung_patches import RUNG_LABELS
from benchmarks.summarize_results import amdahl_gate, amdahl_savings, read_amdahl_rows

CSV = os.path.join(REPO_ROOT, "results/raw/amdahl_probe.csv")

# Every name the derivation in summarize_results.py indexes by.
REQUIRED = {"batch", "ctx", "mode", "rung", "amortized_step_ms", "numerically_valid", "error"}
DERIVED_BY_GATE = {"full", "op_removed", "op_doubled"}

needs_csv = pytest.mark.skipif(not os.path.exists(CSV), reason="amdahl_probe.csv not present")


@needs_csv
def test_every_column_the_gate_indexes_by_is_present():
    with open(CSV, newline="") as f:
        header = set(next(csv.reader(f)))
    missing = REQUIRED - header
    assert not missing, f"gate reads columns that the probe no longer writes: {sorted(missing)}"


@needs_csv
def test_the_rungs_in_the_file_are_the_rungs_the_ladder_defines():
    rungs = {r["rung"] for r in csv.DictReader(open(CSV, newline=""))} - {"skipped"}
    assert rungs <= set(RUNG_LABELS), f"unknown rung(s) in CSV: {sorted(rungs - set(RUNG_LABELS))}"
    assert DERIVED_BY_GATE <= rungs, "the gate's own inputs are missing from the file"


@needs_csv
def test_the_committed_file_still_produces_a_gate():
    """An end-to-end guard: schema drift that passes the column check but breaks the derivation
    shows up here as a missing verdict rather than as a wrong number."""
    gate = amdahl_gate(amdahl_savings(read_amdahl_rows(CSV)))
    assert gate is not None
    assert gate["gate_mode"] == "hf_static_graph"
    assert 0.0 < gate["saving_pct"] < 100.0
    assert gate["verdict"]


@needs_csv
def test_no_row_carries_a_timing_and_an_error_at_once():
    for r in csv.DictReader(open(CSV, newline="")):
        assert not (r["amortized_step_ms"] and r["error"]), (
            f"row {r['mode']}/{r['rung']} is both timed and failed")


@needs_csv
def test_skipped_rows_explain_themselves():
    for r in csv.DictReader(open(CSV, newline="")):
        if r["rung"] == "skipped":
            assert r["error"], "a skipped config must record why, not just that"
            assert not r["amortized_step_ms"]
