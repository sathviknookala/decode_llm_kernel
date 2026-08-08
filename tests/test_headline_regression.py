import os

import pytest
import torch

from benchmarks import positions as pos
from benchmarks.benchmark_operator import run_config
from benchmarks.impls import resolve_impls
from benchmarks.workload import Config
from types import SimpleNamespace

# Small enough to run in seconds, wide enough that the relations below are not noise: the
# committed baseline puts every one of them far outside this configuration's spread.
CFG = Config("mha", 32, 32, 128, 8, 512, "bf16", pos.RAGGED)
SEED = 1234
IMPLS = ["eager", "compile", "graph_eager", "graph_compile"]

# Deliberately loose. Each bound is roughly half the effect summary.md derives from v3, so a
# genuine break trips it and ordinary run-to-run variation does not. Tightening these to the
# measured values would turn a regression guard into a flaky one.
MIN_COMPILE_SPEEDUP = 1.5          # v3 derives 2.33x at b=1, 2.98x at b>=8
MIN_GRAPH_SAVING_PCT = 30.0        # v3 derives 62-69% of the direct path's device median

RUN = os.environ.get("DECODE_RUN_REGRESSION") == "1"
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    # Opt-in despite costing about 5s: these are timing assertions, and a contended GPU
    # compresses the very ratios they check. A spurious red on someone else's ollama server is
    # worse than a check that has to be asked for -- run it before quoting a number.
    pytest.mark.skipif(not RUN, reason="set DECODE_RUN_REGRESSION=1 to re-run the headline "
                                       "relations live against this tree (~5s)"),
]


@pytest.fixture(scope="module")
def rows():
    """One live sweep of the four baseline impls, shared by every relation below.

    This is the check nothing else performs: summarize_results.py proves the claims follow
    from the committed CSV, and the unit tests prove the rig computes what it says, but
    neither notices when the current tree stops reproducing the CSV. That gap is how
    "88% of achievable" and "3.8-4.0x" outlived their data.
    """
    args = SimpleNamespace(seed=SEED, compile_mode=None, compile_backend="inductor",
                           warmup=10, iters=50)
    got = run_config(CFG, "cuda", args, None, resolve_impls(IMPLS))
    return {r["impl"]: r for r in got}


def test_every_baseline_impl_still_validates(rows):
    failures = {label: r["validation_detail"] for label, r in rows.items()
                if r["validation"] != "pass"}
    assert not failures, f"impls no longer validate against the oracle: {failures}"


def test_compile_is_still_faster_than_eager(rows):
    speedup = rows["eager"]["device_median_ms"] / rows["compile"]["device_median_ms"]
    assert speedup > MIN_COMPILE_SPEEDUP, (
        f"compile is only {speedup:.2f}x eager; summary.md's speedup table is derived from a "
        f"CSV that this tree no longer reproduces")


@pytest.mark.parametrize("direct,graphed", [("eager", "graph_eager"),
                                            ("compile", "graph_compile")])
def test_cuda_graphs_still_remove_most_of_the_launch_cost(rows, direct, graphed):
    saving = 100.0 * (1.0 - rows[graphed]["device_median_ms"] / rows[direct]["device_median_ms"])
    assert saving > MIN_GRAPH_SAVING_PCT, (
        f"{graphed} removes only {saving:.1f}% of {direct}; the launch-elimination claim that "
        f"demoted Checkpoint C rests on this")


def test_the_event_harness_floor_still_sits_above_the_amortized_route(rows):
    """device_median_ms carries a 4.3 us bracketing cost that amortized_call_ms does not. If
    that stopped holding, every microsecond-scale claim would need re-deriving."""
    for label, r in rows.items():
        assert r["amortized_call_ms"] < r["device_median_ms"], (
            f"{label}: amortized {r['amortized_call_ms']*1000:.2f} us is not below the event "
            f"median {r['device_median_ms']*1000:.2f} us")


def test_the_gate_that_cleared_these_rows_is_still_the_full_one(rows):
    """A weaker gate under a passing run is exactly what v2 hid; the case names are the check."""
    for label, r in rows.items():
        cases = set(r["validation_cases"].split("|"))
        assert {"permuted-requests", "strided-qkv"} <= cases, f"{label} cleared a weaker gate"
