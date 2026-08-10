import pytest

from benchmarks.benchmark_utils import bracketed, ordering_drift


def test_the_first_rung_is_repeated_at_the_end():
    assert bracketed([1, 8, 32]) == [1, 8, 32, 1]


def test_bracketing_preserves_the_ladder_between_the_brackets():
    """The rungs a probe was asked to sweep must survive unchanged, or the control has
    quietly changed the experiment."""
    assert bracketed([1, 2, 8, 64])[1:-1] == [2, 8, 64]


def test_a_single_rung_ladder_is_still_bracketed():
    assert bracketed([1]) == [1, 1]


def test_an_empty_ladder_stays_empty():
    assert bracketed([]) == []


def _row(impl, idx, ms, baseline=False):
    return {"impl": impl, "rung_index": idx, "amortized_call_ms": ms,
            "is_baseline_rung": baseline}


def test_drift_is_the_closing_baseline_against_the_opening_one():
    rows = [_row("compile", 0, 0.010, True), _row("compile", 1, 0.008),
            _row("compile", 2, 0.008, True)]
    assert all(r["ordering_drift"] == pytest.approx(0.8)
               for r in ordering_drift(rows, "amortized_call_ms"))


def test_a_stable_run_reports_no_drift():
    rows = [_row("compile", 0, 0.010, True), _row("compile", 1, 0.008),
            _row("compile", 2, 0.010, True)]
    assert ordering_drift(rows, "amortized_call_ms")[0]["ordering_drift"] == pytest.approx(1.0)


def test_each_group_gets_its_own_drift():
    """Two impls in one ladder drift independently; one figure for both would attribute one
    impl's warmup to the other."""
    rows = [_row("a", 0, 0.010, True), _row("a", 2, 0.005, True),
            _row("b", 0, 0.010, True), _row("b", 2, 0.010, True)]
    got = {r["impl"]: r["ordering_drift"] for r in ordering_drift(rows, "amortized_call_ms")}
    assert got["a"] == pytest.approx(0.5)
    assert got["b"] == pytest.approx(1.0)


def test_an_unbracketed_ladder_reports_no_drift_rather_than_one():
    """One baseline rung cannot measure drift, and reporting 1.0 would claim it had."""
    rows = [_row("compile", 0, 0.010, True), _row("compile", 1, 0.008)]
    assert all(r["ordering_drift"] == "" for r in ordering_drift(rows, "amortized_call_ms"))


def test_drift_reads_rungs_by_index_not_by_list_order():
    rows = [_row("c", 2, 0.008, True), _row("c", 1, 0.009), _row("c", 0, 0.010, True)]
    assert ordering_drift(rows, "amortized_call_ms")[0]["ordering_drift"] == pytest.approx(0.8)
