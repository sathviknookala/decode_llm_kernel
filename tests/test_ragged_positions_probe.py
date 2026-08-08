import pytest
import torch

from benchmarks import positions as pos
from benchmarks.impls import resolve_impls
from benchmarks.probe_ragged_positions import (
    DISTINCT,
    SHARED,
    SPREAD_LADDER,
    probe,
    rungs,
    spread_position_sets,
    with_ratios,
)
from benchmarks.workload import Config


def test_the_shared_rung_reproduces_the_sweeps_uniform_mode():
    """The baseline rung has to be what the sweep measured, or the ratio is against something
    else: one tensor object, repeated, every request at the last slot."""
    sets = spread_position_sets(4, 2048, 1, 8, seed=1, device="cpu", tensor_mode=SHARED)
    assert len(sets) == 8
    assert all(s is sets[0] for s in sets)
    assert torch.equal(sets[0], torch.full((4,), 2047, dtype=torch.long))


def test_the_control_holds_the_values_and_varies_only_the_object():
    """This is the rung the sweep lacks. Identical values and identical memory traffic, so a
    difference against the shared rung cannot be about position values at all."""
    shared = spread_position_sets(4, 2048, 1, 8, seed=1, device="cpu", tensor_mode=SHARED)
    distinct = spread_position_sets(4, 2048, 1, 8, seed=1, device="cpu", tensor_mode=DISTINCT)
    assert all(torch.equal(d, shared[0]) for d in distinct)
    assert len({id(d) for d in distinct}) == 8


def test_spread_bounds_the_window_at_the_end_of_the_cache():
    for spread in (2, 64, 2048):
        for p in spread_position_sets(8, 2048, spread, 4, seed=7, device="cpu"):
            assert int(p.min()) >= 2048 - spread
            assert int(p.max()) <= 2047


def test_a_wider_spread_touches_more_distinct_slots():
    def distinct_slots(spread):
        sets = spread_position_sets(8, 2048, spread, 8, seed=7, device="cpu")
        return len({v for p in sets for v in p.tolist()})

    assert distinct_slots(1) == 1
    assert distinct_slots(2048) > distinct_slots(8) > distinct_slots(2)


def test_a_shared_tensor_above_spread_one_is_refused():
    """Sets at spread>1 differ from each other, so they cannot be one object; silently
    returning distinct tensors would mislabel the rung."""
    with pytest.raises(ValueError, match="shared tensor only makes sense"):
        spread_position_sets(4, 2048, 8, 4, seed=1, device="cpu", tensor_mode=SHARED)


@pytest.mark.parametrize("spread", [0, 4096])
def test_a_spread_outside_the_cache_is_refused(spread):
    with pytest.raises(ValueError, match="outside"):
        spread_position_sets(4, 2048, spread, 4, seed=1, device="cpu")


def test_the_ladder_brackets_the_spread_rungs_with_the_shared_one():
    """Measured once, the shared rung is always first, and 'the first rung is slower' would be
    indistinguishable from an effect of sharing the tensor."""
    got = rungs(2048)
    assert got[0] == (1, SHARED)
    assert got[-1] == (1, SHARED)
    assert all(mode == DISTINCT for _, mode in got[1:-1])
    assert [s for s, _ in got[1:-1]] == list(SPREAD_LADDER)


def test_the_ladder_drops_rungs_wider_than_the_cache():
    assert all(s <= 128 for s, _ in rungs(128))


def _rung(impl, mode, idx, ms):
    return {"impl": impl, "tensor_mode": mode, "rung_index": idx,
            "device_median_ms": ms * 2, "amortized_call_ms": ms}


def test_ratios_are_against_the_same_impls_opening_shared_rung():
    rows = [_rung("compile", SHARED, 0, 0.010), _rung("compile", DISTINCT, 1, 0.014),
            _rung("compile", SHARED, 2, 0.010),
            _rung("graph_compile", SHARED, 0, 0.005),
            _rung("graph_compile", DISTINCT, 1, 0.005),
            _rung("graph_compile", SHARED, 2, 0.005)]
    got = {(r["impl"], r["tensor_mode"], r["rung_index"]): r for r in with_ratios(rows)}
    assert got[("compile", DISTINCT, 1)]["amortized_call_ratio"] == pytest.approx(1.4)
    assert got[("graph_compile", DISTINCT, 1)]["amortized_call_ratio"] == pytest.approx(1.0)
    assert got[("compile", SHARED, 0)]["device_median_ratio"] == pytest.approx(1.0)


def test_a_drifting_run_is_reported_rather_than_folded_into_the_effect():
    """Both shared rungs hold tensor and values fixed, so a ratio between them is drift over
    the run -- and it bounds how much of the distinct rung's ratio ordering alone explains."""
    rows = [_rung("compile", SHARED, 0, 0.010), _rung("compile", DISTINCT, 1, 0.008),
            _rung("compile", SHARED, 2, 0.008)]
    got = with_ratios(rows)
    assert all(r["ordering_drift"] == pytest.approx(0.8) for r in got)


def test_a_stable_run_reports_no_drift():
    rows = [_rung("compile", SHARED, 0, 0.010), _rung("compile", DISTINCT, 1, 0.008),
            _rung("compile", SHARED, 2, 0.010)]
    assert with_ratios(rows)[0]["ordering_drift"] == pytest.approx(1.0)


def test_a_missing_baseline_leaves_the_ratio_blank_rather_than_one():
    rows = [_rung("compile", DISTINCT, 1, 0.01)]
    got = with_ratios(rows)[0]
    assert got["amortized_call_ratio"] == ""
    assert got["ordering_drift"] == ""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reallocating_per_rung_is_recorded_so_the_two_arms_are_distinguishable():
    cfg = Config("mha", 8, 8, 64, 2, 128, "fp32", pos.RAGGED)
    specs = resolve_impls(["eager"])
    fixed = probe(cfg, specs, "cuda", 1234, 1, 3, 4, fresh_args=False)
    fresh = probe(cfg, specs, "cuda", 1234, 1, 3, 4, fresh_args=True)
    assert all(r["fresh_args"] is False for r in fixed)
    assert all(r["fresh_args"] is True for r in fresh)
    assert {r["spread"] for r in fresh} == {s for s, _ in rungs(128)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sets_land_on_the_requested_device():
    p = spread_position_sets(4, 128, 8, 2, seed=1, device="cuda")[0]
    assert p.is_cuda and p.dtype == torch.long
