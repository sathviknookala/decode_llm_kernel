import pytest
import torch

from benchmarks import positions as pos
from benchmarks.probe_l2_residency import (
    SLOT_LADDER,
    disjoint_slot_sets,
    write_working_set_bytes,
)
from benchmarks.workload import Config

CFG = Config("mha", 32, 32, 128, 128, 2048, "bf16", pos.RAGGED)


def test_slot_sets_are_pairwise_disjoint():
    """Disjointness is the whole probe: overlapping sets would keep the same slots hot and
    the write working set would not grow with the number of sets."""
    sets = disjoint_slot_sets(8, 2048, 16, seed=1234, device="cpu")
    seen = set()
    for p in sets:
        slots = set(p.tolist())
        assert not (slots & seen)
        seen |= slots


def test_slot_sets_stay_in_bounds_and_keep_one_token_per_slot():
    for n in (1, 8, 512):
        for p in disjoint_slot_sets(4, 2048, n, seed=7, device="cpu"):
            assert int(p.min()) >= 0 and int(p.max()) < 2048
            assert p.numel() == 4          # request_indices are distinct, so (r, p) stays unique


def test_more_sets_than_slots_is_refused():
    with pytest.raises(ValueError, match="cache_alloc_len"):
        disjoint_slot_sets(4, 128, 129, seed=7, device="cpu")


def test_working_set_counts_both_caches_and_scales_with_sets():
    one = write_working_set_bytes(CFG, 1)
    assert one == 2 * 128 * 1 * 32 * 128 * 2
    assert write_working_set_bytes(CFG, 32) == 32 * one


def test_the_ladder_crosses_the_l2_boundary_for_the_widest_layout():
    """A ladder that stayed under L2 would answer nothing."""
    l2 = 48 * 1024 ** 2
    assert min(write_working_set_bytes(CFG, n) for n in SLOT_LADDER) < l2
    assert max(write_working_set_bytes(CFG, n) for n in SLOT_LADDER) > l2


def test_dtype_is_read_from_the_config_not_assumed():
    fp32 = Config("mha", 32, 32, 128, 128, 2048, "fp32", pos.RAGGED)
    assert write_working_set_bytes(fp32, 8) == 2 * write_working_set_bytes(CFG, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sets_land_on_the_requested_device():
    p = disjoint_slot_sets(4, 128, 4, seed=1, device="cuda")[0]
    assert p.is_cuda and p.dtype == torch.long


def test_the_slot_ladder_is_bracketed_by_its_first_rung():
    """The probe's reported effect is under 2 percent; measured elsewhere on this rig, the
    first-rung artifact alone was worth 25, so the ladder cannot be read without the control."""
    from benchmarks.benchmark_utils import bracketed
    ladder = bracketed([n for n in SLOT_LADDER if n <= 2048])
    assert ladder[0] == ladder[-1] == SLOT_LADDER[0]
    assert ladder[:-1] == list(SLOT_LADDER)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_the_probe_reports_drift_beside_its_effect():
    from benchmarks.impls import resolve_impls
    from benchmarks.probe_l2_residency import probe
    cfg = Config("mha", 8, 8, 64, 2, 128, "bf16", pos.RAGGED)
    rows = probe(cfg, resolve_impls(["eager"]), "cuda", 1234, 1, 3, 48 * 1024 ** 2)
    assert sum(r["is_baseline_rung"] for r in rows) == 2
    assert all(r["ordering_drift"] != "" for r in rows)
