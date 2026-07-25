import pytest
import torch

from benchmarks import positions as pos


def test_uniform_mode_is_last_valid_slot():
    sets = pos.build_position_sets(4, 128, pos.UNIFORM, 0, 3, "cpu")
    for p in sets:
        assert torch.equal(p, torch.full((4,), 127, dtype=torch.long))


@pytest.mark.parametrize("alloc", [128, 512, 2048])
@pytest.mark.parametrize("num_tokens", [1, 8, 128])
def test_ragged_positions_in_bounds(num_tokens, alloc):
    sets = pos.build_position_sets(num_tokens, alloc, pos.RAGGED, 1234, 8, "cpu")
    for p in sets:
        assert p.shape == (num_tokens,)
        assert p.dtype == torch.long
        assert int(p.min()) >= 0
        assert int(p.max()) < alloc


def test_ragged_is_deterministic_for_a_seed():
    a = pos.build_position_sets(16, 512, pos.RAGGED, 7, 4, "cpu")
    b = pos.build_position_sets(16, 512, pos.RAGGED, 7, 4, "cpu")
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_different_seeds_differ():
    a = torch.cat(pos.build_position_sets(16, 512, pos.RAGGED, 1, 4, "cpu"))
    b = torch.cat(pos.build_position_sets(16, 512, pos.RAGGED, 2, 4, "cpu"))
    assert not torch.equal(a, b)


def test_ragged_spans_early_middle_late():
    sets = pos.build_position_sets(8, 2048, pos.RAGGED, 1234, 8, "cpu")
    span = pos.position_span(sets, 2048)
    assert span["in_bounds"]
    assert span["early"] > 0 and span["middle"] > 0 and span["late"] > 0
    assert span["distinct"] > 1


def test_ragged_is_nonuniform_across_requests():
    sets = pos.build_position_sets(32, 2048, pos.RAGGED, 99, 1, "cpu")
    assert int(torch.unique(sets[0]).numel()) > 1


def test_timed_invocations_touch_multiple_slots():
    sets = pos.build_position_sets(1, 2048, pos.RAGGED, 5, 8, "cpu")
    allp = torch.cat(sets)
    # a single-request sweep must not rewrite one slot for every invocation
    assert int(torch.unique(allp).numel()) > 1


def test_tiny_alloc_still_in_bounds():
    for alloc in (1, 2, 3, 4):
        sets = pos.build_position_sets(4, alloc, pos.RAGGED, 3, 4, "cpu")
        for p in sets:
            assert int(p.min()) >= 0 and int(p.max()) < alloc


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        pos.build_position_sets(4, 128, "sideways", 0, 2, "cpu")
