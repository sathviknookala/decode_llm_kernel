import pytest
import torch

from benchmarks import positions as pos
from benchmarks.impls import (
    DEFAULT_IMPLS,
    IMPL_LABELS,
    IMPL_SPECS,
    DirectRunner,
    resolve_impls,
)
from benchmarks.workload import Config, build_op_args, build_position_sets

CFG = Config("mha", 8, 8, 64, 4, 128, "fp32", pos.RAGGED)
SEED = 1234


def test_labels_are_unique_and_default_covers_the_registry():
    assert len(set(IMPL_LABELS)) == len(IMPL_LABELS)
    assert set(DEFAULT_IMPLS) == set(IMPL_LABELS)


def test_every_spec_declares_a_known_base():
    for spec in IMPL_SPECS:
        assert spec.base in ("eager", "compile")
        assert spec.description


def test_resolve_preserves_the_requested_order():
    assert [s.label for s in resolve_impls(["compile", "eager"])] == ["compile", "eager"]


def test_resolve_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="graph_triton"):
        resolve_impls(["eager", "graph_triton"])


def test_direct_runner_forwards_the_call_and_thunk():
    calls = []
    runner = DirectRunner(lambda *a: calls.append(len(a)))
    runner(*range(9))
    assert calls == [9]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_direct_runner_thunk_rotates_positions():
    seen = []
    position_sets = build_position_sets(CFG, SEED, "cuda", num_sets=3)
    args = build_op_args(CFG, "cuda", SEED, position_sets[0])
    runner = DirectRunner(lambda q, k, v, p, *rest: seen.append(p.data_ptr()))
    thunk = runner.make_thunk(args, position_sets)
    for _ in range(3):
        thunk()
    assert seen == [p.data_ptr() for p in position_sets]
