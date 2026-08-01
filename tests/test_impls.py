from dataclasses import replace

import pytest
import torch

from benchmarks import positions as pos
from benchmarks.impls import (
    DEFAULT_IMPLS,
    IMPL_LABELS,
    IMPL_SPECS,
    DirectRunner,
    GraphRunner,
    resolve_impls,
)
from benchmarks.validation import validate_candidate
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_runner_reuses_capture_for_new_position_tensors_and_recaptures_inputs():
    position_sets = build_position_sets(CFG, SEED, "cuda", num_sets=2)
    args = build_op_args(CFG, "cuda", SEED, position_sets[0])
    runner = GraphRunner(lambda *a: a[0] + a[3][:, None, None], warmup=1)

    first = runner(*args).clone()
    second = runner(*args._replace(positions=position_sets[1])).clone()
    assert runner.capture_count == 1
    assert not torch.equal(first, second)

    fresh = build_op_args(CFG, "cuda", SEED, position_sets[0])
    runner(*fresh)
    assert runner.capture_count == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("impl_label", ["graph_eager", "graph_compile"])
@pytest.mark.parametrize("position_mode", [pos.UNIFORM, pos.RAGGED])
def test_graph_impls_pass_the_full_validation_gate(impl_label, position_mode):
    cfg = replace(CFG, position_mode=position_mode)
    runner = resolve_impls([impl_label])[0].build()
    report = validate_candidate(runner, cfg, "cuda", SEED)
    assert report["ok"], report["failures"]
    assert {"permuted-requests", "strided-qkv"}.issubset(report["cases"])
