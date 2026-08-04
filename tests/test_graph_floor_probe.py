import pytest
import torch

from benchmarks import positions as pos
from benchmarks.probe_graph_floor import PROBE_CONFIGS, STAGES, build_stages
from benchmarks.workload import Config

CFG = Config("mha", 8, 8, 64, 4, 128, "fp32", pos.RAGGED)
SEED = 1234

STAGE_NAMES = [name for name, _ in STAGES]


def test_every_stage_is_described_and_named_once():
    assert len(set(STAGE_NAMES)) == len(STAGE_NAMES)
    assert all(description for _, description in STAGES)


def test_the_ladder_brackets_the_impl_it_decomposes():
    """The first stage must launch nothing and the last must be the timed impl, or the
    subtraction attributes launch cost to the operator."""
    assert STAGE_NAMES[0] == "harness_noop"
    assert STAGE_NAMES[-1] == "graph_compile"
    assert "copy_plus_minimal_replay" in STAGE_NAMES


def test_the_unservable_stage_says_so_in_its_description():
    """Replaying with frozen positions cannot serve decode. This repo has already had one
    orientation-only number reach a headline claim; the label is the guard."""
    described = dict(STAGES)["real_replay_no_copy"]
    assert "NOT SERVABLE" in described


def test_probe_configs_span_batch_and_dtype():
    assert {c.num_requests for c in PROBE_CONFIGS} >= {1, 128}
    assert {c.dtype_label for c in PROBE_CONFIGS} >= {"bf16", "fp32"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_build_stages_returns_a_callable_per_stage():
    thunks, keepalive = build_stages(CFG, "cuda", SEED)
    assert set(thunks) == set(STAGE_NAMES)
    for name in STAGE_NAMES:
        thunks[name]()
    torch.cuda.synchronize()
    del thunks, keepalive


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_the_noop_stage_really_launches_nothing():
    thunks, keepalive = build_stages(CFG, "cuda", SEED)
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    start.record()
    thunks["harness_noop"]()
    end.record()
    torch.cuda.synchronize()
    assert start.elapsed_time(end) < 0.05      # ms; a real launch does not fit in 50 us
    del thunks, keepalive


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_the_full_stage_advances_positions_while_the_frozen_stage_does_not():
    thunks, keepalive = build_stages(CFG, "cuda", SEED)
    _args, runner, *_ = keepalive
    static = runner._static_positions

    thunks["real_replay_no_copy"]()
    torch.cuda.synchronize()
    frozen = static.clone()

    for _ in range(4):
        thunks["graph_compile"]()
    torch.cuda.synchronize()
    assert not torch.equal(static, frozen)
    del thunks, keepalive
