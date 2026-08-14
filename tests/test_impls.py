from dataclasses import replace

import pytest
import torch

from benchmarks import positions as pos
from benchmarks.impls import (
    BASES,
    CUDA_IMPLS,
    CUDAGRAPH_MODES,
    DEFAULT_IMPLS,
    IMPL_LABELS,
    IMPL_SPECS,
    DirectRunner,
    GraphRunner,
    resolve_impls,
)
from benchmarks.validation import validate_candidate
from benchmarks.workload import Config, build_op_args, build_position_sets

BY_LABEL = {s.label: s for s in IMPL_SPECS}
CFG = Config("mha", 8, 8, 64, 4, 128, "fp32", pos.RAGGED)
SEED = 1234


def cuda_ext_available():
    try:
        from decode_kernels import cuda
    except ImportError:
        return False
    return cuda.is_available()


def test_labels_are_unique_and_the_default_is_the_baseline_ladder():
    """The default is no longer the whole registry: the mode variants recompile per config and
    would multiply a sweep's wall clock, so they are opt-in via --impls."""
    assert len(set(IMPL_LABELS)) == len(IMPL_LABELS)
    assert set(DEFAULT_IMPLS) == {"eager", "compile", "graph_eager", "graph_compile"}
    assert set(DEFAULT_IMPLS).issubset(IMPL_LABELS)


def test_a_specs_own_mode_wins_over_the_runs():
    spec = BY_LABEL["compile_max_autotune"]
    assert spec.resolve(None, "inductor")[0] == "max-autotune-no-cudagraphs"
    assert spec.resolve("reduce-overhead", "inductor")[0] == "max-autotune-no-cudagraphs"


def test_an_unpinned_spec_takes_the_runs_mode():
    assert BY_LABEL["compile"].resolve("max-autotune", "inductor") == ("max-autotune", "inductor")


def test_eager_reports_no_compile_settings():
    """Recording the run's compile mode on an eager row would invent provenance it never used."""
    assert BY_LABEL["eager"].resolve("max-autotune", "aot_eager") == (None, None)
    assert BY_LABEL["graph_eager"].resolve("max-autotune", "aot_eager") == (None, None)


@pytest.mark.parametrize("mode", ["reduce-overhead", "max-autotune"])
def test_inductors_own_cudagraphs_are_refused_inside_graph_capture(mode):
    """Capturing a graph whose body already replays an inductor graph measures nesting, not
    the operator. The registry's own graph specs must therefore never pin such a mode."""
    with pytest.raises(ValueError, match="cannot be captured"):
        BY_LABEL["graph_compile"].resolve(mode, "inductor")
    for spec in IMPL_SPECS:
        if spec.graph:
            assert spec.mode not in CUDAGRAPH_MODES


def test_every_spec_declares_a_known_base():
    for spec in IMPL_SPECS:
        assert spec.base in BASES
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


def test_the_custom_kernel_rungs_are_registered_as_five_and_six():
    assert set(CUDA_IMPLS) == {"cuda_separate", "cuda_fused",
                               "graph_cuda_separate", "graph_cuda_fused"}
    assert set(CUDA_IMPLS).issubset(IMPL_LABELS)


def test_the_custom_rungs_stay_out_of_the_default_sweep():
    """Committed baselines must stay a subset of a default run, and these need a built
    extension -- so they are opt-in via --impls, like the compile-mode variants."""
    assert not set(CUDA_IMPLS) & set(DEFAULT_IMPLS)


def test_a_custom_kernel_row_reports_no_compile_provenance():
    """Eager rows do not carry the run's inductor settings and neither should a hand-written
    kernel; reporting them would invent provenance."""
    for label in CUDA_IMPLS:
        spec = resolve_impls([label])[0]
        assert spec.resolve(mode="max-autotune", backend="inductor") == (None, None)


def test_an_unbuilt_extension_fails_before_the_first_config(monkeypatch):
    """A missing _ext raises inside spec.build(), which run_config records as an ERROR row -- so
    without this the whole sweep would be ERROR rows instead of one line saying how to build it."""
    import benchmarks.impls as impls_mod
    monkeypatch.setattr(impls_mod.cuda, "is_available", lambda: False)
    monkeypatch.setattr(impls_mod.cuda, "unavailable_reason", lambda: "no compiled _ext")
    with pytest.raises(SystemExit, match="build_ext"):
        impls_mod.assert_extension_available(resolve_impls(["cuda_fused"]))
    impls_mod.assert_extension_available(resolve_impls(["eager", "compile"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not cuda_ext_available(), reason="CUDA extension not built")
@pytest.mark.parametrize("impl_label", ["cuda_separate", "cuda_fused",
                                        "graph_cuda_separate", "graph_cuda_fused"])
def test_custom_kernel_impls_pass_the_full_validation_gate(impl_label):
    runner = resolve_impls([impl_label])[0].build()
    report = validate_candidate(runner, CFG, "cuda", SEED)
    assert report["ok"], report["failures"]
    assert {"permuted-requests", "strided-qkv"}.issubset(report["cases"])
