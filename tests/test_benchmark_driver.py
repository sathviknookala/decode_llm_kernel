from types import SimpleNamespace

import pytest
import torch

from benchmarks import positions as pos
from benchmarks.benchmark_operator import bench_impl, graph_savings, run_config
from benchmarks.impls import DirectRunner
from benchmarks.workload import (
    IDENTITY,
    PACKED,
    PERMUTED,
    STRIDED,
    Config,
    build_matrix,
    build_op_args,
    build_position_sets,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

CFG = Config("mha", 8, 8, 64, 4, 128, "fp32", pos.RAGGED)
SEED = 1234


def _timed(runner, warmup=2, iters=5):
    position_sets = build_position_sets(CFG, SEED, "cuda", num_sets=2)
    args = build_op_args(CFG, "cuda", SEED, position_sets[0])
    return bench_impl(runner, args, position_sets, warmup, iters, 553.0)


def test_timing_runs_the_impl_under_inference_mode():
    seen = []
    _timed(DirectRunner(lambda *a: seen.append(torch.is_inference_mode_enabled())))
    assert seen and all(seen)


def test_thunk_construction_is_inside_the_inference_region():
    seen = []

    class RecordingRunner(DirectRunner):
        def make_thunk(self, args, position_sets):
            seen.append(torch.is_inference_mode_enabled())
            return super().make_thunk(args, position_sets)

    _timed(RecordingRunner(lambda *a: None))
    assert seen == [True]


def test_timed_row_carries_metrics_and_byte_accounting():
    position_sets = build_position_sets(CFG, SEED, "cuda", num_sets=2)
    args = build_op_args(CFG, "cuda", SEED, position_sets[0])
    row = bench_impl(DirectRunner(lambda *a: None), args, position_sets, 2, 5,
                     553.0, 200.0)
    assert row["device_median_ms"] >= 0
    assert row["logical_total_bytes"] > 0
    assert row["pct_of_empirical_bw"] != ""
    assert row["pct_of_scattered_write_bw"] != ""


def test_capture_failure_becomes_an_error_row():
    class FailingSpec:
        label = "graph_broken"

        @staticmethod
        def build(*_):
            def fail(*_):
                raise RuntimeError("capture permission denied")
            return DirectRunner(fail)

    args = SimpleNamespace(seed=SEED, compile_mode=None, compile_backend="inductor",
                           warmup=1, iters=1)
    rows = run_config(CFG, "cuda", args, None, [FailingSpec()])
    assert rows[0]["validation"] == "ERROR"
    assert "capture permission denied" in rows[0]["validation_detail"]
    assert rows[0]["device_median_ms"] == ""


def test_graph_savings_pairs_matching_configs_only():
    cfg = CFG.as_row()
    other = {**cfg, "num_requests": 32}
    rows = [
        {"impl": "compile", **cfg, "validation": "pass", "device_median_ms": 0.04},
        {"impl": "graph_compile", **cfg, "validation": "pass", "device_median_ms": 0.01},
        {"impl": "compile", **other, "validation": "pass", "device_median_ms": 0.02},
    ]
    got = graph_savings(rows)
    assert len(got) == 1
    assert got[0]["latency_removed_pct"] == pytest.approx(75.0)
    assert got[0]["config"]["num_requests"] == CFG.num_requests


def test_serving_layout_matrix_is_crossed_only_at_b1_and_b32():
    cfgs = list(build_matrix(
        quick=True, head_labels=["mha"], batches=[1, 8, 32], allocs=[128],
        dtypes=["bf16"], position_modes=(pos.RAGGED,)))
    main = [c for c in cfgs if (c.layout, c.request_mapping) == (PACKED, IDENTITY)]
    variants = [c for c in cfgs if c not in main]
    assert [c.num_requests for c in main] == [1, 8, 32]
    assert {c.num_requests for c in variants} == {1, 32}
    assert {(c.layout, c.request_mapping) for c in variants} == {
        (PACKED, PERMUTED), (STRIDED, IDENTITY), (STRIDED, PERMUTED),
    }
    assert len(variants) == 6


def test_config_columns_and_label_preserve_the_main_matrix_names():
    assert CFG.as_row()["layout"] == PACKED
    assert CFG.as_row()["request_mapping"] == IDENTITY
    assert CFG.label() == "mha_b4_alloc128_fp32_ragged"
    variant = Config("mha", 8, 8, 64, 4, 128, "fp32", pos.RAGGED,
                     STRIDED, PERMUTED)
    assert variant.label().endswith("_strided_permuted")


def test_timed_config_builds_strided_qkv_and_permuted_requests():
    cfg = Config("mha", 8, 8, 64, 4, 128, "fp32", pos.RAGGED,
                 STRIDED, PERMUTED)
    positions = build_position_sets(cfg, SEED, "cuda", num_sets=1)[0]
    op_args = build_op_args(cfg, "cuda", SEED, positions)
    assert not op_args.q.is_contiguous()
    assert op_args.q.stride(-1) == 1
    assert not torch.equal(op_args.request_indices, torch.arange(4, device="cuda"))
