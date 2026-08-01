import pytest
import torch

from benchmarks import positions as pos
from benchmarks.benchmark_operator import bench_impl
from benchmarks.impls import DirectRunner
from benchmarks.workload import Config, build_op_args, build_position_sets

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
    row = _timed(DirectRunner(lambda *a: None))
    assert row["device_median_ms"] >= 0
    assert row["logical_total_bytes"] > 0
    assert row["pct_of_empirical_bw"] != ""
