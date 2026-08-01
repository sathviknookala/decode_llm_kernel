import pytest
import torch

from benchmarks.bandwidth_reference import (
    measure_bandwidth_reference,
    measure_scattered_write,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_l2_cache_size_is_read_from_the_measured_device(monkeypatch):
    """Regression: this used to hardcode device 0 regardless of which device was measured."""
    torch.cuda.init()  # force lazy CUDA init before spying, or its internal capability
                       # checks call get_device_properties(0) and contaminate `seen`
    seen = []
    real_get_device_properties = torch.cuda.get_device_properties

    def spy(device):
        seen.append(device)
        return real_get_device_properties(device)

    monkeypatch.setattr(torch.cuda, "get_device_properties", spy)
    measure_bandwidth_reference("cuda:0", buffer_mib=8, warmup=1, iters=2,
                                size_sweep_mib=(), scatter_spec=None)
    assert seen == ["cuda:0"]


def test_copy_size_sweep_keeps_the_primary_copy_as_the_continuity_reference():
    ref = measure_bandwidth_reference("cuda:0", buffer_mib=1, warmup=1, iters=2,
                                      size_sweep_mib=(1, 4), scatter_spec=None)
    assert ref["reference_gbps"] == ref["copy"]["gbps_median"]
    assert [m["buffer_mib"] for m in ref["copy_size_sweep"]] == [1, 4]
    assert ref["copy_size_sweep"][0] is ref["copy"]


def test_scattered_write_uses_kv_slot_granularity_and_one_cache_set():
    got = measure_scattered_write(
        "cuda:0", num_requests=4, cache_alloc_len=64, num_kv_heads=2, head_dim=16,
        warmup=1, iters=2, num_position_sets=2)
    assert got["slot_bytes_per_cache"] == 2 * 16 * 4
    assert got["bytes_moved_per_iter"] == 4 * 4 * 2 * 16 * 4
    assert got["cache_footprint_bytes"] == 2 * 4 * 64 * 2 * 16 * 4
    assert got["gbps_median"] > 0
