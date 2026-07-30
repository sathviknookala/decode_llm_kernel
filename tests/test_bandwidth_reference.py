import pytest
import torch

from benchmarks.bandwidth_reference import measure_bandwidth_reference

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
    measure_bandwidth_reference("cuda:0", buffer_mib=8, warmup=1, iters=2)
    assert seen == ["cuda:0"]
