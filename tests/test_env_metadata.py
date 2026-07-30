import pytest
import torch

import benchmarks.benchmark_utils as bu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_nvidia_smi_query_is_scoped_to_the_selected_device(monkeypatch):
    """Regression: this used to query nvidia-smi without --id, so a multi-GPU machine got
    one driver-version line per GPU instead of the one for the device actually measured."""
    calls = []
    real_capture = bu._capture

    def spy(args):
        calls.append(args)
        return real_capture(args)

    monkeypatch.setattr(bu, "_capture", spy)
    bu.env_metadata(device_index=0)

    nvidia_smi_calls = [c for c in calls if c[0] == "nvidia-smi"]
    assert nvidia_smi_calls, "expected env_metadata to call nvidia-smi"
    assert any("--id=0" in c for c in nvidia_smi_calls)
