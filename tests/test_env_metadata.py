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
    assert all("--id=0" in c for c in nvidia_smi_calls)


def test_env_metadata_records_the_requested_gpu_state_fields(monkeypatch):
    real_capture = bu._capture

    def fake(args):
        if any(a.startswith("--query-gpu=clocks.sm") for a in args):
            return "2100, 14001, 52, 48.25, 0x0000000000000000"
        return real_capture(args)

    monkeypatch.setattr(bu, "_capture", fake)
    meta = bu.env_metadata(device_index=0)
    assert meta["gpu_state_start"] == {
        "clocks.sm": "2100",
        "clocks.mem": "14001",
        "temperature.gpu": "52",
        "power.draw": "48.25",
        "clocks_throttle_reasons.active": "0x0000000000000000",
    }


def test_clock_lock_scopes_lock_and_restore_to_the_selected_device(monkeypatch):
    calls = []
    monkeypatch.setattr(bu, "_capture", lambda _: "2550")

    def run(args):
        calls.append(args)
        return bu.subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(bu, "_run_nvidia_smi", run)
    with bu.gpu_clock_lock(2, True) as status:
        assert status["locked"] is True
        assert status["target_sm_mhz"] == "2550"
    assert calls == [
        ["nvidia-smi", "--id=2", "-lgc", "2550,2550"],
        ["nvidia-smi", "--id=2", "-rgc"],
    ]


def test_clock_lock_failure_continues_without_reset(monkeypatch, capsys):
    monkeypatch.setattr(bu, "_capture", lambda _: "2550")
    calls = []

    def run(args):
        calls.append(args)
        return bu.subprocess.CompletedProcess(args, 1, "", "Insufficient Permissions")

    monkeypatch.setattr(bu, "_run_nvidia_smi", run)
    with bu.gpu_clock_lock(0, True) as status:
        assert status["locked"] is False
    assert len(calls) == 1
    assert "Insufficient Permissions; continuing unlocked" in capsys.readouterr().err
