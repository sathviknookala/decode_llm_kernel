import pytest
import torch

from benchmarks import positions as pos
from benchmarks.validation import (
    ValidationError,
    validate_candidate,
    validate_or_raise,
)
from benchmarks.workload import Config, eager_impl
from decode_kernels.reference import apply_rope

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
SEED = 1234

MHA_BF16 = Config("mha", 8, 8, 64, 4, 128, "bf16", pos.RAGGED)
GQA_BF16 = Config("gqa", 8, 2, 64, 4, 128, "bf16", pos.RAGGED)


def _cfgs(dtype_label, mode):
    return [
        Config("mha", 8, 8, 64, 4, 128, dtype_label, mode),
        Config("gqa", 8, 2, 64, 4, 128, dtype_label, mode),
    ]


def _rope(x, cos, sin, positions):
    cos_p = cos[positions].to(torch.float32)
    sin_p = sin[positions].to(torch.float32)
    return apply_rope(x.to(torch.float32), cos_p, sin_p).to(x.dtype)


def correct_alternate(q, k, v, positions, cos, sin, k_cache, v_cache, request_indices):
    """Independent but semantically identical implementation: positive control."""
    q_rot = _rope(q, cos, sin, positions)
    k_rot = _rope(k, cos, sin, positions)
    for i in range(q.shape[0]):
        k_cache[request_indices[i], positions[i]] = k_rot[i]
        v_cache[request_indices[i], positions[i]] = v[i]
    return q_rot


def unrotated_q(q, k, v, positions, cos, sin, k_cache, v_cache, request_indices):
    k_cache[request_indices, positions] = _rope(k, cos, sin, positions)
    v_cache[request_indices, positions] = v
    return q.clone()


def unrotated_k_in_cache(q, k, v, positions, cos, sin, k_cache, v_cache, request_indices):
    k_cache[request_indices, positions] = k
    v_cache[request_indices, positions] = v
    return _rope(q, cos, sin, positions)


def rotated_v_in_cache(q, k, v, positions, cos, sin, k_cache, v_cache, request_indices):
    k_cache[request_indices, positions] = _rope(k, cos, sin, positions)
    v_cache[request_indices, positions] = _rope(v, cos, sin, positions)
    return _rope(q, cos, sin, positions)


def wrong_slot(q, k, v, positions, cos, sin, k_cache, v_cache, request_indices):
    shifted = torch.clamp(positions - 1, min=0)
    k_cache[request_indices, shifted] = _rope(k, cos, sin, positions)
    v_cache[request_indices, shifted] = v
    return _rope(q, cos, sin, positions)


def clobbers_unrelated_slot(q, k, v, positions, cos, sin, k_cache, v_cache, request_indices):
    k_cache[request_indices, positions] = _rope(k, cos, sin, positions)
    v_cache[request_indices, positions] = v
    k_cache[0, 0] = 0.0
    return _rope(q, cos, sin, positions)


@pytest.mark.parametrize("dtype_label", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("mode", [pos.UNIFORM, pos.RAGGED])
def test_oracle_validates_against_itself(dtype_label, mode):
    for cfg in _cfgs(dtype_label, mode):
        report = validate_candidate(eager_impl(), cfg, DEVICE, SEED)
        assert report["ok"], report["failures"]
        assert report["v_byte_exact"]
        assert report["unaddressed_slots_intact"]


@pytest.mark.parametrize("dtype_label", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("mode", [pos.UNIFORM, pos.RAGGED])
def test_independent_correct_implementation_passes(dtype_label, mode):
    for cfg in _cfgs(dtype_label, mode):
        report = validate_candidate(correct_alternate, cfg, DEVICE, SEED)
        assert report["ok"], report["failures"]


@pytest.mark.parametrize("broken,expect", [
    (unrotated_q, "q_rot"),
    (unrotated_k_in_cache, "k_cache"),
    (rotated_v_in_cache, "v_cache"),
    (wrong_slot, "cache"),
    (clobbers_unrelated_slot, "unaddressed"),
])
def test_broken_implementations_are_caught(broken, expect):
    report = validate_candidate(broken, MHA_BF16, DEVICE, SEED)
    assert not report["ok"]
    assert any(expect in f for f in report["failures"]), report["failures"]


def test_clobbered_unrelated_slot_flags_intact_false():
    report = validate_candidate(clobbers_unrelated_slot, MHA_BF16, DEVICE, SEED)
    assert report["unaddressed_slots_intact"] is False


def test_rotated_v_flags_byte_exact_false():
    report = validate_candidate(rotated_v_in_cache, MHA_BF16, DEVICE, SEED)
    assert report["v_byte_exact"] is False


def test_validation_covers_both_position_modes():
    ragged = validate_candidate(eager_impl(), MHA_BF16, DEVICE, SEED)
    uniform_cfg = Config("mha", 8, 8, 64, 4, 128, "bf16", pos.UNIFORM)
    uniform = validate_candidate(eager_impl(), uniform_cfg, DEVICE, SEED)
    assert ragged["num_position_sets"] == 3
    assert uniform["num_position_sets"] == 2


def test_gqa_broken_implementation_caught():
    report = validate_candidate(unrotated_k_in_cache, GQA_BF16, DEVICE, SEED)
    assert not report["ok"]


def test_validate_or_raise_raises_on_failure():
    with pytest.raises(ValidationError):
        validate_or_raise(unrotated_q, MHA_BF16, DEVICE, SEED)


def test_validate_or_raise_returns_report_on_success():
    report = validate_or_raise(eager_impl(), MHA_BF16, DEVICE, SEED)
    assert report["ok"]
