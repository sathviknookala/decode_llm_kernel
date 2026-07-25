import pytest
import torch

from benchmarks.benchmark_utils import (
    cache_footprint_bytes,
    dtype_bytes,
    logical_bytes,
    logical_eff_gbps,
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
HEADS = [("mha", 32, 32), ("gqa", 32, 8), ("mqa", 32, 1)]


def _tensors(num_tokens, num_q_heads, num_kv_heads, head_dim, dtype,
             table_dtype=torch.float32, cache_dtype=None, alloc=16):
    cache_dtype = cache_dtype or dtype
    q = torch.empty(num_tokens, num_q_heads, head_dim, dtype=dtype)
    k = torch.empty(num_tokens, num_kv_heads, head_dim, dtype=dtype)
    v = torch.empty(num_tokens, num_kv_heads, head_dim, dtype=dtype)
    cos = torch.empty(alloc, head_dim, dtype=table_dtype)
    sin = torch.empty(alloc, head_dim, dtype=table_dtype)
    kc = torch.empty(num_tokens, alloc, num_kv_heads, head_dim, dtype=cache_dtype)
    vc = torch.empty(num_tokens, alloc, num_kv_heads, head_dim, dtype=cache_dtype)
    return q, k, v, cos, sin, kc, vc


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("label,hq,hkv", HEADS)
def test_byte_formula_matches_hand_computation(dtype, label, hq, hkv):
    t, d = 5, 64
    lb = logical_bytes(*_tensors(t, hq, hkv, d, dtype))
    e = dtype_bytes(dtype)
    assert lb["reads"]["q"] == t * hq * d * e
    assert lb["reads"]["k"] == t * hkv * d * e
    assert lb["reads"]["v"] == t * hkv * d * e
    # split-half stores each row duplicated: only head_dim/2 unique scalars per table
    assert lb["reads"]["cos"] == t * (d // 2) * 4
    assert lb["reads"]["sin"] == t * (d // 2) * 4
    assert lb["writes"]["q_rot"] == t * hq * d * e
    assert lb["writes"]["k_cache"] == t * hkv * d * e
    assert lb["writes"]["v_cache"] == t * hkv * d * e
    assert lb["read_bytes"] == sum(lb["reads"].values())
    assert lb["write_bytes"] == sum(lb["writes"].values())
    assert lb["total_bytes"] == lb["read_bytes"] + lb["write_bytes"]


@pytest.mark.parametrize("dtype", DTYPES)
def test_byte_formula_uses_actual_element_size_not_assumed_fp32(dtype):
    t, hq, hkv, d = 3, 8, 8, 64
    lb = logical_bytes(*_tensors(t, hq, hkv, d, dtype, table_dtype=torch.float32))
    assert lb["reads"]["q"] == t * hq * d * dtype_bytes(dtype)
    assert lb["reads"]["cos"] == t * (d // 2) * dtype_bytes(torch.float32)


@pytest.mark.parametrize("table_dtype", DTYPES)
def test_table_dtype_is_read_from_tensor(table_dtype):
    t, d = 4, 32
    lb = logical_bytes(*_tensors(t, 8, 8, d, torch.bfloat16, table_dtype=table_dtype))
    assert lb["reads"]["cos"] == t * (d // 2) * dtype_bytes(table_dtype)
    assert lb["reads"]["sin"] == t * (d // 2) * dtype_bytes(table_dtype)


def test_mixed_activation_and_cache_dtypes():
    t, hq, hkv, d = 2, 16, 4, 128
    lb = logical_bytes(*_tensors(t, hq, hkv, d, torch.bfloat16, cache_dtype=torch.float16))
    assert lb["writes"]["k_cache"] == t * hkv * d * dtype_bytes(torch.float16)
    assert lb["writes"]["q_rot"] == t * hq * d * dtype_bytes(torch.bfloat16)


def test_gqa_moves_strictly_less_than_mha():
    t, d = 8, 128
    mha = logical_bytes(*_tensors(t, 32, 32, d, torch.bfloat16))["total_bytes"]
    gqa = logical_bytes(*_tensors(t, 32, 8, d, torch.bfloat16))["total_bytes"]
    assert gqa < mha


def test_bytes_scale_linearly_in_tokens():
    d = 64
    one = logical_bytes(*_tensors(1, 8, 8, d, torch.float32))["total_bytes"]
    ten = logical_bytes(*_tensors(10, 8, 8, d, torch.float32))["total_bytes"]
    assert ten == 10 * one


def test_logical_eff_gbps():
    assert logical_eff_gbps(1e9, 1000.0) == pytest.approx(1.0)
    assert logical_eff_gbps(2e9, 1.0) == pytest.approx(2000.0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_cache_footprint(dtype):
    got = cache_footprint_bytes(4, 2048, 8, 128, dtype)
    assert got == 2 * 4 * 2048 * 8 * 128 * dtype_bytes(dtype)
