import pytest
import torch

from benchmarks.probe_cache_read import (
    ARMS,
    HEAD_MAJOR,
    READ_FLOOR,
    TOKEN_MAJOR_COPY,
    TOKEN_MAJOR_VIEW,
    build_arm,
    cache_read_bytes,
    cache_shape,
    footprint_gb,
    probe_one,
    with_layout_ratio,
)


def test_head_major_is_sdpas_layout_and_token_major_is_the_locked_one():
    assert cache_shape(HEAD_MAJOR, 4, 8, 512, 128) == (4, 8, 512, 128)
    for arm in (TOKEN_MAJOR_VIEW, TOKEN_MAJOR_COPY):
        assert cache_shape(arm, 4, 8, 512, 128) == (4, 512, 8, 128)


def test_read_bytes_count_both_caches_and_follow_the_dtype():
    bf16 = cache_read_bytes(4, 8, 512, 128, torch.bfloat16)
    assert bf16 == 2 * 4 * 8 * 512 * 128 * 2
    assert cache_read_bytes(4, 8, 512, 128, torch.float32) == 2 * bf16


def test_read_bytes_scale_with_context_not_with_query_heads():
    """A decode step reads the whole cache; the number of query heads does not enter, which is
    why GQA moves so much less than MHA here."""
    assert (cache_read_bytes(1, 8, 1024, 128, torch.bfloat16)
            == 2 * cache_read_bytes(1, 8, 512, 128, torch.bfloat16))


def test_the_footprint_guard_accounts_for_the_copy_arms_second_buffer():
    assert footprint_gb(1, 8, 512, 128, torch.bfloat16) == pytest.approx(
        2 * cache_read_bytes(1, 8, 512, 128, torch.bfloat16) / 1e9)


def test_ratios_are_against_head_major():
    rows = [{"arm": HEAD_MAJOR, "amortized_call_ms": 0.10},
            {"arm": TOKEN_MAJOR_VIEW, "amortized_call_ms": 0.11},
            {"arm": TOKEN_MAJOR_COPY, "amortized_call_ms": 0.30}]
    got = {r["arm"]: r["vs_head_major"] for r in with_layout_ratio(rows)}
    assert got[HEAD_MAJOR] == pytest.approx(1.0)
    assert got[TOKEN_MAJOR_VIEW] == pytest.approx(1.1)
    assert got[TOKEN_MAJOR_COPY] == pytest.approx(3.0)


def test_a_head_major_arm_that_failed_leaves_ratios_blank():
    rows = [{"arm": HEAD_MAJOR, "amortized_call_ms": ""},
            {"arm": TOKEN_MAJOR_VIEW, "amortized_call_ms": 0.11}]
    assert all(r["vs_head_major"] == "" for r in with_layout_ratio(rows))


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="unknown arm"):
        build_arm("head_minor", 1, 8, 8, 128, 64, torch.float32, "cpu", 1)


def test_the_view_arm_really_reads_a_strided_cache():
    """If the transposed view were contiguous the arm would be measuring head-major twice."""
    _, _, (k, _, _) = build_arm(TOKEN_MAJOR_VIEW, 2, 8, 8, 128, 64, torch.float32, "cpu", 1)
    assert k.is_contiguous()
    assert not k.transpose(1, 2).is_contiguous()


def test_gqa_configs_take_the_enable_gqa_path_rather_than_materializing_heads():
    """Expanding KV heads to match Q would multiply the bytes read and silently make the GQA
    rows incomparable to the MHA ones."""
    _, gqa_path, _ = build_arm(HEAD_MAJOR, 1, 32, 8, 128, 64, torch.float32, "cpu", 1)
    assert gqa_path == "enable_gqa"
    _, mha_path, _ = build_arm(HEAD_MAJOR, 1, 8, 8, 128, 64, torch.float32, "cpu", 1)
    assert mha_path == "none"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_every_arm_produces_a_timed_row_with_its_bytes():
    rows = probe_one(("mha", 8, 8, 64), 1, 128, "bf16", "cuda", 1234, 1, 3, ARMS)
    assert [r["arm"] for r in rows] == list(ARMS)
    for r in rows:
        assert r["error"] == ""
        assert r["amortized_call_ms"] > 0
        assert r["cache_read_bytes"] == cache_read_bytes(1, 8, 128, 64, torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_the_read_floor_arm_needs_no_attention_backend():
    rows = probe_one(("mqa", 71, 1, 64), 1, 128, "bf16", "cuda", 1234, 1, 3, [READ_FLOOR])
    assert rows[0]["gqa_path"] == "n/a"
    assert rows[0]["read_gbps"] > 0
