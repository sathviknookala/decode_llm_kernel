import torch
import pytest

from decode_kernels.reference import (
    build_rope_tables,
    rotate_half,
    apply_rope,
    fused_rope_kv_append_ref,
)

THETA = 10000.0


def test_rope_matches_hf_llama():
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    T, H, D, max_pos = 5, 4, 8, 16
    cos, sin = build_rope_tables(max_pos, D, THETA)
    positions = torch.tensor([0, 1, 2, 7, 15])
    x = torch.randn(T, H, D, dtype=torch.float32)

    mine = apply_rope(x, cos[positions], sin[positions])              # [T,H,D]

    xb = x.permute(1, 0, 2).unsqueeze(0)                             # [1,H,T,D]
    cosb, sinb = cos[positions].unsqueeze(0), sin[positions].unsqueeze(0)
    q_hf, _ = apply_rotary_pos_emb(xb, xb, cosb, sinb)              # unsqueeze_dim=1
    hf = q_hf.squeeze(0).permute(1, 0, 2)

    torch.testing.assert_close(mine, hf, atol=1e-6, rtol=1e-6)


def test_rope_matches_hf_mistral():
    """Mistral ships its own copy of apply_rotary_pos_emb (not an alias of Llama's), and it is
    the GQA anchor's model family, so parity is asserted against it separately. Rotates Q and
    K together at the anchor's real 32:8 head split."""
    from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

    T, Hq, Hkv, D, max_pos = 5, 32, 8, 128, 4096
    cos, sin = build_rope_tables(max_pos, D, THETA)
    positions = torch.tensor([0, 1, 2, 1023, 4095])
    q = torch.randn(T, Hq, D, dtype=torch.float32)
    k = torch.randn(T, Hkv, D, dtype=torch.float32)

    mine_q = apply_rope(q, cos[positions], sin[positions])            # [T,Hq,D]
    mine_k = apply_rope(k, cos[positions], sin[positions])            # [T,Hkv,D]

    qb = q.permute(1, 0, 2).unsqueeze(0)                             # [1,H,T,D]
    kb = k.permute(1, 0, 2).unsqueeze(0)
    cosb, sinb = cos[positions].unsqueeze(0), sin[positions].unsqueeze(0)
    q_hf, k_hf = apply_rotary_pos_emb(qb, kb, cosb, sinb)            # unsqueeze_dim=1

    torch.testing.assert_close(mine_q, q_hf.squeeze(0).permute(1, 0, 2), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(mine_k, k_hf.squeeze(0).permute(1, 0, 2), atol=1e-6, rtol=1e-6)


def test_fused_op_on_the_gqa_anchors_real_shape():
    """End-to-end oracle run at Mistral-7B-v0.1's exact attention shape (32:8 @ 128)."""
    from benchmarks.anchor_models import MISTRAL_7B as m

    T, B, max_seq = 4, 4, 2048
    cos, sin = build_rope_tables(max_seq, m.head_dim, m.rope_theta)
    q = torch.randn(T, m.num_q_heads, m.head_dim)
    k = torch.randn(T, m.num_kv_heads, m.head_dim)
    v = torch.randn(T, m.num_kv_heads, m.head_dim)
    positions = torch.tensor([0, 17, 1023, 2047])
    request_indices = torch.arange(T)
    k_cache = torch.zeros(B, max_seq, m.num_kv_heads, m.head_dim)
    v_cache = torch.zeros(B, max_seq, m.num_kv_heads, m.head_dim)

    q_rot = fused_rope_kv_append_ref(q, k, v, positions, cos, sin, k_cache, v_cache,
                                     request_indices)

    assert q_rot.shape == (T, m.num_q_heads, m.head_dim)
    k_exp = apply_rope(k.float(), cos[positions], sin[positions])
    for t in range(T):
        torch.testing.assert_close(k_cache[request_indices[t], positions[t]], k_exp[t])
        torch.testing.assert_close(v_cache[request_indices[t], positions[t]], v[t])


def test_rotate_half_and_pairing():
    D = 8
    x = torch.arange(D, dtype=torch.float32).reshape(1, 1, D)
    r = rotate_half(x)
    # pair i couples dim i with dim i+D/2 -> [-x2, x1]
    assert torch.equal(r[0, 0], torch.tensor([-4., -5., -6., -7., 0., 1., 2., 3.]))


def test_zero_position_is_identity():
    T, H, D, max_pos = 3, 2, 8, 16
    cos, sin = build_rope_tables(max_pos, D, THETA)
    x = torch.randn(T, H, D)
    out = apply_rope(x, cos[torch.zeros(T, dtype=torch.long)],
                     sin[torch.zeros(T, dtype=torch.long)])
    torch.testing.assert_close(out, x)                              # pos 0: cos=1, sin=0


def _run_case(dtype, T=4, Hq=8, Hkv=8, D=8, B=4, max_seq=16, theta=THETA):
    cos, sin = build_rope_tables(max_seq, D, theta)
    q = torch.randn(T, Hq, D, dtype=dtype)
    k = torch.randn(T, Hkv, D, dtype=dtype)
    v = torch.randn(T, Hkv, D, dtype=dtype)
    positions = torch.arange(T)
    request_indices = torch.arange(T)
    k_cache = torch.zeros(B, max_seq, Hkv, D, dtype=dtype)
    v_cache = torch.zeros(B, max_seq, Hkv, D, dtype=dtype)
    q_rot = fused_rope_kv_append_ref(q, k, v, positions, cos, sin, k_cache,
                                     v_cache, request_indices)
    return locals()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_returned_q_is_rotated(dtype):
    c = _run_case(dtype)
    expect = apply_rope(c["q"].float(), c["cos"][c["positions"]],
                        c["sin"][c["positions"]]).to(dtype)
    tol = dict(atol=1e-6, rtol=1e-6) if dtype == torch.float32 else dict(atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(c["q_rot"], expect, **tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cache_slots_receive_rotated_k_and_raw_v(dtype):
    c = _run_case(dtype)
    k_exp = apply_rope(c["k"].float(), c["cos"][c["positions"]],
                       c["sin"][c["positions"]]).to(dtype)
    for t in range(c["T"]):
        r, p = c["request_indices"][t], c["positions"][t]
        torch.testing.assert_close(c["k_cache"][r, p], k_exp[t])
        torch.testing.assert_close(c["v_cache"][r, p], c["v"][t])       # V unrotated


def test_unrelated_cache_entries_unchanged():
    c = _run_case(torch.float32)
    written = torch.zeros(c["B"], c["max_seq"], dtype=torch.bool)
    written[c["request_indices"], c["positions"]] = True
    # every non-written (b, seq) slot must remain exactly zero
    assert torch.equal(c["k_cache"][~written], torch.zeros_like(c["k_cache"][~written]))
    assert torch.equal(c["v_cache"][~written], torch.zeros_like(c["v_cache"][~written]))


def test_deterministic():
    torch.manual_seed(0)
    c1 = _run_case(torch.float32)
    torch.manual_seed(0)
    c2 = _run_case(torch.float32)
    torch.testing.assert_close(c1["q_rot"], c2["q_rot"])
    torch.testing.assert_close(c1["k_cache"], c2["k_cache"])


def test_gqa_shapes():
    c = _run_case(torch.float32, Hq=8, Hkv=2)                        # GQA 4:1
    assert c["q_rot"].shape == (c["T"], 8, c["D"])
    assert c["k_cache"].shape[2] == 2


def test_non_identity_request_mapping():
    cos, sin = build_rope_tables(16, 8, THETA)
    T, H, D = 4, 4, 8
    q = torch.randn(T, H, D); k = torch.randn(T, H, D); v = torch.randn(T, H, D)
    k_cache = torch.zeros(T, 16, H, D); v_cache = torch.zeros(T, 16, H, D)
    positions = torch.tensor([3, 3, 3, 3])                          # same slot, different rows
    request_indices = torch.tensor([2, 0, 3, 1])
    fused_rope_kv_append_ref(q, k, v, positions, cos, sin, k_cache, v_cache, request_indices)
    k_exp = apply_rope(k.float(), cos[positions], sin[positions])
    for t in range(T):
        torch.testing.assert_close(k_cache[request_indices[t], 3], k_exp[t])
        torch.testing.assert_close(v_cache[request_indices[t], 3], v[t])


def test_strided_qkv_from_fused_projection():
    """Splitting a fused QKV projection gives contiguous head_dim but strided tokens/heads."""
    cos, sin = build_rope_tables(16, 8, THETA)
    T, Hq, Hkv, D = 4, 8, 2, 8
    fused = torch.randn(T, (Hq + 2 * Hkv) * D)
    q, k, v = [x.view(T, -1, D) for x in fused.split([Hq * D, Hkv * D, Hkv * D], dim=-1)]
    assert not q.is_contiguous() and q.stride(-1) == 1

    k_cache = torch.zeros(T, 16, Hkv, D); v_cache = torch.zeros(T, 16, Hkv, D)
    positions = torch.arange(T)
    q_rot = fused_rope_kv_append_ref(q, k, v, positions, cos, sin, k_cache, v_cache,
                                     torch.arange(T))

    torch.testing.assert_close(q_rot, apply_rope(q.float(), cos[positions], sin[positions]))
    k_exp = apply_rope(k.float(), cos[positions], sin[positions])
    for t in range(T):
        torch.testing.assert_close(k_cache[t, positions[t]], k_exp[t])
        torch.testing.assert_close(v_cache[t, positions[t]], v[t])


def test_mqa_single_kv_head():
    c = _run_case(torch.float32, Hq=8, Hkv=1)
    assert c["q_rot"].shape == (c["T"], 8, c["D"])
    k_exp = apply_rope(c["k"].float(), c["cos"][c["positions"]], c["sin"][c["positions"]])
    for t in range(c["T"]):
        torch.testing.assert_close(c["k_cache"][t, c["positions"][t]], k_exp[t])


def test_single_request_and_boundary_positions():
    cos, sin = build_rope_tables(16, 8, THETA)
    q = torch.randn(1, 4, 8); k = torch.randn(1, 4, 8); v = torch.randn(1, 4, 8)
    k_cache = torch.zeros(1, 16, 4, 8); v_cache = torch.zeros(1, 16, 4, 8)
    for pos in (0, 15):                                             # first and last valid slot
        kc, vc = k_cache.clone(), v_cache.clone()
        fused_rope_kv_append_ref(q, k, v, torch.tensor([pos]), cos, sin, kc, vc,
                                 torch.tensor([0]))
        assert kc[0, pos].abs().sum() > 0
