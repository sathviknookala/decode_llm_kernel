import torch


def build_rope_tables(max_position, head_dim, theta=10000.0, *, device="cpu",
                      dtype=torch.float32):
    assert head_dim % 2 == 0, "head_dim must be even for split-half RoPE"
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32,
                                             device=device) * 2.0 / head_dim))
    t = torch.arange(max_position, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)              # [max_position, half]
    emb = torch.cat((freqs, freqs), dim=-1)       # [max_position, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: [T, H, D]; cos, sin: [T, D] already gathered at each token's position
    cos = cos.unsqueeze(-2)                       # [T, 1, D] -> broadcast over heads
    sin = sin.unsqueeze(-2)
    return x * cos + rotate_half(x) * sin


def fused_rope_kv_append_ref(q, k, v, positions, cos, sin, k_cache, v_cache,
                             request_indices):
    # q:[T,Hq,D] k,v:[T,Hkv,D] positions:[T] cos,sin:[max_pos,D]
    # k_cache,v_cache:[B,max_seq,Hkv,D] request_indices:[T]
    cos_p = cos[positions].to(torch.float32)      # gather + FP32 trig
    sin_p = sin[positions].to(torch.float32)
    q_rot = apply_rope(q.to(torch.float32), cos_p, sin_p).to(q.dtype)
    k_rot = apply_rope(k.to(torch.float32), cos_p, sin_p).to(k.dtype)
    k_cache[request_indices, positions] = k_rot.to(k_cache.dtype)
    v_cache[request_indices, positions] = v.to(v_cache.dtype)
    return q_rot
