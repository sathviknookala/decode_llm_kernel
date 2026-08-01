import itertools
from dataclasses import dataclass, asdict
from typing import NamedTuple

import torch

from decode_kernels.reference import build_rope_tables
from benchmarks import positions as pos
from benchmarks.anchor_models import ANCHOR_MODELS

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# Every swept head layout is a real released model's attention shape: MHA 32/32 @ 128 is
# Llama-2-7B, GQA 32/8 @ 128 is Mistral-7B-v0.1. Provenance in anchor_models.py.
HEAD_CONFIGS = [m.head_config() for m in ANCHOR_MODELS]

CACHE_SENTINEL = 7.5
NUM_POSITION_SETS = 8

PACKED = "packed"
STRIDED = "strided"
LAYOUTS = (PACKED, STRIDED)

IDENTITY = "identity"
PERMUTED = "permuted"
REQUEST_MAPPINGS = (IDENTITY, PERMUTED)


@dataclass(frozen=True)
class Config:
    head_label: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_requests: int
    cache_alloc_len: int
    dtype_label: str
    position_mode: str

    @property
    def dtype(self):
        return DTYPES[self.dtype_label]

    def as_row(self):
        return asdict(self)

    def label(self):
        return (f"{self.head_label}_b{self.num_requests}_alloc{self.cache_alloc_len}"
                f"_{self.dtype_label}_{self.position_mode}")


class OpArgs(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    positions: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    request_indices: torch.Tensor


def build_position_sets(cfg, seed, device, num_sets=NUM_POSITION_SETS):
    return pos.build_position_sets(cfg.num_requests, cfg.cache_alloc_len,
                                   cfg.position_mode, seed, num_sets, device)


def build_request_mapping(num_requests, kind, seed, device):
    """Which cache row each token targets. A permuted mapping is a bijection, so slots stay
    unique while `token index == request index` stops holding."""
    if kind == IDENTITY:
        return torch.arange(num_requests, device=device)
    if kind == PERMUTED:
        g = torch.Generator().manual_seed(seed + 9973)
        return torch.randperm(num_requests, generator=g).to(device)
    raise ValueError(f"unknown request mapping {kind!r}, expected one of {REQUEST_MAPPINGS}")


def _build_qkv(t, hq, hkv, d, dt, device, g, layout):
    if layout == PACKED:
        return tuple(torch.randn(t, h, d, generator=g, dtype=dt, device=device)
                     for h in (hq, hkv, hkv))
    if layout == STRIDED:
        # views into one fused QKV projection: head_dim contiguous, token stride = fused width
        fused = torch.randn(t, (hq + 2 * hkv) * d, generator=g, dtype=dt, device=device)
        parts = fused.split([hq * d, hkv * d, hkv * d], dim=-1)
        return tuple(p.view(t, -1, d) for p in parts)
    raise ValueError(f"unknown layout {layout!r}, expected one of {LAYOUTS}")


def build_op_args(cfg, device, seed, positions, *, sentinel_cache=False,
                  request_mapping=IDENTITY, layout=PACKED):
    """Allocate one invocation's tensors. RoPE tables stay FP32 per locked semantics."""
    t, hq, hkv, d = cfg.num_requests, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim
    dt = cfg.dtype
    g = torch.Generator(device=device).manual_seed(seed)
    q, k, v = _build_qkv(t, hq, hkv, d, dt, device, g, layout)
    cos, sin = build_rope_tables(cfg.cache_alloc_len, d, theta=10000.0, device=device,
                                 dtype=torch.float32)
    request_indices = build_request_mapping(t, request_mapping, seed, device)
    fill = CACHE_SENTINEL if sentinel_cache else 0.0
    k_cache = torch.full((t, cfg.cache_alloc_len, hkv, d), fill, dtype=dt, device=device)
    v_cache = torch.full((t, cfg.cache_alloc_len, hkv, d), fill, dtype=dt, device=device)
    return OpArgs(q, k, v, positions, cos, sin, k_cache, v_cache, request_indices)


def make_thunk(fn, args, position_sets):
    """Timed callable. Cycles pre-built device position tensors; no host generation or sync."""
    cyc = itertools.cycle(position_sets)
    q, k, v = args.q, args.k, args.v
    cos, sin = args.cos, args.sin
    kc, vc, ri = args.k_cache, args.v_cache, args.request_indices

    def run():
        fn(q, k, v, next(cyc), cos, sin, kc, vc, ri)
    return run


def build_matrix(quick=False, position_modes=(pos.RAGGED,), head_labels=None,
                 batches=None, allocs=None, dtypes=None):
    if quick:
        batches = batches or [1, 32]
        allocs = allocs or [2048]
        dtypes = dtypes or ["bf16"]
        heads = HEAD_CONFIGS
    else:
        batches = batches or [1, 8, 32, 128]
        allocs = allocs or [128, 2048]
        dtypes = dtypes or ["bf16", "fp16", "fp32"]
        heads = HEAD_CONFIGS
    if head_labels:
        heads = [h for h in heads if h[0] in head_labels]
    for (hl, hq, hkv, hd), b, alloc, dl, pm in itertools.product(
            heads, batches, allocs, dtypes, position_modes):
        yield Config(hl, hq, hkv, hd, b, alloc, dl, pm)
