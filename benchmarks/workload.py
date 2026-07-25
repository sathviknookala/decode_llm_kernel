import itertools
from dataclasses import dataclass, asdict
from typing import NamedTuple

import torch
import torch._dynamo

from decode_kernels.reference import build_rope_tables, fused_rope_kv_append_ref
from benchmarks import positions as pos

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# Llama-2-7B anchor is MHA 32/32 @ head_dim 128; the GQA row is the synthetic 4:1 variant.
HEAD_CONFIGS = [
    ("mha", 32, 32, 128),
    ("gqa", 32, 8, 128),
]

CACHE_SENTINEL = 7.5
NUM_POSITION_SETS = 8


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


def build_op_args(cfg, device, seed, positions, *, sentinel_cache=False):
    """Allocate one invocation's tensors. RoPE tables stay FP32 per locked semantics."""
    t, hq, hkv, d = cfg.num_requests, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim
    dt = cfg.dtype
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(t, hq, d, generator=g, dtype=dt, device=device)
    k = torch.randn(t, hkv, d, generator=g, dtype=dt, device=device)
    v = torch.randn(t, hkv, d, generator=g, dtype=dt, device=device)
    cos, sin = build_rope_tables(cfg.cache_alloc_len, d, theta=10000.0, device=device,
                                 dtype=torch.float32)
    request_indices = torch.arange(t, device=device)
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


def eager_impl():
    return fused_rope_kv_append_ref


def compile_impl(mode=None, backend="inductor"):
    """Fresh compile per config: dynamo caches on the code object and would otherwise
    hit recompile_limit mid-sweep and silently fall back to eager."""
    torch._dynamo.reset()
    kwargs = {"backend": backend, "dynamic": False}
    if mode:
        kwargs["mode"] = mode
    return torch.compile(fused_rope_kv_append_ref, **kwargs)


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
