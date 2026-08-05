"""A real-weights decode loop, built to measure what fraction of a step the operator is.

Sliding windows are disabled on the config before anything is constructed. Mistral-7B-v0.1
declares sliding_window=4096, which makes StaticCache build StaticSlidingWindowLayer, whose
update() rolls the entire cache tensor once full -- O(window) per layer per step, not the O(1)
append this project studies. Patching that away in a rung would attribute eviction cost to
RoPE+append and inflate the headline saving.
"""
import gc
from dataclasses import dataclass, asdict

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import StaticCache

from benchmarks.rung_patches import assert_non_sliding

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

HF_EAGER = "hf_eager"
HF_COMPILE = "hf_compile"
HF_STATIC_GRAPH = "hf_static_graph"
MODES = (HF_EAGER, HF_COMPILE, HF_STATIC_GRAPH)

DEFAULT_MODEL = "mistralai/Mistral-7B-v0.1"


@dataclass(frozen=True)
class DecodeConfig:
    model_id: str
    batch: int
    ctx: int
    dtype_label: str = "bf16"

    @property
    def dtype(self):
        return DTYPES[self.dtype_label]

    def as_row(self):
        return asdict(self)

    def label(self):
        short = self.model_id.split("/")[-1]
        return f"{short}_b{self.batch}_ctx{self.ctx}_{self.dtype_label}"


def load_model(cfg, device="cuda"):
    config = AutoConfig.from_pretrained(cfg.model_id)
    text = config.get_text_config(decoder=True)
    text.sliding_window = None
    text.layer_types = ["full_attention"] * text.num_hidden_layers
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, config=config, dtype=cfg.dtype, attn_implementation="sdpa")
    model.to(device).eval()
    return model


def kv_bytes_per_token(config, dtype):
    text = config.get_text_config(decoder=True)
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    elem = torch.empty(0, dtype=dtype).element_size()
    return 2 * text.num_key_value_heads * head_dim * elem * text.num_hidden_layers


def footprint_bytes(config, cfg, max_cache_len):
    """Weights already resident plus the KV this config would allocate. Excludes activations
    and inductor workspace, which is what `activation_reserve_gb` covers at the call site."""
    return cfg.batch * max_cache_len * kv_bytes_per_token(config, cfg.dtype)


def build_cache(model, cfg, max_cache_len, device="cuda"):
    cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    assert_non_sliding(cache)
    return cache


def prefill(model, cache, cfg, device="cuda"):
    """Never compiled. StaticLayer.lazy_initialization only calls mark_static_address when not
    tracing, and without it cudagraphs are skipped; HF's own docstring says compiling prefill
    under reduce-overhead is known to fail."""
    vocab = model.config.get_text_config(decoder=True).vocab_size
    g = torch.Generator(device="cpu").manual_seed(1234)
    input_ids = torch.randint(0, vocab, (cfg.batch, cfg.ctx), generator=g).to(device)
    cache_position = torch.arange(cfg.ctx, device=device)
    with torch.inference_mode():
        model(input_ids=input_ids, past_key_values=cache, cache_position=cache_position,
              use_cache=True)
    for layer in cache.layers:
        if not layer.is_initialized:
            raise RuntimeError("prefill did not initialize every cache layer")
    return input_ids


def assert_static_addresses(cache):
    for i, layer in enumerate(cache.layers):
        for name in ("keys", "values"):
            t = getattr(layer, name)
            if not getattr(t, "_dynamo_static_input_type", None):
                raise RuntimeError(
                    f"cache layer {i}.{name} is not marked static; cudagraphs would be skipped")


def build_callable(model, mode):
    if mode == HF_EAGER:
        return model
    if mode == HF_COMPILE:
        return torch.compile(model, fullgraph=True)
    if mode == HF_STATIC_GRAPH:
        return torch.compile(model, mode="reduce-overhead", fullgraph=True)
    raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")


def counter_state(cache):
    return [int(l.cumulative_length) for l in cache.layers]


def reset_counters(cache, to_length):
    for layer in cache.layers:
        layer.cumulative_length.fill_(to_length)


def make_step(callable_, cache, cfg, prefill_len, device="cuda"):
    """One decode step. The timed region holds the forward and nothing else -- no sampling,
    no detokenize, no .item(), no logits post-processing."""
    g = torch.Generator(device="cpu").manual_seed(99)
    vocab = callable_.config.get_text_config(decoder=True).vocab_size if hasattr(
        callable_, "config") else 32000
    next_ids = torch.randint(0, vocab, (cfg.batch, 1), generator=g).to(device)
    cache_position = torch.tensor([prefill_len], device=device)

    def step():
        callable_(input_ids=next_ids, past_key_values=cache, cache_position=cache_position,
                  use_cache=True)
    return step


def release(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
