"""The ablation ladder: each rung is a reversible patch over the HF decode path.

Five of the six rungs are numerically invalid and timing-valid -- they corrupt the model's output
on purpose to remove or duplicate work while holding every shape, dtype and allocation fixed.
Only `full` and `op_replaced_ref` produce meaningful logits.
"""
import contextlib
import sys
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, StaticLayer
from transformers.models.mistral import modeling_mistral

from decode_kernels.reference import apply_rope

FULL = "full"
OP_REMOVED = "op_removed"
ROPE_REMOVED = "rope_removed"
APPEND_REMOVED = "append_removed"
OP_DOUBLED = "op_doubled"
OP_REPLACED_REF = "op_replaced_ref"


@dataclass(frozen=True)
class Rung:
    label: str
    description: str
    numerically_valid: bool
    # decode steps the rung advances the cache counter by, per model step. op_doubled writes twice.
    slots_per_step: int = 1


RUNGS = (
    Rung(FULL, "unmodified decode step", True),
    Rung(OP_REMOVED, "RoPE skipped and cache write skipped -- upper bound on a fused kernel", False),
    Rung(ROPE_REMOVED, "RoPE skipped, cache write kept", False),
    Rung(APPEND_REMOVED, "cache write skipped, RoPE kept", False),
    Rung(OP_DOUBLED, "RoPE and cache write applied twice -- the Amdahl slope", False, 2),
    Rung(OP_REPLACED_REF, "HF's pair replaced by fused_rope_kv_append_ref", True),
)
RUNG_LABELS = tuple(r.label for r in RUNGS)
BY_LABEL = {r.label: r for r in RUNGS}


def resolve_rungs(labels):
    unknown = [l for l in labels if l not in BY_LABEL]
    if unknown:
        raise ValueError(f"unknown rung(s) {unknown}, expected any of {list(RUNG_LABELS)}")
    return [BY_LABEL[l] for l in labels]


@dataclass(frozen=True)
class Arch:
    """Where a given model architecture keeps the two things the ladder patches.

    Resolved from the live model rather than imported, because patching
    `models.mistral.modeling_mistral` while running Phi-3 installs cleanly, does nothing, and
    reports a clean 0% operator share -- the inert-patch failure the meta-test exists to catch.
    """
    model_type: str
    module: object
    attention_cls: type
    ref_forward: object = None

    @property
    def supports_ref_replacement(self):
        return self.ref_forward is not None


def resolve_arch(model):
    """The modeling module the attention layer's `apply_rotary_pos_emb` actually resolves in."""
    attn = _first_attention(model)
    module = sys.modules[type(attn).__module__]
    if not hasattr(module, "apply_rotary_pos_emb"):
        raise RuntimeError(
            f"{module.__name__} has no apply_rotary_pos_emb; the RoPE rungs have nothing to "
            "patch and would silently measure nothing")
    model_type = getattr(model.config.get_text_config(decoder=True), "model_type", "unknown")
    return Arch(model_type, module, type(attn), REF_FORWARDS.get(model_type))


def _first_attention(model):
    for m in model.modules():
        if type(m).__name__.endswith("Attention") and hasattr(m, "layer_idx"):
            return m
    raise RuntimeError("no attention module with a layer_idx found; cannot resolve patch targets")


def assert_patch_target_reachable(model, arch=None):
    """`kernels` would reroute apply_rotary_pos_emb past the monkeypatch without erroring."""
    import importlib.util
    if importlib.util.find_spec("kernels") is not None:
        raise RuntimeError(
            "the `kernels` package is installed; apply_rotary_pos_emb may be routed to a hub "
            "kernel and the rung patches would silently not apply")
    if getattr(model, "use_kernels", False):
        raise RuntimeError("model.use_kernels is True; rung patches would silently not apply")
    arch = arch or resolve_arch(model)
    if arch.module.apply_rotary_pos_emb is not _rope_of(arch):
        raise RuntimeError(f"{arch.module.__name__}.apply_rotary_pos_emb is already patched")
    return arch


def _rope_of(arch):
    return getattr(arch.module, "apply_rotary_pos_emb")


def _identity_rope(q, k, cos, sin, unsqueeze_dim=1):
    return q, k


def _no_write_update(self, key_states, value_states, layer_idx, *args, **kwargs):
    """Return what attention would have read, without writing and without advancing the counter."""
    layer = self.layers[layer_idx]
    if not layer.is_initialized:
        layer.lazy_initialization(key_states, value_states)
    return layer.keys, layer.values


def _doubled_update(orig):
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        orig(self, key_states, value_states, layer_idx, *args, **kwargs)
        return orig(self, key_states, value_states, layer_idx, *args, **kwargs)
    return update


def _ref_rope_append(q, k, v, cos, sin, layer, positions, request_indices):
    """`fused_rope_kv_append_ref` adapted to HF's seam.

    Two deviations, both recorded in LIMITATIONS: HF hoists the cos/sin table gather out of the
    per-layer loop into MistralRotaryEmbedding, so the per-layer operation genuinely does not
    include it; and HF's cache is [B, H, S, D], so the append writes through a transposed view.
    """
    cos_p = cos.squeeze(1).to(torch.float32)          # [B, S=1, D] -> [T, D]
    sin_p = sin.squeeze(1).to(torch.float32)
    q_rot = apply_rope(q.squeeze(2).to(torch.float32), cos_p, sin_p).to(q.dtype)
    k_rot = apply_rope(k.squeeze(2).to(torch.float32), cos_p, sin_p).to(k.dtype)
    keys = layer.keys.transpose(1, 2)
    values = layer.values.transpose(1, 2)
    keys[request_indices, positions] = k_rot.to(keys.dtype)
    values[request_indices, positions] = v.squeeze(2).to(values.dtype)
    layer.cumulative_length.add_(1)
    return q_rot.unsqueeze(2), layer.keys, layer.values


def _ref_attention_forward(self, hidden_states, position_embeddings, attention_mask,
                           past_key_values=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    layer = past_key_values.layers[self.layer_idx]
    positions = layer.cumulative_length.expand(hidden_states.shape[0])
    request_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    query_states, key_states, value_states = _ref_rope_append(
        query_states, key_states, value_states, cos, sin, layer, positions, request_indices)

    attention_interface = modeling_mistral.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, modeling_mistral.eager_attention_forward)
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0, scaling=self.scaling,
        sliding_window=getattr(self.config, "sliding_window", None), **kwargs)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


# op_replaced_ref needs an adapter per architecture: the seam depends on how the model shapes
# Q/K/V (Phi-3 fuses the projection) and on the attention-interface call. Architectures without
# one refuse the rung rather than running it as a no-op.
REF_FORWARDS = {"mistral": _ref_attention_forward}


@contextlib.contextmanager
def _patched(targets):
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in targets]
    try:
        for obj, name, new in targets:
            setattr(obj, name, new)
        yield
    finally:
        for obj, name, old in saved:
            setattr(obj, name, old)


MISTRAL_ARCH = Arch("mistral", modeling_mistral, modeling_mistral.MistralAttention,
                    _ref_attention_forward)


@contextlib.contextmanager
def apply_rung(label, arch=MISTRAL_ARCH):
    """Install a rung's patches on `arch`. Reverts on exit, including on exception."""
    rung = BY_LABEL[label]
    orig_rope = _rope_of(arch)
    orig_update = Cache.update
    targets = []

    if rung.label == FULL:
        yield rung
        return

    if rung.label in (OP_REMOVED, ROPE_REMOVED):
        targets.append((arch.module, "apply_rotary_pos_emb", _identity_rope))
    if rung.label in (OP_REMOVED, APPEND_REMOVED):
        targets.append((Cache, "update", _no_write_update))
    if rung.label == OP_DOUBLED:
        def doubled_rope(q, k, cos, sin, unsqueeze_dim=1):
            q, k = orig_rope(q, k, cos, sin, unsqueeze_dim)
            return orig_rope(q, k, cos, sin, unsqueeze_dim)
        targets.append((arch.module, "apply_rotary_pos_emb", doubled_rope))
        targets.append((Cache, "update", _doubled_update(orig_update)))
    if rung.label == OP_REPLACED_REF:
        if not arch.supports_ref_replacement:
            raise NotImplementedError(
                f"op_replaced_ref has no seam adapter for {arch.model_type!r}; the rung would "
                "patch nothing and time an unmodified step. Write one or exclude the rung with "
                "--rungs.")
        targets.append((arch.attention_cls, "forward", arch.ref_forward))

    with _patched(targets):
        yield rung


def cache_layers(cache):
    return [l for l in cache.layers if isinstance(l, StaticLayer)]


def assert_non_sliding(cache):
    for i, layer in enumerate(cache.layers):
        if getattr(layer, "is_sliding", False):
            raise RuntimeError(
                f"cache layer {i} is {type(layer).__name__}: sliding-window eviction rolls the "
                "whole cache every step and would be attributed to RoPE+append")


def assert_no_roll(cache):
    """Sliding layers roll once cumulative_length reaches max_cache_len. Non-sliding layers
    index_copy_ out of bounds instead, so overrun is a hard error either way."""
    for i, layer in enumerate(cache.layers):
        if not layer.is_initialized:
            continue
        used = int(layer.cumulative_length)
        if used >= layer.max_cache_len:
            raise RuntimeError(
                f"cache layer {i} reached {used}/{layer.max_cache_len} slots; "
                "the run overran its headroom")
