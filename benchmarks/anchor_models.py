from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorModel:
    """A real released model whose attention shape one swept head layout reproduces.

    Shapes here are declarations, cross-checked against the published config.json by
    tests/test_anchor_models.py. Nothing in this module reads model weights.
    """
    head_label: str
    model_id: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    max_position: int
    license: str
    gated: bool
    citation: str

    @property
    def hidden_size(self):
        return self.num_q_heads * self.head_dim

    def head_config(self) -> tuple[str, int, int, int]:
        return (self.head_label, self.num_q_heads, self.num_kv_heads, self.head_dim)


LLAMA2_7B = AnchorModel(
    head_label="mha",
    model_id="meta-llama/Llama-2-7b-hf",
    num_q_heads=32,
    num_kv_heads=32,
    head_dim=128,
    rope_theta=10000.0,
    max_position=4096,
    license="llama2",
    gated=True,
    citation="Touvron et al. 2023, arXiv:2307.09288",
)

# Chosen as the GQA anchor over Llama-3-8B specifically because it is ungated: its 32:8 @ 128
# shape is identical to the GQA row the sweep already ran, so no timing was invalidated.
MISTRAL_7B = AnchorModel(
    head_label="gqa",
    model_id="mistralai/Mistral-7B-v0.1",
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=128,
    rope_theta=10000.0,
    max_position=32768,
    license="apache-2.0",
    gated=False,
    citation="Jiang et al. 2023, arXiv:2310.06825",
)

ANCHOR_MODELS = (LLAMA2_7B, MISTRAL_7B)


def by_head_label(head_label):
    for m in ANCHOR_MODELS:
        if m.head_label == head_label:
            return m
    raise KeyError(f"no anchor model for head label {head_label!r}")


def _rope_theta(cfg):
    # transformers 5.x moved rope_theta into a rope_parameters dict; 4.x had it top-level.
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict) and "rope_theta" in params:
        return float(params["rope_theta"])
    theta = getattr(cfg, "rope_theta", None)
    if theta is None:
        raise AttributeError(f"no rope_theta on {type(cfg).__name__}")
    return float(theta)


def _num_kv_heads(cfg):
    # Falcon predates num_key_value_heads: it declares multi-query with a flag, and its
    # num_kv_heads field is stale and ignored unless the new decoder architecture is on.
    if getattr(cfg, "multi_query", False) and not getattr(cfg, "new_decoder_architecture", False):
        return 1
    for name in ("num_key_value_heads", "num_kv_heads"):
        value = getattr(cfg, name, None)
        if value is not None:
            return int(value)
    return int(cfg.num_attention_heads)


def published_shape(model_id):
    """Attention shape as published on the Hub, for cross-checking the declarations above.

    Reads config.json only -- no weights, no tokenizer. Needs transformers plus either a
    warm HF cache or network; callers are expected to skip when neither is available.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return {
        "num_q_heads": cfg.num_attention_heads,
        "num_kv_heads": _num_kv_heads(cfg),
        "head_dim": head_dim,
        "hidden_size": cfg.hidden_size,
        "rope_theta": _rope_theta(cfg),
        "max_position": cfg.max_position_embeddings,
    }
