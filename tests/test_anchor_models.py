from types import SimpleNamespace

import pytest
import torch

from benchmarks.anchor_models import (
    ANCHOR_MODELS,
    FALCON_7B,
    LLAMA2_7B,
    MISTRAL_7B,
    _num_kv_heads,
    by_head_label,
    published_shape,
)
from benchmarks.workload import HEAD_CONFIGS, build_matrix
from decode_kernels.reference import build_rope_tables


def _published_or_skip(model_id):
    """config.json only. Skips rather than fails when transformers is absent or the Hub is
    unreachable with a cold cache -- an offline machine must not fail the suite."""
    pytest.importorskip("transformers")
    try:
        return published_shape(model_id)
    except Exception as e:                                  # noqa: BLE001 -- network/cache/API drift
        pytest.skip(f"cannot read published config for {model_id}: {type(e).__name__}: {e}")


def test_head_configs_track_the_registry_in_declaration_order():
    assert HEAD_CONFIGS == [m.head_config() for m in ANCHOR_MODELS]


def test_the_originally_swept_layouts_are_still_swept_unchanged():
    """New anchors may extend the matrix but must not restate the two layouts the earlier
    baselines were measured on, or old and new CSVs stop overlapping at all."""
    assert HEAD_CONFIGS[:2] == [("mha", 32, 32, 128), ("gqa", 32, 8, 128)]


def test_registry_covers_more_than_one_head_dim():
    """head_dim sets the vectorisation width, so a single-valued registry would let a kernel
    be tuned to one width with nothing in the rig to reveal it."""
    assert len({m.head_dim for m in ANCHOR_MODELS}) > 1


def test_every_swept_head_layout_is_backed_by_an_anchor_model():
    for head_label, hq, hkv, hd in HEAD_CONFIGS:
        m = by_head_label(head_label)
        assert (m.num_q_heads, m.num_kv_heads, m.head_dim) == (hq, hkv, hd)


def test_head_labels_are_unique():
    labels = [m.head_label for m in ANCHOR_MODELS]
    assert len(labels) == len(set(labels))


def test_unknown_head_label_raises():
    with pytest.raises(KeyError):
        by_head_label("sideways")


def test_multi_query_config_reports_one_kv_head():
    """Falcon-7B's shape: no num_key_value_heads at all, and a num_kv_heads field that its
    own attention ignores while multi_query is set. Read literally it looks like MHA."""
    falcon = SimpleNamespace(num_attention_heads=71, num_kv_heads=71, multi_query=True,
                             new_decoder_architecture=False)
    assert _num_kv_heads(falcon) == 1


def test_new_decoder_architecture_honours_num_kv_heads():
    falcon_40b = SimpleNamespace(num_attention_heads=128, num_kv_heads=8, multi_query=True,
                                 new_decoder_architecture=True)
    assert _num_kv_heads(falcon_40b) == 8


def test_standard_config_reads_num_key_value_heads():
    mistral = SimpleNamespace(num_attention_heads=32, num_key_value_heads=8)
    assert _num_kv_heads(mistral) == 8


def test_config_without_any_kv_field_falls_back_to_mha():
    llama = SimpleNamespace(num_attention_heads=32)
    assert _num_kv_heads(llama) == 32


@pytest.mark.parametrize("model", ANCHOR_MODELS, ids=lambda m: m.head_label)
def test_hidden_size_is_consistent_with_the_head_split(model):
    assert model.hidden_size == model.num_q_heads * model.head_dim


@pytest.mark.parametrize("model", ANCHOR_MODELS, ids=lambda m: m.head_label)
def test_head_dim_is_even_as_split_half_rope_requires(model):
    assert model.head_dim % 2 == 0


@pytest.mark.parametrize("model", ANCHOR_MODELS, ids=lambda m: m.head_label)
def test_q_heads_divide_evenly_into_kv_groups(model):
    assert model.num_q_heads % model.num_kv_heads == 0


def test_mha_anchor_really_is_mha_and_gqa_anchor_really_is_gqa():
    assert LLAMA2_7B.num_kv_heads == LLAMA2_7B.num_q_heads
    assert MISTRAL_7B.num_kv_heads < MISTRAL_7B.num_q_heads


def test_mqa_anchor_really_is_mqa():
    assert FALCON_7B.num_kv_heads == 1
    assert FALCON_7B.num_q_heads > 1


def test_the_gqa_anchor_is_ungated_which_is_why_it_was_chosen():
    """The MHA anchor's weights are gated; the GQA anchor's are not. This is the whole reason
    Mistral-7B-v0.1 is the GQA anchor rather than a gated GQA model."""
    assert MISTRAL_7B.gated is False
    assert MISTRAL_7B.license == "apache-2.0"


@pytest.mark.parametrize("model", ANCHOR_MODELS, ids=lambda m: m.head_label)
def test_every_anchor_model_carries_a_citation(model):
    assert "arXiv:" in model.citation


@pytest.mark.parametrize("model", ANCHOR_MODELS, ids=lambda m: m.head_label)
def test_swept_cache_alloc_lens_fit_the_anchor_context_window(model):
    """A cache longer than the model's own context would make the config unrepresentable."""
    allocs = {cfg.cache_alloc_len for cfg in build_matrix() if cfg.head_label == model.head_label}
    assert allocs, f"no swept configs for {model.head_label}"
    assert max(allocs) <= model.max_position


@pytest.mark.parametrize("model", ANCHOR_MODELS, ids=lambda m: m.head_label)
def test_declared_shape_matches_the_published_config(model):
    """The drift-catcher: our declared shapes vs the actual config.json on the Hub."""
    got = _published_or_skip(model.model_id)
    assert got["num_q_heads"] == model.num_q_heads
    assert got["num_kv_heads"] == model.num_kv_heads
    assert got["head_dim"] == model.head_dim
    assert got["hidden_size"] == model.hidden_size
    assert got["rope_theta"] == model.rope_theta
    assert got["max_position"] == model.max_position


def test_rope_tables_match_mistrals_own_rotary_module():
    """Our FP32 tables vs MistralRotaryEmbedding driven by the real published config --
    pins inv_freq, theta and the cat(freqs, freqs) split-half layout against the model's
    own code. Config only; no weights are loaded."""
    pytest.importorskip("transformers")
    _published_or_skip(MISTRAL_7B.model_id)                 # skip early if the Hub is cold
    from transformers import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding

    cfg = MistralConfig.from_pretrained(MISTRAL_7B.model_id)
    rot = MistralRotaryEmbedding(cfg, device="cpu")

    max_pos = MISTRAL_7B.max_position
    positions = torch.tensor([0, 1, 2, 7, 4095, max_pos - 1])
    cos, sin = build_rope_tables(max_pos, MISTRAL_7B.head_dim, MISTRAL_7B.rope_theta)
    hf_cos, hf_sin = rot(torch.zeros(1, positions.numel()), positions.unsqueeze(0))

    torch.testing.assert_close(cos[positions], hf_cos[0], atol=0, rtol=0)
    torch.testing.assert_close(sin[positions], hf_sin[0], atol=0, rtol=0)
