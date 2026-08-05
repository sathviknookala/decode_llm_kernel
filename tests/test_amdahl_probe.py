import contextlib

import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM
from transformers.cache_utils import Cache, StaticLayer, StaticSlidingWindowLayer
from transformers.models.mistral import modeling_mistral

from benchmarks import decode_loop as dl
from benchmarks.rung_patches import (
    APPEND_REMOVED,
    FULL,
    OP_DOUBLED,
    OP_REMOVED,
    OP_REPLACED_REF,
    ROPE_REMOVED,
    RUNGS,
    RUNG_LABELS,
    apply_rung,
    assert_no_roll,
    assert_non_sliding,
    resolve_rungs,
)

CTX = 8
BATCH = 2
HEADROOM = 16

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def tiny_config(sliding_window=4):
    return MistralConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, sliding_window=sliding_window, rope_theta=10000.0,
        attn_implementation="sdpa")


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    config = tiny_config()
    config.sliding_window = None
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    model = MistralForCausalLM(config).to("cuda", torch.float32).eval()
    return model


def _cache_and_prefill(model):
    cfg = dl.DecodeConfig("tiny", BATCH, CTX, "fp32")
    cache = dl.build_cache(model, cfg, CTX + HEADROOM)
    dl.prefill(model, cache, cfg)
    return cfg, cache


def _step_logits(model, cache, cfg):
    ids = torch.full((BATCH, 1), 3, device="cuda")
    pos = torch.tensor([cfg.ctx], device="cuda")
    with torch.inference_mode():
        return model(input_ids=ids, past_key_values=cache, cache_position=pos,
                     use_cache=True).logits


# --- the sliding-window defect this plan exists to avoid -------------------------------------

@cuda_only
def test_an_unpatched_mistral_config_builds_a_sliding_cache():
    """Guard on the premise. If this ever stops holding, the disable in load_model is dead code
    and someone will delete it."""
    from transformers.cache_utils import StaticCache
    cache = StaticCache(config=tiny_config(sliding_window=4), max_cache_len=CTX + HEADROOM)
    assert any(isinstance(l, StaticSlidingWindowLayer) for l in cache.layers)
    with pytest.raises(RuntimeError, match="sliding"):
        assert_non_sliding(cache)


@cuda_only
def test_disabling_the_window_builds_plain_static_layers(tiny_model):
    _cfg, cache = _cache_and_prefill(tiny_model)
    assert all(type(l) is StaticLayer for l in cache.layers)
    assert_non_sliding(cache)


@cuda_only
def test_assert_no_roll_fires_when_headroom_is_overrun(tiny_model):
    _cfg, cache = _cache_and_prefill(tiny_model)
    with torch.inference_mode():
        for layer in cache.layers:
            layer.cumulative_length.fill_(layer.max_cache_len)
    with pytest.raises(RuntimeError, match="overran"):
        assert_no_roll(cache)


# --- the patches do what they claim ----------------------------------------------------------

@pytest.mark.parametrize("label", RUNG_LABELS)
def test_every_rung_is_reversible(label):
    before = (modeling_mistral.apply_rotary_pos_emb, Cache.update,
              modeling_mistral.MistralAttention.forward)
    with apply_rung(label):
        pass
    after = (modeling_mistral.apply_rotary_pos_emb, Cache.update,
             modeling_mistral.MistralAttention.forward)
    assert before == after


@pytest.mark.parametrize("label", RUNG_LABELS)
def test_every_rung_reverts_through_an_exception(label):
    before = (modeling_mistral.apply_rotary_pos_emb, Cache.update)
    with contextlib.suppress(ValueError):
        with apply_rung(label):
            raise ValueError("boom")
    assert (modeling_mistral.apply_rotary_pos_emb, Cache.update) == before


@cuda_only
def test_op_removed_leaves_the_cache_slot_unwritten(tiny_model):
    cfg, cache = _cache_and_prefill(tiny_model)
    slot = cfg.ctx
    with torch.inference_mode():
        for layer in cache.layers:
            layer.keys[:, :, slot].fill_(7.5)
            layer.values[:, :, slot].fill_(7.5)
    with apply_rung(OP_REMOVED):
        _step_logits(tiny_model, cache, cfg)
    for layer in cache.layers:
        assert torch.all(layer.keys[:, :, slot] == 7.5)
        assert torch.all(layer.values[:, :, slot] == 7.5)


@cuda_only
def test_full_does_write_the_cache_slot(tiny_model):
    """The complement of the test above: without it, a cache that was never writable would make
    op_removed look correct."""
    cfg, cache = _cache_and_prefill(tiny_model)
    slot = cfg.ctx
    with torch.inference_mode():
        for layer in cache.layers:
            layer.keys[:, :, slot].fill_(7.5)
    _step_logits(tiny_model, cache, cfg)
    assert all(not torch.all(l.keys[:, :, slot] == 7.5) for l in cache.layers)


@cuda_only
@pytest.mark.parametrize("label", [OP_REMOVED, ROPE_REMOVED, APPEND_REMOVED, OP_DOUBLED])
def test_numerically_invalid_rungs_change_the_logits(tiny_model, label):
    cfg, cache = _cache_and_prefill(tiny_model)
    reference = _step_logits(tiny_model, cache, cfg).clone()
    _cfg2, cache2 = _cache_and_prefill(tiny_model)
    with apply_rung(label):
        patched = _step_logits(tiny_model, cache2, cfg)
    assert not torch.allclose(reference, patched, atol=1e-4), (
        f"{label} left the logits unchanged, so its patch did nothing")


@cuda_only
@pytest.mark.parametrize("label", RUNG_LABELS)
def test_every_rung_preserves_shape_and_dtype(tiny_model, label):
    cfg, cache = _cache_and_prefill(tiny_model)
    reference = _step_logits(tiny_model, cache, cfg)
    _cfg2, cache2 = _cache_and_prefill(tiny_model)
    with apply_rung(label):
        out = _step_logits(tiny_model, cache2, cfg)
    assert out.shape == reference.shape
    assert out.dtype == reference.dtype


@cuda_only
def test_op_replaced_ref_matches_the_unpatched_step(tiny_model):
    """The integration seam must be numerically equivalent, or the ablation is comparing two
    different models rather than two implementations of one operation."""
    cfg, cache = _cache_and_prefill(tiny_model)
    reference = _step_logits(tiny_model, cache, cfg).clone()
    _cfg2, cache2 = _cache_and_prefill(tiny_model)
    with apply_rung(OP_REPLACED_REF):
        replaced = _step_logits(tiny_model, cache2, cfg)
    torch.testing.assert_close(replaced, reference, atol=2e-4, rtol=2e-4)


@cuda_only
def test_op_doubled_advances_the_cache_twice_per_step(tiny_model):
    """Its slots_per_step is what sizes the headroom; if the counter did not double, the
    headroom arithmetic would be wrong in the safe direction and hide a real overrun."""
    cfg, cache = _cache_and_prefill(tiny_model)
    before = int(cache.layers[0].cumulative_length)
    with apply_rung(OP_DOUBLED):
        _step_logits(tiny_model, cache, cfg)
    assert int(cache.layers[0].cumulative_length) - before == 2


# --- the meta-test: prove the gate above catches an inert patch --------------------------------

@cuda_only
def test_an_inert_patch_is_caught_by_the_change_detector(tiny_model):
    """The standard test_benchmark_validation.py sets: prove the check works by feeding it
    something deliberately broken. An inert patch reverts itself before the call, so it looks
    installed and does nothing -- exactly the failure that would report a 0% operator share."""

    @contextlib.contextmanager
    def inert_rung(_label):
        original = modeling_mistral.apply_rotary_pos_emb
        modeling_mistral.apply_rotary_pos_emb = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should never be called"))
        modeling_mistral.apply_rotary_pos_emb = original
        yield

    cfg, cache = _cache_and_prefill(tiny_model)
    reference = _step_logits(tiny_model, cache, cfg).clone()
    _cfg2, cache2 = _cache_and_prefill(tiny_model)
    with inert_rung(OP_REMOVED):
        patched = _step_logits(tiny_model, cache2, cfg)
    assert torch.allclose(reference, patched, atol=1e-4), "premise: the inert patch changed nothing"

    with pytest.raises(AssertionError):
        assert not torch.allclose(reference, patched, atol=1e-4), (
            "op_removed left the logits unchanged, so its patch did nothing")


# --- registry hygiene ---------------------------------------------------------------------

def test_rung_labels_are_unique_and_described():
    assert len(set(RUNG_LABELS)) == len(RUNG_LABELS)
    assert all(r.description for r in RUNGS)


def test_only_full_and_the_ref_replacement_are_numerically_valid():
    valid = {r.label for r in RUNGS if r.numerically_valid}
    assert valid == {FULL, OP_REPLACED_REF}


def test_resolve_rungs_rejects_unknown_labels():
    with pytest.raises(ValueError, match="unknown rung"):
        resolve_rungs(["full", "not_a_rung"])


# --- the footprint guard and crash safety ----------------------------------------------------

from benchmarks.decode_loop import footprint_bytes, kv_bytes_per_token  # noqa: E402
from benchmarks.probe_amdahl import headroom_slots  # noqa: E402


class _Args:
    warmup, iters = 20, 150


def test_kv_bytes_per_token_matches_the_hand_computation():
    """2 (K+V) x kv_heads x head_dim x elem x layers. Mistral-7B: 2*8*128*2*32 = 128 KiB."""
    cfg = tiny_config(sliding_window=None)
    cfg.num_key_value_heads, cfg.head_dim, cfg.num_hidden_layers = 8, 128, 32
    assert kv_bytes_per_token(cfg, torch.bfloat16) == 2 * 8 * 128 * 2 * 32


def test_headroom_covers_the_doubling_rung():
    """op_doubled advances two slots per step, so headroom must be 2x the step count or the
    run overruns the cache mid-measurement."""
    a = _Args()
    assert headroom_slots(a) >= 2 * (a.warmup + a.iters)


def test_the_guard_denies_the_config_that_actually_oomed():
    """b=32 ctx=1024 died with 20.10 GB allocated of a 23.42 GB card. It must be skipped, not
    retried -- and b=32 ctx=512, which ran, must not be."""
    cfg = tiny_config(sliding_window=None)
    cfg.num_key_value_heads, cfg.head_dim, cfg.num_hidden_layers = 8, 128, 32
    weights_gb, reserve, budget = 14.5, 3.0, 22.5

    def need(ctx):
        dc = dl.DecodeConfig("m", 32, ctx, "bf16")
        kv = footprint_bytes(cfg, dc, ctx + headroom_slots(_Args())) / 1e9
        return weights_gb + kv + reserve

    assert need(1024) > budget
    assert need(512) <= budget
