"""Patch targets must come from the live model, not from an import.

Patching `models.mistral.modeling_mistral` while running Phi-3 installs cleanly, is never
consulted, and reports a clean 0% operator share. That is the inert-patch failure
test_amdahl_probe.py's meta-test guards against -- it just never ran against a second
architecture, which is how the hole survived.
"""
import pytest
import torch
from transformers import Phi3Config, Phi3ForCausalLM
from transformers.models.mistral import modeling_mistral
from transformers.models.phi3 import modeling_phi3

from benchmarks import decode_loop as dl
from benchmarks.rung_patches import (
    MISTRAL_ARCH,
    Arch,
    OP_DOUBLED,
    OP_REMOVED,
    OP_REPLACED_REF,
    ROPE_REMOVED,
    apply_rung,
    resolve_arch,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
CTX, BATCH, HEADROOM = 8, 2, 16


@pytest.fixture(scope="module")
def tiny_phi3():
    torch.manual_seed(0)
    cfg = Phi3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=128, sliding_window=None,
                     pad_token_id=0, attn_implementation="sdpa")
    cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
    return Phi3ForCausalLM(cfg).to("cuda", torch.float32).eval()


def _prefilled(model):
    """Prefill runs unpatched, exactly as the probe does it. The seam adapters assume a decode
    step (S=1); running prefill inside a rung would feed them an S=ctx tensor."""
    cfg = dl.DecodeConfig("tiny-phi3", BATCH, CTX, "fp32")
    cache = dl.build_cache(model, cfg, CTX + HEADROOM)
    dl.prefill(model, cache, cfg)
    return cfg, cache


def _step(model, cache, cfg):
    ids = torch.full((BATCH, 1), 3, device="cuda")
    pos = torch.tensor([cfg.ctx], device="cuda")
    with torch.inference_mode():
        return model(input_ids=ids, past_key_values=cache, cache_position=pos,
                     use_cache=True).logits.clone()


def _decode_step(model):
    cfg, cache = _prefilled(model)
    return _step(model, cache, cfg)


@cuda_only
def test_arch_resolves_to_the_module_the_model_actually_calls(tiny_phi3):
    assert resolve_arch(tiny_phi3).module is modeling_phi3
    assert MISTRAL_ARCH.module is modeling_mistral


@cuda_only
def test_the_old_hardcoded_target_was_inert_on_a_second_architecture(tiny_phi3):
    """Both halves matter: the wrong target must change nothing (so the bug was real and
    silent), and the resolved target must change the output (so the fix bites)."""
    arch = resolve_arch(tiny_phi3)
    reference = _decode_step(tiny_phi3)

    with apply_rung(ROPE_REMOVED, MISTRAL_ARCH):
        wrong_target = _decode_step(tiny_phi3)
    with apply_rung(ROPE_REMOVED, arch):
        resolved = _decode_step(tiny_phi3)

    assert torch.allclose(reference, wrong_target), "premise: mistral's patch never bit phi3"
    assert not torch.allclose(reference, resolved), "the resolved patch must actually apply"


@cuda_only
@pytest.mark.parametrize("label", [OP_REMOVED, ROPE_REMOVED, OP_DOUBLED])
def test_the_architecture_independent_rungs_bite_on_phi3(tiny_phi3, label):
    arch = resolve_arch(tiny_phi3)
    reference = _decode_step(tiny_phi3)
    with apply_rung(label, arch):
        out = _decode_step(tiny_phi3)
    assert out.shape == reference.shape
    assert not torch.allclose(reference, out, atol=1e-4), f"{label} was inert on phi3"


@cuda_only
def test_phi3_has_a_seam_adapter_and_it_is_numerically_equivalent(tiny_phi3):
    """Phi-3 slices one fused qkv_proj three ways -- the layout operation_semantics.md calls a
    hard requirement, and the only anchor in the rig that actually exercises it. If the seam
    were not equivalent the ablation would be comparing two different models."""
    arch = resolve_arch(tiny_phi3)
    assert arch.supports_ref_replacement
    cfg, cache = _prefilled(tiny_phi3)
    reference = _step(tiny_phi3, cache, cfg)

    cfg2, cache2 = _prefilled(tiny_phi3)
    with apply_rung(OP_REPLACED_REF, arch):
        replaced = _step(tiny_phi3, cache2, cfg2)
    torch.testing.assert_close(replaced, reference, atol=2e-4, rtol=2e-4)


@cuda_only
def test_the_fused_and_separate_seams_are_not_the_same_function(tiny_phi3):
    """A registry that silently handed Phi-3 the separate-projection adapter would raise on
    q_proj rather than mislead, but the split is the whole point of the entry."""
    assert resolve_arch(tiny_phi3).ref_forward is not MISTRAL_ARCH.ref_forward


@cuda_only
def test_an_architecture_with_no_entry_still_refuses(tiny_phi3):
    unadapted = Arch("not_a_real_arch", modeling_phi3, modeling_phi3.Phi3Attention, None)
    before = modeling_phi3.Phi3Attention.forward
    with pytest.raises(NotImplementedError, match="no seam adapter"):
        with apply_rung(OP_REPLACED_REF, unadapted):
            pass
    assert modeling_phi3.Phi3Attention.forward is before
