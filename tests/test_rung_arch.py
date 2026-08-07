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


def _decode_step(model):
    """The rungs patch the decode path, so they have to be exercised on it: op_removed returns
    the cache's own tensors, which only exist once prefill has initialized them."""
    cfg = dl.DecodeConfig("tiny-phi3", BATCH, CTX, "fp32")
    cache = dl.build_cache(model, cfg, CTX + HEADROOM)
    dl.prefill(model, cache, cfg)
    ids = torch.full((BATCH, 1), 3, device="cuda")
    pos = torch.tensor([CTX], device="cuda")
    with torch.inference_mode():
        return model(input_ids=ids, past_key_values=cache, cache_position=pos,
                     use_cache=True).logits.clone()


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
def test_op_replaced_ref_refuses_an_architecture_it_has_no_seam_for(tiny_phi3):
    """Phi-3 fuses QKV into one projection, so the mistral seam does not transfer. Refusing is
    the only safe answer -- running it would time an unmodified step and call it the reference."""
    arch = resolve_arch(tiny_phi3)
    assert not arch.supports_ref_replacement
    with pytest.raises(NotImplementedError, match="no seam adapter"):
        with apply_rung(OP_REPLACED_REF, arch):
            pass


@cuda_only
def test_refusing_leaves_no_patch_installed(tiny_phi3):
    before = modeling_phi3.Phi3Attention.forward
    with pytest.raises(NotImplementedError):
        with apply_rung(OP_REPLACED_REF, resolve_arch(tiny_phi3)):
            pass
    assert modeling_phi3.Phi3Attention.forward is before
