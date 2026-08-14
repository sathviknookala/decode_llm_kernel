"""The custom-kernel rungs of the baseline ladder, behind the reference's own signature.

Both callables here are drop-in replacements for `fused_rope_kv_append_ref`, so the validation
gate and the sweep drive them through exactly the path they drive the reference through.

Nothing in this module synchronizes, reads device memory on the host, or allocates outside the
caching allocator: these are timed at microsecond scale and captured into CUDA graphs. In-bounds
addressing is the caller's guarantee per the locked contract, enforced device-side by the kernels
at the cost of a compare -- `assert_valid_addressing` synchronizes and belongs to the gate, not
to a timed path.
"""
from . import require


def rope_forward(q, k, positions, cos, sin):
    """Split-half RoPE over strided Q and K in one launch. Returns (q_rot, k_rot)."""
    return tuple(require().rope_forward(q, k, positions, cos, sin))


def kv_append(k_rot, v, positions, request_indices, k_cache, v_cache):
    """Scatter rotated K and raw V into their cache slots, in place."""
    require().kv_append(k_rot, v, positions, request_indices, k_cache, v_cache)


def separate_rope_kv_append(q, k, v, positions, cos, sin, k_cache, v_cache, request_indices):
    """Ladder rung 5: RoPE kernel then append kernel.

    The `k_rot` intermediate is materialized on purpose. Two launches and this allocation are
    the entire difference from the fused rung, which is what isolates the fusion benefit from
    the benefit of merely replacing framework code.
    """
    q_rot, k_rot = rope_forward(q, k, positions, cos, sin)
    kv_append(k_rot, v, positions, request_indices, k_cache, v_cache)
    return q_rot


def fused_rope_kv_append(q, k, v, positions, cos, sin, k_cache, v_cache, request_indices):
    """Ladder rung 6: one launch, and `k_rot` never exists as a tensor."""
    return require().fused_rope_kv_append(q, k, v, positions, cos, sin, k_cache, v_cache,
                                          request_indices)


__all__ = ["fused_rope_kv_append", "kv_append", "rope_forward", "separate_rope_kv_append"]
