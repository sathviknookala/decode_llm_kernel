"""The custom-kernel rungs against the oracle, and against the traps the rig was built to set.

The validation gate does the heavy lifting -- oracle agreement, byte-exact V, unaddressed slots,
permuted requests, strided QKV -- so most of this file drives the gate over the registry. What
follows it covers what the gate does not: head-dim independence, int32 positions, degenerate
token counts, graph capture, and the fp32 margin.
"""
import pytest
import torch

try:
    from decode_kernels import cuda as cuda_ext
except ImportError:
    cuda_ext = None

pytestmark = pytest.mark.skipif(
    cuda_ext is None or not cuda_ext.is_available(),
    reason="CUDA extension not built (python setup.py build_ext --inplace)")

from benchmarks import positions as pos
from benchmarks.validation import TOLERANCES, validate_candidate
from benchmarks.workload import HEAD_CONFIGS, Config, build_op_args
from decode_kernels.reference import fused_rope_kv_append_ref

SEED = 1234


def impls():
    from decode_kernels.cuda import ops
    return {"separate": ops.separate_rope_kv_append, "fused": ops.fused_rope_kv_append}


def kernel(name):
    return impls()[name]


NAMES = ("separate", "fused")


def cfg_for(head_cfg, dtype_label, batch=8, alloc=128, mode=pos.RAGGED):
    label, hq, hkv, d = head_cfg
    return Config(label, hq, hkv, d, batch, alloc, dtype_label, mode)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("head_cfg", HEAD_CONFIGS, ids=[h[0] for h in HEAD_CONFIGS])
@pytest.mark.parametrize("dtype_label", ["fp32", "bf16", "fp16"])
def test_the_gate_clears_every_registry_shape(name, head_cfg, dtype_label):
    """All five cases, including strided-qkv and permuted-requests, which is what rejects a
    kernel that assumes D-packed strides or substitutes the token index for the request index."""
    report = validate_candidate(kernel(name), cfg_for(head_cfg, dtype_label), "cuda", SEED)
    assert report["ok"], "; ".join(report["failures"])
    assert report["num_cases"] == 5
    assert report["v_byte_exact"]
    assert report["unaddressed_slots_intact"]


@pytest.mark.parametrize("name", NAMES)
def test_a_kernel_fitted_to_head_dim_128_would_fail_here(name):
    """head_dim 96 makes half 48, so any power-of-two assumption in the pair indexing breaks.
    The registry spans 64/96/128 precisely so this cannot pass unnoticed."""
    head_cfg = next(h for h in HEAD_CONFIGS if h[3] == 96)
    assert head_cfg[3] // 2 == 48, "the non-power-of-two half is the point of this case"
    report = validate_candidate(kernel(name), cfg_for(head_cfg, "bf16"), "cuda", SEED)
    assert report["ok"], "; ".join(report["failures"])


def test_the_two_rungs_agree_with_each_other():
    """Both clear the oracle within tolerance, which leaves room for them to differ from each
    other. They share a rotation and an addressing helper, so they should not."""
    cfg = cfg_for(("mha", 32, 32, 128), "bf16")
    p = pos.build_position_sets(cfg.num_requests, cfg.cache_alloc_len, cfg.position_mode,
                               SEED, 1, "cuda")[0]
    outs, caches = [], []
    for impl in NAMES:
        args = build_op_args(cfg, "cuda", SEED, p)
        outs.append(kernel(impl)(*args))
        caches.append((args.k_cache, args.v_cache))
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(caches[0][0], caches[1][0])
    assert torch.equal(caches[0][1], caches[1][1])


@pytest.mark.parametrize("name", NAMES)
def test_int32_positions_are_accepted(name):
    """The contract accepts int32 or int64; the rig only ever passes int64, so nothing else
    exercises the other instantiation."""
    cfg = cfg_for(("gqa", 32, 8, 128), "bf16")
    p = pos.build_position_sets(cfg.num_requests, cfg.cache_alloc_len, cfg.position_mode,
                               SEED, 1, "cuda")[0]
    wide = build_op_args(cfg, "cuda", SEED, p)
    narrow = build_op_args(cfg, "cuda", SEED, p.to(torch.int32))
    narrow = narrow._replace(request_indices=narrow.request_indices.to(torch.int32))

    q_wide = kernel(name)(*wide)
    q_narrow = kernel(name)(*narrow)
    assert torch.equal(q_wide, q_narrow)
    assert torch.equal(wide.k_cache, narrow.k_cache)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("batch", [1, 2])
def test_small_token_counts(name, batch):
    cfg = cfg_for(("mqa", 71, 1, 64), "bf16", batch=batch)
    report = validate_candidate(kernel(name), cfg, "cuda", SEED)
    assert report["ok"], "; ".join(report["failures"])


@pytest.mark.parametrize("name", NAMES)
def test_zero_tokens_is_a_no_op_not_a_crash(name):
    cfg = cfg_for(("mha", 32, 32, 128), "bf16", batch=0)
    args = build_op_args(cfg, "cuda", SEED,
                        torch.zeros(0, dtype=torch.long, device="cuda"))
    q_rot = kernel(name)(*args)
    torch.cuda.synchronize()
    assert q_rot.shape == (0, 32, 128)


@pytest.mark.parametrize("name", NAMES)
def test_capturable_into_a_cuda_graph(name):
    """Rung 4 showed graph replay removes 62-69% of the compiled path, so the custom rungs are
    only comparable against that ceiling if they capture too. Mutating the cache is fine here:
    GraphRunner captures manually, unlike inductor's cudagraph path."""
    cfg = cfg_for(("mha", 32, 32, 128), "bf16")
    sets = pos.build_position_sets(cfg.num_requests, cfg.cache_alloc_len, cfg.position_mode,
                                  SEED, 2, "cuda")
    args = build_op_args(cfg, "cuda", SEED, sets[0])
    static_positions = args.positions.clone()
    bound = args._replace(positions=static_positions)
    fn = kernel(name)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn(*bound)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn(*bound)

    static_positions.copy_(sets[1])
    graph.replay()
    torch.cuda.synchronize()

    expected_args = build_op_args(cfg, "cuda", SEED, sets[1])
    expected = fused_rope_kv_append_ref(*expected_args)
    torch.testing.assert_close(captured, expected, atol=8e-3, rtol=8e-3)
    # Only the slots this replay addressed: the warmup and the capture wrote sets[0] into the
    # same cache, and those slots are correctly still holding their own tokens.
    ri = bound.request_indices
    torch.testing.assert_close(bound.k_cache[ri, sets[1]], expected_args.k_cache[ri, sets[1]],
                               atol=8e-3, rtol=8e-3)
    assert torch.equal(bound.v_cache[ri, sets[1]], bound.v)


@pytest.mark.parametrize("name", NAMES)
def test_the_fp32_margin_is_reported_not_just_passed(name):
    """fp32 tolerance is atol=rtol=1e-6, roughly ten ulps at O(1). Asserting the margin means a
    change in arithmetic order shows up as a shrinking number rather than a sudden failure."""
    cfg = cfg_for(("mha", 32, 32, 128), "fp32")
    report = validate_candidate(kernel(name), cfg, "cuda", SEED)
    atol = TOLERANCES[torch.float32]["atol"]
    assert report["ok"]
    assert report["max_abs_diff_q"] < atol / 2, (
        f"fp32 q delta {report['max_abs_diff_q']:.3e} has less than 2x margin on {atol:.0e}")
    assert report["max_abs_diff_k_cache"] < atol / 2


@pytest.mark.parametrize("name", NAMES)
def test_a_strided_qkv_view_is_not_copied_first(name):
    """A fused QKV projection is split into views whose token stride is the full fused width.
    Requiring packed inputs would force a copy on the exact path this operation accelerates."""
    from benchmarks.workload import PACKED, STRIDED
    cfg = cfg_for(("mha96", 32, 32, 96), "bf16")
    p = pos.build_position_sets(cfg.num_requests, cfg.cache_alloc_len, cfg.position_mode,
                               SEED, 1, "cuda")[0]
    packed = build_op_args(cfg, "cuda", SEED, p, layout=PACKED)
    strided = build_op_args(cfg, "cuda", SEED, p, layout=STRIDED)
    assert not strided.q.is_contiguous(), "the strided arm must actually be strided"

    kernel(name)(*packed)
    kernel(name)(*strided)
    # Different random draws, so the outputs differ; what must hold is that the strided arm is
    # rotated from its own inputs rather than read with packed strides.
    expected = fused_rope_kv_append_ref(*strided)
    torch.testing.assert_close(kernel(name)(*strided), expected, atol=8e-3, rtol=8e-3)


def test_rope_forward_returns_both_rotations():
    from decode_kernels.cuda import ops
    cfg = cfg_for(("gqa", 32, 8, 128), "fp32")
    p = pos.build_position_sets(cfg.num_requests, cfg.cache_alloc_len, cfg.position_mode,
                               SEED, 1, "cuda")[0]
    args = build_op_args(cfg, "cuda", SEED, p)
    q_rot, k_rot = ops.rope_forward(args.q, args.k, args.positions, args.cos, args.sin)
    assert q_rot.shape == (cfg.num_requests, 32, 128)
    assert k_rot.shape == (cfg.num_requests, 8, 128)
    assert q_rot.is_contiguous() and k_rot.is_contiguous()


def test_a_cache_dtype_mismatch_is_refused_rather_than_silently_cast():
    """A raw scalar copy is what makes V byte-exact; a differing cache dtype would need a cast
    and would stop being byte-exact without saying so."""
    from decode_kernels.cuda import ops
    cfg = cfg_for(("mha", 32, 32, 128), "bf16", batch=2)
    p = pos.build_position_sets(cfg.num_requests, cfg.cache_alloc_len, cfg.position_mode,
                               SEED, 1, "cuda")[0]
    a = build_op_args(cfg, "cuda", SEED, p)
    with pytest.raises(RuntimeError, match="must match its source dtype"):
        ops.fused_rope_kv_append(a.q, a.k, a.v, a.positions, a.cos, a.sin,
                                 a.k_cache.float(), a.v_cache, a.request_indices)


def test_fp16_tables_are_refused():
    """The locked semantics require FP32 trig; accepting a half table would quietly change the
    numerics the whole ladder is compared on."""
    from decode_kernels.cuda import ops
    cfg = cfg_for(("mha", 32, 32, 128), "bf16", batch=2)
    p = pos.build_position_sets(cfg.num_requests, cfg.cache_alloc_len, cfg.position_mode,
                               SEED, 1, "cuda")[0]
    a = build_op_args(cfg, "cuda", SEED, p)
    with pytest.raises(RuntimeError, match="must be FP32"):
        ops.fused_rope_kv_append(a.q, a.k, a.v, a.positions, a.cos.half(), a.sin.half(),
                                 a.k_cache, a.v_cache, a.request_indices)


@pytest.mark.parametrize("name", NAMES)
def test_a_host_tensor_is_refused_rather_than_dereferenced_on_device(name):
    """A CPU cos table was accepted and appeared to work: the host pointer reached the device
    and was dereferenced there, which is undefined behaviour wearing a passing result."""
    from decode_kernels.cuda import ops
    cfg = cfg_for(("mha", 32, 32, 128), "bf16", batch=2)
    p = pos.build_position_sets(cfg.num_requests, cfg.cache_alloc_len, cfg.position_mode,
                               SEED, 1, "cuda")[0]
    a = build_op_args(cfg, "cuda", SEED, p)
    fn = kernel(name)
    for label, args in (
            ("cos", (a.q, a.k, a.v, a.positions, a.cos.cpu(), a.sin, a.k_cache, a.v_cache,
                     a.request_indices)),
            ("positions", (a.q, a.k, a.v, a.positions.cpu(), a.cos, a.sin, a.k_cache, a.v_cache,
                           a.request_indices)),
            ("request_indices", (a.q, a.k, a.v, a.positions, a.cos, a.sin, a.k_cache, a.v_cache,
                                 a.request_indices.cpu()))):
        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            fn(*args)
    assert ops  # the module import is the point of the fixture above


@pytest.mark.parametrize("name", NAMES)
def test_one_buffer_used_as_both_caches_is_refused(name):
    """K and V would race for the same slot and the winner is undefined."""
    cfg = cfg_for(("mha", 32, 32, 128), "bf16", batch=2)
    p = pos.build_position_sets(cfg.num_requests, cfg.cache_alloc_len, cfg.position_mode,
                               SEED, 1, "cuda")[0]
    a = build_op_args(cfg, "cuda", SEED, p)
    with pytest.raises(RuntimeError, match="distinct buffers"):
        kernel(name)(a.q, a.k, a.v, a.positions, a.cos, a.sin, a.k_cache, a.k_cache,
                     a.request_indices)


def test_rope_forward_returns_a_tuple_not_a_list():
    """pybind maps std::vector to a list; the callers all want to unpack a pair."""
    from decode_kernels import cuda as ext
    cfg = cfg_for(("gqa", 32, 8, 128), "fp32", batch=2)
    p = pos.build_position_sets(cfg.num_requests, cfg.cache_alloc_len, cfg.position_mode,
                               SEED, 1, "cuda")[0]
    a = build_op_args(cfg, "cuda", SEED, p)
    assert isinstance(ext.require().rope_forward(a.q, a.k, a.positions, a.cos, a.sin), tuple)


@pytest.mark.parametrize("name", NAMES)
def test_zero_token_caches_are_not_mistaken_for_aliases(name):
    """Two distinct empty CUDA tensors both report data_ptr() == 0, so an unguarded pointer
    compare rejects a legitimate zero-token call."""
    cfg = cfg_for(("mha", 32, 32, 128), "bf16", batch=0)
    args = build_op_args(cfg, "cuda", SEED, torch.zeros(0, dtype=torch.long, device="cuda"))
    assert args.k_cache.data_ptr() == args.v_cache.data_ptr() == 0
    kernel(name)(*args)
    torch.cuda.synchronize()
