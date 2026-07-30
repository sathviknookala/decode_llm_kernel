import math

import pytest
import torch

try:
    from decode_kernels import cuda as cuda_ext
except ImportError:
    cuda_ext = None

pytestmark = pytest.mark.skipif(
    cuda_ext is None or not cuda_ext.is_available(),
    reason="CUDA extension not built (python setup.py build_ext --inplace)")

DTYPES = [torch.float64, torch.float32, torch.float16, torch.bfloat16]

# 65535 blocks x 256 threads covers 16.7M elements in one pass; more forces the grid-stride wrap.
GRID_STRIDE_NUMEL = 20_000_000


@pytest.mark.parametrize("dtype", DTYPES)
def test_fills_every_element(dtype):
    x = torch.zeros(1024, dtype=dtype, device="cuda")
    cuda_ext.smoke_fill(x, 3.5)
    torch.cuda.synchronize()
    assert torch.equal(x, torch.full_like(x, 3.5))


@pytest.mark.parametrize("value", [0.0, 1.0, -2.25, 3.5, 256.0])
def test_writes_the_requested_value(value):
    x = torch.ones(97, dtype=torch.float32, device="cuda")     # exactly representable values only
    cuda_ext.smoke_fill(x, value)
    torch.cuda.synchronize()
    assert torch.equal(x, torch.full_like(x, value))


def test_double_tensor_keeps_double_precision():
    """Regression: the fill value used to be routed through float32 before the final cast,
    silently truncating precision for a float64 tensor."""
    x = torch.zeros(64, dtype=torch.float64, device="cuda")
    cuda_ext.smoke_fill(x, math.pi)
    torch.cuda.synchronize()
    assert torch.equal(x, torch.full_like(x, math.pi))


@pytest.mark.parametrize("shape", [(1,), (7,), (255,), (256,), (257,), (4, 8, 16), (1, 1, 1)])
def test_shapes_that_do_not_divide_the_block_size(shape):
    x = torch.zeros(shape, dtype=torch.float32, device="cuda")
    cuda_ext.smoke_fill(x, 1.0)
    torch.cuda.synchronize()
    assert int((x != 1.0).sum()) == 0


def test_grid_stride_loop_covers_more_than_one_pass():
    x = torch.zeros(GRID_STRIDE_NUMEL, dtype=torch.bfloat16, device="cuda")
    cuda_ext.smoke_fill(x, 2.0)
    torch.cuda.synchronize()
    assert int((x != 2.0).sum()) == 0


def test_empty_tensor_is_a_no_op():
    x = torch.empty(0, dtype=torch.float32, device="cuda")
    cuda_ext.smoke_fill(x, 1.0)                                 # must not launch or raise
    torch.cuda.synchronize()
    assert x.numel() == 0


def test_mutates_in_place_and_returns_none():
    x = torch.zeros(16, device="cuda")
    assert cuda_ext.smoke_fill(x, 5.0) is None
    torch.cuda.synchronize()
    assert int(x[0]) == 5


def test_does_not_write_past_the_tensor():
    backing = torch.zeros(1024, dtype=torch.float32, device="cuda")
    head = backing[:100]                                        # a contiguous view
    cuda_ext.smoke_fill(head, 9.0)
    torch.cuda.synchronize()
    assert torch.equal(backing[:100], torch.full((100,), 9.0, device="cuda"))
    assert int(backing[100:].abs().sum()) == 0


def test_rejects_cpu_tensors():
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        cuda_ext.smoke_fill(torch.zeros(4), 1.0)


def test_rejects_non_contiguous_tensors():
    strided = torch.zeros(8, 8, device="cuda")[:, ::2]
    assert not strided.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        cuda_ext.smoke_fill(strided, 1.0)


def test_rejects_dtypes_outside_the_dispatch_list():
    with pytest.raises(RuntimeError, match="not implemented"):
        cuda_ext.smoke_fill(torch.zeros(4, dtype=torch.int32, device="cuda"), 1.0)


def test_runs_correctly_on_a_non_default_stream():
    x = torch.zeros(4096, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cuda_ext.smoke_fill(x, 4.0)
    torch.cuda.synchronize()
    assert int((x != 4.0).sum()) == 0


def test_is_cuda_graph_capturable():
    """Capture fails outright if the kernel goes to the default stream, so this pins stream
    affinity deterministically -- and graph capture is how the kernel will be benchmarked."""
    x = torch.zeros(1024, device="cuda")
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        cuda_ext.smoke_fill(x, 1.0)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cuda_ext.smoke_fill(x, 7.0)

    x.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(x, torch.full_like(x, 7.0))
