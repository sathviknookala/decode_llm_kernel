import itertools
from dataclasses import dataclass

import torch
import torch._dynamo

from decode_kernels import cuda
from decode_kernels.reference import fused_rope_kv_append_ref
from benchmarks.workload import make_thunk

EAGER = "eager"
COMPILE = "compile"
CUDA_SEPARATE = "cuda_separate"
CUDA_FUSED = "cuda_fused"
# The custom-kernel rungs. They take no compile mode or backend, for the same reason eager does
# not: reporting the run's inductor settings on a hand-written kernel would invent provenance.
CUDA_BASES = (CUDA_SEPARATE, CUDA_FUSED)
BASES = (EAGER, COMPILE) + CUDA_BASES
GRAPH_WARMUP = 3

# Modes that turn on inductor's own CUDA graphs. Wrapping one in GraphRunner would capture a
# graph whose body already replays a graph, so a spec cannot ask for both.
CUDAGRAPH_MODES = ("reduce-overhead", "max-autotune")


def eager_impl():
    return fused_rope_kv_append_ref


def compile_impl(mode=None, backend="inductor"):
    """Fresh compile per config: dynamo caches on the code object and would otherwise
    hit recompile_limit mid-sweep and silently fall back to eager."""
    torch._dynamo.reset()
    kwargs = {"backend": backend, "dynamic": False}
    if mode:
        kwargs["mode"] = mode
    return torch.compile(fused_rope_kv_append_ref, **kwargs)


def inductor_cudagraph_skips():
    """Times inductor declined to apply its own CUDA graphs. This operation mutates its cache
    arguments, which is exactly what that path refuses, so a `reduce-overhead` row can be an
    ordinary compiled row wearing a cudagraph label."""
    from torch._dynamo.utils import counters
    return counters["inductor"].get("cudagraph_skips", 0)


def cuda_impl(base):
    from decode_kernels.cuda import ops
    if base == CUDA_SEPARATE:
        return ops.separate_rope_kv_append
    return ops.fused_rope_kv_append


def base_callable(base, mode=None, backend="inductor"):
    if base == EAGER:
        return eager_impl()
    if base == COMPILE:
        return compile_impl(mode, backend)
    if base in CUDA_BASES:
        return cuda_impl(base)
    raise ValueError(f"unknown base {base!r}, expected one of {BASES}")


def assert_extension_available(specs):
    """Fail before the first configuration rather than per impl per config.

    A missing _ext raises inside spec.build(), which run_config records as an ERROR row -- so an
    unbuilt extension would otherwise produce a whole sweep of ERROR rows instead of one line
    saying how to build it.
    """
    needed = sorted({s.label for s in specs if s.base in CUDA_BASES})
    if needed and not cuda.is_available():
        raise SystemExit(f"{needed} need the CUDA extension: {cuda.unavailable_reason()}. "
                         f"Build it with '{cuda.BUILD_COMMAND}'.")


class DirectRunner:
    """Calls the implementation through the normal dispatch path."""

    captured = False

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *op_args):
        return self.fn(*op_args)

    def make_thunk(self, args, position_sets):
        return make_thunk(self.fn, args, position_sets)


class GraphRunner:
    def __init__(self, fn, warmup=GRAPH_WARMUP):
        self.fn = fn
        self.warmup = warmup
        self.capture_count = 0
        self._key = None
        self._graph = None
        self._output = None
        self._static_positions = None
        self._bound_args = None

    @property
    def captured(self):
        return self._graph is not None

    @staticmethod
    def _pointer_key(op_args):
        q, k, v, _, cos, sin, k_cache, v_cache, request_indices = op_args
        return tuple(t.data_ptr() for t in (
            q, k, v, cos, sin, k_cache, v_cache, request_indices))

    def _capture(self, op_args):
        self.release()
        q, k, v, positions, cos, sin, k_cache, v_cache, request_indices = op_args
        static_positions = torch.empty_like(positions)
        static_positions.copy_(positions)
        bound_args = (q, k, v, static_positions, cos, sin,
                      k_cache, v_cache, request_indices)

        current = torch.cuda.current_stream(q.device)
        side = torch.cuda.Stream(device=q.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                self.fn(*bound_args)
        current.wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.fn(*bound_args)

        self._key = self._pointer_key(op_args)
        self._graph = graph
        self._output = output
        self._static_positions = static_positions
        self._bound_args = bound_args
        self.capture_count += 1

    def __call__(self, *op_args):
        key = self._pointer_key(op_args)
        if key != self._key:
            self._capture(op_args)
        self._static_positions.copy_(op_args[3])
        self._graph.replay()
        return self._output

    def make_thunk(self, args, position_sets):
        cyc = itertools.cycle(position_sets)
        q, k, v = args.q, args.k, args.v
        cos, sin = args.cos, args.sin
        kc, vc, ri = args.k_cache, args.v_cache, args.request_indices

        def run():
            self(q, k, v, next(cyc), cos, sin, kc, vc, ri)
        return run

    def release(self):
        self._output = None
        self._graph = None
        self._static_positions = None
        self._bound_args = None
        self._key = None


@dataclass(frozen=True)
class ImplSpec:
    label: str
    base: str
    graph: bool
    description: str
    mode: str | None = None
    backend: str | None = None

    def resolve(self, mode=None, backend="inductor"):
        """A spec's own mode/backend pins it; otherwise it takes the run's. Eager and the custom
        kernels have neither, and reporting the run's compile settings on those rows would
        invent provenance."""
        if self.base == EAGER or self.base in CUDA_BASES:
            return None, None
        resolved_mode = self.mode if self.mode is not None else mode
        resolved_backend = self.backend if self.backend is not None else backend
        if self.graph and resolved_mode in CUDAGRAPH_MODES:
            raise ValueError(
                f"{self.label}: mode {resolved_mode!r} enables inductor's own CUDA graphs, which "
                f"cannot be captured inside GraphRunner; use max-autotune-no-cudagraphs")
        return resolved_mode, resolved_backend

    def build(self, mode=None, backend="inductor"):
        resolved_mode, resolved_backend = self.resolve(mode, backend)
        fn = base_callable(self.base, resolved_mode, resolved_backend or "inductor")
        return GraphRunner(fn) if self.graph else DirectRunner(fn)


IMPL_SPECS = (
    ImplSpec("eager", EAGER, False, "reference through eager PyTorch dispatch"),
    ImplSpec("compile", COMPILE, False, "reference through torch.compile"),
    ImplSpec("graph_eager", EAGER, True, "eager reference replayed through a CUDA graph"),
    ImplSpec("graph_compile", COMPILE, True, "compiled reference replayed through a CUDA graph"),
    ImplSpec("compile_max_autotune", COMPILE, False,
             "torch.compile with coordinate-descent autotuning, no inductor cudagraphs",
             mode="max-autotune-no-cudagraphs"),
    ImplSpec("compile_reduce_overhead", COMPILE, False,
             "torch.compile with inductor's own cudagraph wrapping, for comparison against "
             "GraphRunner's manual capture",
             mode="reduce-overhead"),
    ImplSpec("graph_compile_max_autotune", COMPILE, True,
             "autotuned compile replayed through a CUDA graph",
             mode="max-autotune-no-cudagraphs"),
    # Ladder rungs 5 and 6. The pair is a fusion ablation: identical rotation and addressing,
    # differing only in launch count and whether k_rot is materialized.
    ImplSpec("cuda_separate", CUDA_SEPARATE, False,
             "custom RoPE kernel then custom append kernel, k_rot materialized"),
    ImplSpec("cuda_fused", CUDA_FUSED, False,
             "one custom kernel; k_rot goes straight to its cache slot"),
    ImplSpec("graph_cuda_separate", CUDA_SEPARATE, True,
             "the separate custom kernels replayed through a CUDA graph"),
    ImplSpec("graph_cuda_fused", CUDA_FUSED, True,
             "the fused custom kernel replayed through a CUDA graph"),
)

IMPL_LABELS = tuple(s.label for s in IMPL_SPECS)
# The four rungs the committed baselines are built from. Everything else is opt-in via --impls:
# the mode variants each recompile per config, and the custom-kernel rungs need a built
# extension, so neither belongs in a default that committed CSVs are a subset of.
DEFAULT_IMPLS = ("eager", "compile", "graph_eager", "graph_compile")
CUDA_IMPLS = tuple(s.label for s in IMPL_SPECS if s.base in CUDA_BASES)


def resolve_impls(labels):
    by_label = {s.label: s for s in IMPL_SPECS}
    unknown = [l for l in labels if l not in by_label]
    if unknown:
        raise ValueError(f"unknown impl(s) {unknown}, expected any of {list(IMPL_LABELS)}")
    return [by_label[l] for l in labels]
