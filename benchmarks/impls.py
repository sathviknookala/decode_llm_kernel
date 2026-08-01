from dataclasses import dataclass

import torch
import torch._dynamo

from decode_kernels.reference import fused_rope_kv_append_ref
from benchmarks.workload import make_thunk

EAGER = "eager"
COMPILE = "compile"
BASES = (EAGER, COMPILE)


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


def base_callable(base, mode=None, backend="inductor"):
    if base == EAGER:
        return eager_impl()
    if base == COMPILE:
        return compile_impl(mode, backend)
    raise ValueError(f"unknown base {base!r}, expected one of {BASES}")


class DirectRunner:
    """Calls the implementation through the normal dispatch path."""

    captured = False

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *op_args):
        return self.fn(*op_args)

    def make_thunk(self, args, position_sets):
        return make_thunk(self.fn, args, position_sets)


@dataclass(frozen=True)
class ImplSpec:
    label: str
    base: str
    graph: bool
    description: str

    def build(self, mode=None, backend="inductor"):
        return DirectRunner(base_callable(self.base, mode, backend))


IMPL_SPECS = (
    ImplSpec("eager", EAGER, False, "reference through eager PyTorch dispatch"),
    ImplSpec("compile", COMPILE, False, "reference through torch.compile"),
)

IMPL_LABELS = tuple(s.label for s in IMPL_SPECS)
DEFAULT_IMPLS = IMPL_LABELS


def resolve_impls(labels):
    by_label = {s.label: s for s in IMPL_SPECS}
    unknown = [l for l in labels if l not in by_label]
    if unknown:
        raise ValueError(f"unknown impl(s) {unknown}, expected any of {list(IMPL_LABELS)}")
    return [by_label[l] for l in labels]
