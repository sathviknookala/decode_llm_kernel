import os

import pytest
import torch

from benchmarks.benchmark_utils import _nvcc_version

try:
    from decode_kernels import cuda as cuda_ext
except ImportError:
    cuda_ext = None

pytestmark = pytest.mark.skipif(
    cuda_ext is None or not cuda_ext.is_available(),
    reason="CUDA extension not built (python setup.py build_ext --inplace)")

REQUIRED_KEYS = ("gencode_arch", "cuda_version_compiled", "cuda_version_runtime",
                 "cuda_version_driver", "nvcc_version", "fast_math")


def _version_tuple(text):
    return tuple(int(part) for part in text.split("."))


@pytest.fixture(scope="module")
def info():
    return cuda_ext.build_info()


def test_reports_every_required_key(info):
    assert set(REQUIRED_KEYS) <= set(info)
    assert all(info[k] for k in REQUIRED_KEYS if k != "fast_math")


def test_arch_is_the_one_this_device_runs(info):
    major, minor = torch.cuda.get_device_capability(0)
    assert int(info["gencode_arch"]) == major * 100 + minor * 10, (
        "the binary's device code does not target this GPU; rebuild it")


def test_binary_was_built_without_fast_math(info):
    """setup.py omits --use_fast_math; this asserts it in the compiled artifact."""
    assert info["fast_math"] == "0", (
        "fast math is enabled in the binary, which contradicts the locked FP32 rotation policy")


def test_compiled_cuda_version_comes_from_the_nvcc_that_built_it(info):
    nvcc_major_minor = ".".join(info["nvcc_version"].split(".")[:2])
    assert info["cuda_version_compiled"] == nvcc_major_minor


def test_runtime_cuda_is_the_one_torch_loads_not_the_build_toolkit(info):
    """The known skew, pinned: built against a newer toolkit, run against torch's runtime."""
    assert info["cuda_version_runtime"] == torch.version.cuda


def test_driver_is_new_enough_for_the_runtime(info):
    assert _version_tuple(info["cuda_version_driver"]) >= _version_tuple(info["cuda_version_runtime"])


def test_binary_matches_the_nvcc_the_environment_capture_records(info):
    system_nvcc = _nvcc_version()
    if system_nvcc is None:
        pytest.skip("nvcc not on PATH")
    assert info["nvcc_version"] == system_nvcc, (
        "the extension was built by a different nvcc than env_metadata() records, so results "
        "would be attributed to the wrong toolchain; rebuild the extension")


def test_identifies_which_binary_it_read(info):
    path = info["extension_path"]
    assert path and os.path.isfile(path)
    assert os.path.basename(path).startswith("_ext")
    assert info["extension_mtime"] == pytest.approx(os.path.getmtime(path))


def test_is_callable_repeatedly(info):
    assert cuda_ext.build_info()["gencode_arch"] == info["gencode_arch"]
