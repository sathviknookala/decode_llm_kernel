import importlib
import os

import pytest
import torch

import setup as build_config           # importing is side-effect free; setup() runs under __main__

try:
    from decode_kernels import cuda as cuda_ext
except ImportError:                    # the wrapper package does not exist yet
    cuda_ext = None

EXT_BUILT = bool(cuda_ext is not None and cuda_ext.is_available())
requires_ext = pytest.mark.skipif(not EXT_BUILT, reason="CUDA extension not built")
requires_wrapper = pytest.mark.skipif(cuda_ext is None, reason="decode_kernels.cuda not present")


def test_extension_name_matches_the_wrapper_import_path():
    package, _, module = build_config.EXTENSION_NAME.rpartition(".")
    assert package == "decode_kernels.cuda", "the .so must land beside the Python wrapper"
    assert module.startswith("_"), "the compiled module is private; the wrapper is the API"


def test_build_command_is_in_place():
    assert "build_ext" in build_config.BUILD_COMMAND
    assert "--inplace" in build_config.BUILD_COMMAND


def test_cuda_arch_is_pinned_not_autodetected():
    assert build_config.CUDA_ARCH == "12.0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pinned_arch_matches_this_device():
    major, minor = torch.cuda.get_device_capability(0)
    assert build_config.CUDA_ARCH == f"{major}.{minor}", (
        "pinned arch has drifted from the GPU it is built for")


def test_nvcc_flags_preserve_exact_float_math():
    assert "--use_fast_math" not in build_config.NVCC_FLAGS, (
        "fast math contradicts the locked FP32 rotation policy")
    assert "-lineinfo" in build_config.NVCC_FLAGS, "needed for Nsight Compute attribution"


def test_extra_nvcc_flags_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("DECODE_NVCC_EXTRA_FLAGS", "-allow-unsupported-compiler -Xptxas=-v")
    reloaded = importlib.reload(build_config)
    try:
        assert reloaded.NVCC_EXTRA_FLAGS == ["-allow-unsupported-compiler", "-Xptxas=-v"]
    finally:
        monkeypatch.delenv("DECODE_NVCC_EXTRA_FLAGS")
        importlib.reload(build_config)


def test_source_discovery_reports_a_usable_error_or_real_sources():
    if os.path.isdir(build_config.CSRC_DIR) and os.listdir(build_config.CSRC_DIR):
        found = build_config.sources()
        assert found == sorted(found)                       # deterministic link order
        assert all(f.startswith(build_config.CSRC_DIR) for f in found)
        assert all(f.endswith((".cpp", ".cu")) for f in found)
    else:
        with pytest.raises(SystemExit, match="no extension sources"):
            build_config.sources()


def test_source_discovery_error_names_the_build_command():
    if os.path.isdir(build_config.CSRC_DIR) and os.listdir(build_config.CSRC_DIR):
        pytest.skip("sources exist; the empty-csrc path is covered elsewhere")
    with pytest.raises(SystemExit) as e:
        build_config.sources()
    assert build_config.BUILD_COMMAND in str(e.value)


@requires_wrapper
@pytest.mark.skipif(EXT_BUILT, reason="extension is built; nothing to refuse")
def test_require_explains_how_to_build_when_missing():
    with pytest.raises(RuntimeError, match="build_ext"):
        cuda_ext.require()


@requires_ext
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_smoke_kernel_launches_and_writes(dtype):
    x = torch.zeros(1024, dtype=dtype, device="cuda")
    cuda_ext.smoke_fill(x, 3.5)
    torch.cuda.synchronize()
    assert torch.equal(x, torch.full_like(x, 3.5))


@requires_ext
def test_build_info_matches_the_running_device():
    info = cuda_ext.build_info()
    major, minor = torch.cuda.get_device_capability(0)
    assert f"{major}{minor}" in str(info["gencode_arch"])


@requires_ext
def test_build_info_records_the_compile_runtime_cuda_skew():
    """Records rather than asserts: nvcc 12.9 against a cu128 torch is expected here."""
    info = cuda_ext.build_info()
    assert info["cuda_version_compiled"]
    assert info["cuda_version_runtime"]
