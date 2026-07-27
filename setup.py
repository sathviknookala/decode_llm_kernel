import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CSRC_DIR = os.path.join(REPO_ROOT, "csrc")

EXTENSION_NAME = "decode_kernels.cuda._ext"
CUDA_ARCH = "12.0"                    # sm_120; pinned rather than autodetected so builds match
BUILD_COMMAND = "python setup.py build_ext --inplace"

CXX_FLAGS = ["-O3"]
# -lineinfo gives Nsight Compute source attribution at no runtime cost. --use_fast_math is
# deliberately absent: the locked semantics require exact FP32 rotation arithmetic.
NVCC_FLAGS = ["-O3", "-lineinfo"]

# Escape hatch for the nvcc 12.9 / torch cu128 skew, e.g. "-allow-unsupported-compiler".
NVCC_EXTRA_FLAGS = os.environ.get("DECODE_NVCC_EXTRA_FLAGS", "").split()


def sources():
    found = sorted(glob.glob(os.path.join(CSRC_DIR, "*.cpp"))
                   + glob.glob(os.path.join(CSRC_DIR, "*.cu")))
    if not found:
        raise SystemExit(
            f"no extension sources found in {os.path.relpath(CSRC_DIR, REPO_ROOT)}/ -- add "
            f"bindings.cpp and at least one .cu file before running '{BUILD_COMMAND}'")
    return found


def make_extension():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", CUDA_ARCH)   # an explicit shell value wins
    return CUDAExtension(
        name=EXTENSION_NAME,
        sources=sources(),
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS + NVCC_EXTRA_FLAGS},
    )


# Builds the extension in place; it does not package the project.
if __name__ == "__main__":
    setup(
        name="decode_kernels",
        ext_modules=[make_extension()],
        cmdclass={"build_ext": BuildExtension},
    )
