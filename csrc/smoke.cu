#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <string>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 65535;

template <typename scalar_t>
__global__ void smoke_fill_kernel(scalar_t* __restrict__ out, int64_t numel, scalar_t value) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < numel; i += stride) {
    out[i] = value;
  }
}

// Reports the arch of the code that actually runs here, rather than a compile-time flag list.
__global__ void arch_probe_kernel(int* out) {
#ifdef __CUDA_ARCH__
  *out = __CUDA_ARCH__;
#else
  *out = 0;
#endif
}

std::string version_string(int packed) {            // 12080 -> "12.8"
  if (packed <= 0) return "";
  return std::to_string(packed / 1000) + "." + std::to_string((packed % 1000) / 10);
}

}  // namespace

void smoke_fill(at::Tensor tensor, double value) {
  TORCH_CHECK(tensor.is_cuda(), "smoke_fill expects a CUDA tensor, got ", tensor.device());
  TORCH_CHECK(tensor.is_contiguous(), "smoke_fill expects a contiguous tensor");

  const at::cuda::CUDAGuard guard(tensor.device());
  const int64_t numel = tensor.numel();
  if (numel == 0) return;

  const int blocks = static_cast<int>(
      std::min<int64_t>((numel + kThreads - 1) / kThreads, kMaxBlocks));
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  tensor.scalar_type(), "smoke_fill", [&] {
    smoke_fill_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        tensor.data_ptr<scalar_t>(), numel, static_cast<scalar_t>(value));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::map<std::string, std::string> smoke_build_info() {
  auto probe = at::empty({1}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  arch_probe_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(probe.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int runtime_version = 0;
  int driver_version = 0;
  C10_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  C10_CUDA_CHECK(cudaDriverGetVersion(&driver_version));

  std::map<std::string, std::string> info;
  info["gencode_arch"] = std::to_string(probe.cpu().item<int>());
  info["cuda_version_compiled"] = version_string(CUDART_VERSION);
  info["cuda_version_runtime"] = version_string(runtime_version);
  info["cuda_version_driver"] = version_string(driver_version);
  info["nvcc_version"] = std::to_string(__CUDACC_VER_MAJOR__) + "." +
                         std::to_string(__CUDACC_VER_MINOR__) + "." +
                         std::to_string(__CUDACC_VER_BUILD__);
  // Proves the no-fast-math policy in the binary, not just in setup.py.
#ifdef __FAST_MATH__
  info["fast_math"] = "1";
#else
  info["fast_math"] = "0";
#endif
  return info;
}
