#include "kernel_common.cuh"

namespace decode_kernels {
namespace {

// Grid sized to the tokens being written, never to the cache: every slot this kernel can reach
// is one it was asked to write, which is what keeps unaddressed slots byte-for-byte unchanged.
//
// V moves as a raw scalar_t copy. The gate asserts torch.equal on the written V slots, so a
// round-trip through float would be both wasted work and a correctness risk.
template <typename scalar_t, typename index_t>
__global__ void kv_append_kernel(
    const scalar_t* __restrict__ k_rot, const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache, scalar_t* __restrict__ v_cache,
    const index_t* __restrict__ positions, const index_t* __restrict__ request_indices,
    int64_t num_tokens, int num_kv_heads, int head_dim,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t v_token_stride, int64_t v_head_stride,
    int64_t batch_size, int64_t cache_alloc_len) {
  const int64_t per_token = static_cast<int64_t>(num_kv_heads) * head_dim;
  const int64_t work = num_tokens * per_token;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < work; idx += stride) {
    const int64_t token = idx / per_token;
    const int64_t within = idx % per_token;
    const int head = static_cast<int>(within / head_dim);
    const int d = static_cast<int>(within % head_dim);

    // request_indices is a real indirection: a permuted mapping makes token index and request
    // index differ, so substituting the token here is wrong and the gate catches it.
    const int64_t slot = cache_offset(static_cast<int64_t>(request_indices[token]),
                                      static_cast<int64_t>(positions[token]),
                                      batch_size, cache_alloc_len, per_token);
    const int64_t out = slot + static_cast<int64_t>(head) * head_dim + d;
    k_cache[out] = k_rot[token * k_token_stride + head * k_head_stride + d];
    v_cache[out] = v[token * v_token_stride + head * v_head_stride + d];
  }
}

}  // namespace
}  // namespace decode_kernels

void kv_append(at::Tensor k_rot, at::Tensor v, at::Tensor positions,
               at::Tensor request_indices, at::Tensor k_cache, at::Tensor v_cache) {
  using namespace decode_kernels;
  TORCH_CHECK(k_rot.dim() == 3, "k_rot must be [num_tokens, num_kv_heads, head_dim]");
  const int64_t num_tokens = k_rot.size(0);
  const int64_t num_kv_heads = k_rot.size(1);
  const int64_t head_dim = k_rot.size(2);
  const at::Device device = k_rot.device();
  check_qkv(k_rot, "k_rot", num_tokens, head_dim, device);
  check_qkv(v, "v", num_tokens, head_dim, device);
  TORCH_CHECK(v.size(1) == num_kv_heads, "v has ", v.size(1), " kv heads, expected ",
              num_kv_heads);
  TORCH_CHECK(k_rot.scalar_type() == v.scalar_type(), "k_rot and v must share a dtype, got ",
              k_rot.scalar_type(), "/", v.scalar_type());
  check_index_tensor(positions, "positions", num_tokens, device);
  check_index_tensor(request_indices, "request_indices", num_tokens, device);
  TORCH_CHECK(positions.scalar_type() == request_indices.scalar_type(),
              "positions and request_indices must share a dtype, got ", positions.scalar_type(),
              "/", request_indices.scalar_type());
  check_cache(k_cache, "k_cache", k_rot, num_kv_heads, head_dim, device);
  check_cache(v_cache, "v_cache", v, num_kv_heads, head_dim, device);
  check_caches_are_distinct(k_cache, v_cache);

  if (num_tokens == 0) return;
  const at::cuda::CUDAGuard guard(device);
  const int64_t work = num_tokens * num_kv_heads * head_dim;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  k_rot.scalar_type(), "kv_append", [&] {
    AT_DISPATCH_INDEX_TYPES(positions.scalar_type(), "kv_append_index", [&] {
      decode_kernels::kv_append_kernel<scalar_t, index_t>
          <<<blocks_for(work), kThreads, 0, stream>>>(
              k_rot.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
              k_cache.data_ptr<scalar_t>(), v_cache.data_ptr<scalar_t>(),
              positions.data_ptr<index_t>(), request_indices.data_ptr<index_t>(),
              num_tokens, static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
              k_rot.stride(0), k_rot.stride(1), v.stride(0), v.stride(1),
              k_cache.size(0), k_cache.size(1));
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
