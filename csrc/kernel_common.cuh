#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

namespace decode_kernels {

constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = 65535;

inline int blocks_for(int64_t work) {
  return static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>((work + kThreads - 1) / kThreads, kMaxBlocks)));
}

// One split-half RoPE pair, FP32 regardless of input dtype.
//
// From rot = concat(-x2, x1) and x' = x*cos + rot*sin, given that the tables are built as
// concat(f, f) so cos[i] == cos[i + half]. That identity is why a thread owns a *pair* rather
// than an element: it loads two table floats instead of four.
__device__ __forceinline__ void rotate_pair(float x1, float x2, float c, float s,
                                            float* out1, float* out2) {
  *out1 = x1 * c - x2 * s;
  *out2 = x2 * c + x1 * s;
}

// Where one token's K/V lands in a contiguous token-major cache [B, S, Hkv, D].
//
// The addressing contract makes in-bounds the caller's guarantee, and these asserts cost a
// compare with no host work and no synchronization. Without them an out-of-range index writes
// outside the cache allocation, which is strictly worse than the reference: the locked semantics
// record that out-of-range positive indices already fail loudly with a device-side assert.
__device__ __forceinline__ int64_t cache_offset(int64_t request_index, int64_t position,
                                                int64_t batch_size, int64_t cache_alloc_len,
                                                int64_t per_token) {
  CUDA_KERNEL_ASSERT(position >= 0 && position < cache_alloc_len);
  CUDA_KERNEL_ASSERT(request_index >= 0 && request_index < batch_size);
  return (request_index * cache_alloc_len + position) * per_token;
}

// Same reasoning for the table gather: a position past the table reads out of bounds.
__device__ __forceinline__ int64_t table_offset(int64_t position, int64_t max_position,
                                                int64_t table_stride) {
  CUDA_KERNEL_ASSERT(position >= 0 && position < max_position);
  return position * table_stride;
}

// Every tensor must live on the one device the kernel is launched on. Without this a host
// pointer reaches the device and is dereferenced there: a CPU cos table was accepted and
// "worked", which is undefined behaviour wearing a passing result.
inline void check_device(const at::Tensor& t, const char* name, const at::Device& device) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor, got ", t.device());
  TORCH_CHECK(t.device() == device, name, " is on ", t.device(), " but the operation runs on ",
              device, "; every tensor must be on one device");
}

inline void check_rope_tables(const at::Tensor& cos, const at::Tensor& sin, int64_t head_dim,
                              const at::Device& device) {
  check_device(cos, "cos", device);
  check_device(sin, "sin", device);
  TORCH_CHECK(cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat,
              "cos/sin must be FP32 per the locked RoPE semantics, got ", cos.scalar_type(),
              "/", sin.scalar_type());
  TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(), "cos/sin must be contiguous");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, "cos/sin must be [max_position, head_dim]");
  TORCH_CHECK(cos.size(1) == head_dim && sin.size(1) == head_dim,
              "cos/sin head_dim ", cos.size(1), " does not match q/k head_dim ", head_dim);
  TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin must have the same shape");
}

// head_dim (the last dim) must be contiguous; token and head dims may carry arbitrary strides,
// because a fused QKV projection is split into views whose token stride is the fused width.
inline void check_qkv(const at::Tensor& t, const char* name, int64_t num_tokens,
                      int64_t head_dim, const at::Device& device) {
  check_device(t, name, device);
  TORCH_CHECK(t.dim() == 3, name, " must be [num_tokens, num_heads, head_dim], got ", t.dim(),
              " dims");
  TORCH_CHECK(t.size(0) == num_tokens, name, " has ", t.size(0), " tokens, expected ",
              num_tokens);
  TORCH_CHECK(t.size(2) == head_dim, name, " head_dim ", t.size(2), " expected ", head_dim);
  TORCH_CHECK(t.stride(2) == 1, name, " needs a contiguous head_dim (last-dim stride ",
              t.stride(2), ")");
}

inline void check_index_tensor(const at::Tensor& t, const char* name, int64_t num_tokens,
                               const at::Device& device) {
  check_device(t, name, device);
  TORCH_CHECK(t.scalar_type() == at::kInt || t.scalar_type() == at::kLong,
              name, " must be int32 or int64, got ", t.scalar_type());
  TORCH_CHECK(t.dim() == 1 && t.size(0) == num_tokens,
              name, " must be [num_tokens]; got ", t.sizes());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

inline void check_cache(const at::Tensor& cache, const char* name, const at::Tensor& like,
                       int64_t num_kv_heads, int64_t head_dim, const at::Device& device) {
  check_device(cache, name, device);
  TORCH_CHECK(cache.is_contiguous(), name, " must be contiguous, token-major");
  TORCH_CHECK(cache.dim() == 4, name, " must be [batch, cache_alloc_len, num_kv_heads, "
              "head_dim], got ", cache.dim(), " dims");
  TORCH_CHECK(cache.size(2) == num_kv_heads && cache.size(3) == head_dim,
              name, " kv-head/head_dim ", cache.size(2), "/", cache.size(3), " expected ",
              num_kv_heads, "/", head_dim);
  // A raw scalar copy is what makes V byte-exact; a differing cache dtype would need a cast
  // and would silently stop being byte-exact.
  TORCH_CHECK(cache.scalar_type() == like.scalar_type(),
              name, " dtype ", cache.scalar_type(), " must match its source dtype ",
              like.scalar_type());
}

inline void check_caches_are_distinct(const at::Tensor& k_cache, const at::Tensor& v_cache) {
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(),
              "k_cache and v_cache must have the same shape");
  // Guarded on numel: two distinct empty tensors both report data_ptr() == 0, so an unguarded
  // pointer compare rejects a legitimate zero-token call. Pointer equality catches the realistic
  // case; it is not full overlap detection, and two views into one storage would still race.
  TORCH_CHECK(k_cache.numel() == 0 || k_cache.data_ptr() != v_cache.data_ptr(),
              "k_cache and v_cache must be distinct buffers; sharing one makes the K and V "
              "writes race for the same slot");
}

}  // namespace decode_kernels
