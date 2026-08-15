#include "kernel_common.cuh"

namespace decode_kernels {
namespace {

// The primary path: one launch, and the rotated K goes straight into its cache slot without ever
// being materialized. That is the whole difference from the separate-kernels rung, which is two
// launches and a [T, Hkv, D] intermediate -- both share this file's rotation and addressing, so
// nothing else varies between them.
//
// Three head groups in one flat index space, each a pair of elements per thread: Q rotate, K
// rotate-and-scatter, V copy. Splitting V out rather than folding it into the K group keeps the
// per-thread work equal, which matters at MHA where Hq == Hkv.
template <typename scalar_t, typename index_t>
__global__ void fused_rope_kv_append_kernel(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_cache, scalar_t* __restrict__ v_cache,
    const float* __restrict__ cos_table, const float* __restrict__ sin_table,
    const index_t* __restrict__ positions, const index_t* __restrict__ request_indices,
    int64_t num_tokens, int num_q_heads, int num_kv_heads, int head_dim,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t v_token_stride, int64_t v_head_stride,
    int64_t table_stride, int64_t max_position,
    int64_t batch_size, int64_t cache_alloc_len) {
  const int half = head_dim / 2;
  const int heads_total = num_q_heads + 2 * num_kv_heads;
  const int64_t work = num_tokens * static_cast<int64_t>(heads_total) * half;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t per_token = static_cast<int64_t>(num_kv_heads) * head_dim;

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < work; idx += stride) {
    // Divided, not masked: head_dim 96 makes half 48.
    const int pair = static_cast<int>(idx % half);
    const int64_t rest = idx / half;
    const int head = static_cast<int>(rest % heads_total);
    const int64_t token = rest / heads_total;

    // Loaded once for the whole body: the K branch used to read it again for its table row.
    const int64_t position = static_cast<int64_t>(positions[token]);

    if (head < num_q_heads) {
      const int64_t row = table_offset(position, max_position, table_stride);
      const float c = cos_table[row + pair];
      const float s = sin_table[row + pair];
      const int64_t in_base = token * q_token_stride + static_cast<int64_t>(head) * q_head_stride;
      const int64_t out_base = (token * num_q_heads + head) * head_dim;
      float rotated1, rotated2;
      rotate_pair(static_cast<float>(q[in_base + pair]),
                  static_cast<float>(q[in_base + pair + half]), c, s, &rotated1, &rotated2);
      q_out[out_base + pair] = static_cast<scalar_t>(rotated1);
      q_out[out_base + pair + half] = static_cast<scalar_t>(rotated2);
      continue;
    }

    const int64_t slot = cache_offset(static_cast<int64_t>(request_indices[token]), position,
                                      batch_size, cache_alloc_len, per_token);
    if (head < num_q_heads + num_kv_heads) {
      const int kv_head = head - num_q_heads;
      const int64_t row = table_offset(position, max_position, table_stride);
      const float c = cos_table[row + pair];
      const float s = sin_table[row + pair];
      const int64_t in_base =
          token * k_token_stride + static_cast<int64_t>(kv_head) * k_head_stride;
      const int64_t out = slot + static_cast<int64_t>(kv_head) * head_dim;
      float rotated1, rotated2;
      rotate_pair(static_cast<float>(k[in_base + pair]),
                  static_cast<float>(k[in_base + pair + half]), c, s, &rotated1, &rotated2);
      k_cache[out + pair] = static_cast<scalar_t>(rotated1);
      k_cache[out + pair + half] = static_cast<scalar_t>(rotated2);
      continue;
    }

    // V is copied, never rotated and never round-tripped through float: the gate asserts the
    // written V slots are byte-equal to the input.
    const int kv_head = head - num_q_heads - num_kv_heads;
    const int64_t in_base = token * v_token_stride + static_cast<int64_t>(kv_head) * v_head_stride;
    const int64_t out = slot + static_cast<int64_t>(kv_head) * head_dim;
    v_cache[out + pair] = v[in_base + pair];
    v_cache[out + pair + half] = v[in_base + pair + half];
  }
}

}  // namespace
}  // namespace decode_kernels

at::Tensor fused_rope_kv_append(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor positions,
                                at::Tensor cos, at::Tensor sin, at::Tensor k_cache,
                                at::Tensor v_cache, at::Tensor request_indices) {
  using namespace decode_kernels;
  TORCH_CHECK(q.dim() == 3, "q must be [num_tokens, num_q_heads, head_dim]");
  const int64_t num_tokens = q.size(0);
  const int64_t num_q_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even for split-half RoPE, got ", head_dim);
  const at::Device device = q.device();
  check_qkv(q, "q", num_tokens, head_dim, device);
  check_qkv(k, "k", num_tokens, head_dim, device);
  check_qkv(v, "v", num_tokens, head_dim, device);
  const int64_t num_kv_heads = k.size(1);
  TORCH_CHECK(v.size(1) == num_kv_heads, "v has ", v.size(1), " kv heads, expected ",
              num_kv_heads);
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q, k and v must share a dtype");
  check_index_tensor(positions, "positions", num_tokens, device);
  check_index_tensor(request_indices, "request_indices", num_tokens, device);
  TORCH_CHECK(positions.scalar_type() == request_indices.scalar_type(),
              "positions and request_indices must share a dtype, got ", positions.scalar_type(),
              "/", request_indices.scalar_type());
  check_rope_tables(cos, sin, head_dim, device);
  check_cache(k_cache, "k_cache", k, num_kv_heads, head_dim, device);
  check_cache(v_cache, "v_cache", v, num_kv_heads, head_dim, device);
  check_caches_are_distinct(k_cache, v_cache);

  const at::cuda::CUDAGuard guard(device);
  auto q_out = at::empty({num_tokens, num_q_heads, head_dim}, q.options());
  if (num_tokens == 0) return q_out;

  const int64_t work = num_tokens * (num_q_heads + 2 * num_kv_heads) * (head_dim / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  q.scalar_type(), "fused_rope_kv_append", [&] {
    AT_DISPATCH_INDEX_TYPES(positions.scalar_type(), "fused_rope_kv_append_index", [&] {
      decode_kernels::fused_rope_kv_append_kernel<scalar_t, index_t>
          <<<blocks_for(work), kThreads, 0, stream>>>(
              q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
              q_out.data_ptr<scalar_t>(), k_cache.data_ptr<scalar_t>(),
              v_cache.data_ptr<scalar_t>(),
              cos.data_ptr<float>(), sin.data_ptr<float>(),
              positions.data_ptr<index_t>(), request_indices.data_ptr<index_t>(),
              num_tokens, static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
              static_cast<int>(head_dim),
              q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
              cos.stride(0), cos.size(0), k_cache.size(0), k_cache.size(1));
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return q_out;
}
