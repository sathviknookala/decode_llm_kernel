#include "kernel_common.cuh"

namespace decode_kernels {
namespace {

// Q heads then KV heads in one flat index space, so this rung is one launch and its comparison
// against the fused kernel is a launch-count and materialization story rather than a shape one.
//
// The flat grid-stride loop is deliberate: at b=1 with Hq=32 and half=64 there are only 2048
// pairs, and a grid keyed on tokens or heads leaves most of the device idle at exactly the
// batch sizes the sweep spends most of its rows on.
template <typename scalar_t, typename index_t>
__global__ void rope_kernel(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    scalar_t* __restrict__ q_out, scalar_t* __restrict__ k_out,
    const float* __restrict__ cos_table, const float* __restrict__ sin_table,
    const index_t* __restrict__ positions,
    int64_t num_tokens, int num_q_heads, int num_kv_heads, int head_dim,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t table_stride, int64_t max_position) {
  const int half = head_dim / 2;
  const int heads_total = num_q_heads + num_kv_heads;
  const int64_t work = num_tokens * static_cast<int64_t>(heads_total) * half;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < work; idx += stride) {
    // Divided, not masked: head_dim 96 makes half 48, so nothing here may assume a power of two.
    const int pair = static_cast<int>(idx % half);
    const int64_t rest = idx / half;
    const int head = static_cast<int>(rest % heads_total);
    const int64_t token = rest / heads_total;

    const int64_t row = table_offset(static_cast<int64_t>(positions[token]), max_position,
                                    table_stride);
    const float c = cos_table[row + pair];
    const float s = sin_table[row + pair];

    const bool is_q = head < num_q_heads;
    const scalar_t* src = is_q ? q : k;
    scalar_t* dst = is_q ? q_out : k_out;
    const int local_head = is_q ? head : head - num_q_heads;
    const int64_t token_stride = is_q ? q_token_stride : k_token_stride;
    const int64_t head_stride = is_q ? q_head_stride : k_head_stride;
    const int num_heads_out = is_q ? num_q_heads : num_kv_heads;

    const int64_t in_base = token * token_stride + static_cast<int64_t>(local_head) * head_stride;
    const int64_t out_base = (token * num_heads_out + local_head) * head_dim;

    float rotated1, rotated2;
    rotate_pair(static_cast<float>(src[in_base + pair]),
                static_cast<float>(src[in_base + pair + half]), c, s, &rotated1, &rotated2);
    dst[out_base + pair] = static_cast<scalar_t>(rotated1);
    dst[out_base + pair + half] = static_cast<scalar_t>(rotated2);
  }
}

}  // namespace
}  // namespace decode_kernels

std::vector<at::Tensor> rope_forward(at::Tensor q, at::Tensor k, at::Tensor positions,
                                     at::Tensor cos, at::Tensor sin) {
  using namespace decode_kernels;
  TORCH_CHECK(q.dim() == 3, "q must be [num_tokens, num_q_heads, head_dim]");
  const int64_t num_tokens = q.size(0);
  const int64_t num_q_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even for split-half RoPE, got ", head_dim);
  check_qkv(q, "q", num_tokens, head_dim);
  check_qkv(k, "k", num_tokens, head_dim);
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must share a dtype, got ",
              q.scalar_type(), "/", k.scalar_type());
  check_index_tensor(positions, "positions", num_tokens);
  check_rope_tables(cos, sin, head_dim);
  const int64_t num_kv_heads = k.size(1);

  const at::cuda::CUDAGuard guard(q.device());
  auto q_out = at::empty({num_tokens, num_q_heads, head_dim}, q.options());
  auto k_out = at::empty({num_tokens, num_kv_heads, head_dim}, k.options());
  if (num_tokens == 0) return {q_out, k_out};

  const int64_t work = num_tokens * (num_q_heads + num_kv_heads) * (head_dim / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  q.scalar_type(), "rope_forward", [&] {
    AT_DISPATCH_INDEX_TYPES(positions.scalar_type(), "rope_forward_index", [&] {
      decode_kernels::rope_kernel<scalar_t, index_t>
          <<<blocks_for(work), kThreads, 0, stream>>>(
              q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
              q_out.data_ptr<scalar_t>(), k_out.data_ptr<scalar_t>(),
              cos.data_ptr<float>(), sin.data_ptr<float>(),
              positions.data_ptr<index_t>(),
              num_tokens, static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
              static_cast<int>(head_dim),
              q.stride(0), q.stride(1), k.stride(0), k.stride(1),
              cos.stride(0), cos.size(0));
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q_out, k_out};
}
