// HB bounded elementwise ops (kernel-control program S2b).
// Grid-stride variants of vLLM's wave-enumerated elementwise kernels so the
// dense lane's widest launches (silu_and_mul: cdiv(tokens*inter, 1024) CTAs
// under inductor, tokens CTAs under _C) can run under a CTA budget. Budget
// semantics: grid = min(cta_budget, num_tokens); 0 = one CTA per token
// (stock _C width). Outputs are elementwise-identical to the stock op.
// JIT-compiled at import by __init__.py (fmha_sm100 in-tree precedent).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

template <typename T>
__device__ inline float to_f(T v);
template <>
__device__ inline float to_f<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ inline float to_f<__half>(__half v) {
  return __half2float(v);
}

template <typename T>
__device__ inline T from_f(float v);
template <>
__device__ inline __nv_bfloat16 from_f<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <>
__device__ inline __half from_f<__half>(float v) {
  return __float2half(v);
}

// silu(x[:d]) * x[d:], row-per-block with a grid-stride row loop.
// VEC=8 halves/bf16s per uint4 (16B) load; d % 8 == 0 required.
template <typename T, int VEC>
__global__ void silu_mul_bounded_kernel(T* __restrict__ out,
                                        const T* __restrict__ in,
                                        int64_t num_tokens, int64_t d) {
  for (int64_t token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    const T* xrow = in + token * 2 * d;
    const T* yrow = xrow + d;
    T* orow = out + token * d;
    const int64_t nvec = d / VEC;
    for (int64_t i = threadIdx.x; i < nvec; i += blockDim.x) {
      T xv[VEC], yv[VEC], ov[VEC];
      *reinterpret_cast<uint4*>(xv) = reinterpret_cast<const uint4*>(xrow)[i];
      *reinterpret_cast<uint4*>(yv) = reinterpret_cast<const uint4*>(yrow)[i];
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        float x = to_f(xv[j]);
        // Mirror the stock _C op's rounding: silu is rounded to T first
        // (packed_silu_kernel), then the product rounds again.
        float s = to_f(from_f<T>(x / (1.0f + expf(-x))));
        ov[j] = from_f<T>(s * to_f(yv[j]));
      }
      reinterpret_cast<uint4*>(orow)[i] = *reinterpret_cast<uint4*>(ov);
    }
  }
}

}  // namespace

void silu_and_mul_bounded(at::Tensor out, at::Tensor x, int64_t cta_budget) {
  TORCH_CHECK(x.is_cuda() && out.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "contiguous required");
  const int64_t d = x.size(-1) / 2;
  const int64_t num_tokens = x.numel() / (2 * d);
  TORCH_CHECK(out.numel() == num_tokens * d, "output shape mismatch");
  TORCH_CHECK(d % 8 == 0, "d must be a multiple of 8");
  const int64_t grid =
      (cta_budget > 0 && cta_budget < num_tokens) ? cta_budget : num_tokens;
  const dim3 block(256);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == at::kBFloat16) {
    silu_mul_bounded_kernel<__nv_bfloat16, 8>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), num_tokens,
            d);
  } else if (x.scalar_type() == at::kHalf) {
    silu_mul_bounded_kernel<__half, 8><<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(out.data_ptr()),
        reinterpret_cast<const __half*>(x.data_ptr()), num_tokens, d);
  } else {
    TORCH_CHECK(false, "bf16/fp16 only");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_and_mul_bounded", &silu_and_mul_bounded, py::arg("out"),
        py::arg("x"), py::arg("cta_budget"));
}
