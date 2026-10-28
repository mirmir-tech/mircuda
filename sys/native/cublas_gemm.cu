// Strided batched cuBLAS GEMM with separate input and output element types.
// Callers pass column-major cuBLAS parameters; the Rust layer maps its
// row-major description onto them.
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <mutex>
#include <new>

#include "cublas_shared.h"

namespace mircuda::cublas_gemm {

// Matches `mircuda_sys::CublasDataType`.
enum class DataType : int { Bf16 = 0, F32 = 1 };

struct Descriptor {
  int transpose_first;
  int transpose_second;
  int m;
  int n;
  int k;
  long long leading_first;
  long long leading_second;
  long long leading_output;
  long long stride_first;
  long long stride_second;
  long long stride_output;
  int batch;
  int input_type;
  int output_type;
};

struct Plan {
  Descriptor descriptor;
  cudaStream_t stream;
  cublas_shared::BlasLease lease;
};

bool known(int type) {
  return type == static_cast<int>(DataType::Bf16) || type == static_cast<int>(DataType::F32);
}

cudaDataType_t cuda_type(int type) {
  return type == static_cast<int>(DataType::F32) ? CUDA_R_32F : CUDA_R_16BF;
}

void release(Plan* plan) {
  if (plan == nullptr) return;
  if (plan->lease.handle != nullptr) cublas_shared::release_blas(plan->lease, plan->stream);
  delete plan;
}

}  // namespace mircuda::cublas_gemm

extern "C" int mircuda_cublas_gemm_create(
    const mircuda::cublas_gemm::Descriptor* descriptor, void* stream, void** output) {
  using namespace mircuda::cublas_gemm;
  if (descriptor == nullptr || stream == nullptr || output == nullptr ||
      descriptor->m <= 0 || descriptor->n <= 0 || descriptor->k <= 0 ||
      descriptor->batch <= 0 || !known(descriptor->input_type) ||
      !known(descriptor->output_type)) {
    return -1;
  }
  auto* plan = new (std::nothrow) Plan{*descriptor, static_cast<cudaStream_t>(stream), {}};
  if (plan == nullptr) return -2;
  const int status = mircuda::cublas_shared::acquire_blas(plan->stream, &plan->lease);
  if (status != CUBLAS_STATUS_SUCCESS) {
    release(plan);
    return status;
  }
  *output = plan;
  return 0;
}

extern "C" int mircuda_cublas_gemm_execute(
    void* raw, void* stream, const void* first, const void* second, void* output,
    float alpha, float beta) {
  using namespace mircuda::cublas_gemm;
  if (raw == nullptr || stream == nullptr || first == nullptr || second == nullptr ||
      output == nullptr) {
    return -1;
  }
  auto* plan = static_cast<Plan*>(raw);
  if (plan->stream != static_cast<cudaStream_t>(stream)) return -1;
  const Descriptor& d = plan->descriptor;
  // F32 inputs never reach tensor cores; BF16 inputs use them.
  const bool single = d.input_type == static_cast<int>(DataType::F32);
  std::lock_guard<std::mutex> guard(*plan->lease.launch);
  const auto launch = [&] {
    return static_cast<int>(cublasGemmStridedBatchedEx(
        plan->lease.handle, d.transpose_first ? CUBLAS_OP_T : CUBLAS_OP_N,
        d.transpose_second ? CUBLAS_OP_T : CUBLAS_OP_N, d.m, d.n, d.k, &alpha, first,
        cuda_type(d.input_type), d.leading_first, d.stride_first, second,
        cuda_type(d.input_type), d.leading_second, d.stride_second, &beta, output,
        cuda_type(d.output_type), d.leading_output, d.stride_output, d.batch,
        CUBLAS_COMPUTE_32F, single ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };
  return single ? mircuda::cublas_shared::with_math_mode(
                      plan->lease.handle, CUBLAS_DEFAULT_MATH, launch)
                : launch();
}

extern "C" void mircuda_cublas_gemm_destroy(void* raw) {
  mircuda::cublas_gemm::release(static_cast<mircuda::cublas_gemm::Plan*>(raw));
}
