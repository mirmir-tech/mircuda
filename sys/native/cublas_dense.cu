#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <mutex>
#include <new>

#include "cublas_shared.h"

namespace mircuda::cublas_dense {

// Element type of every operand; matches `mircuda_sys::CublasDataType`.
enum class DataType : int { Bf16 = 0, F32 = 1 };

struct Plan {
  int m;
  int n;
  int k;
  DataType data_type;
  cudaStream_t stream;
  cublas_shared::BlasLease lease;
};

void release(Plan* plan) {
  if (plan == nullptr) return;
  if (plan->lease.handle != nullptr) {
    cublas_shared::release_blas(plan->lease, plan->stream);
  }
  delete plan;
}

int prepare(Plan* plan) {
  return cublas_shared::acquire_blas(plan->stream, &plan->lease);
}

}  // namespace mircuda::cublas_dense

extern "C" int mircuda_cublas_dense_create(
    int m, int n, int k, int data_type, void* stream, void** output) {
  using namespace mircuda::cublas_dense;
  if (m <= 0 || n <= 0 || k <= 0 || stream == nullptr || output == nullptr ||
      (data_type != static_cast<int>(DataType::Bf16) &&
       data_type != static_cast<int>(DataType::F32))) {
    return -1;
  }
  auto* plan = new (std::nothrow) Plan{
      m, n, k, static_cast<DataType>(data_type), static_cast<cudaStream_t>(stream), {}};
  if (plan == nullptr) return -2;
  const int status = prepare(plan);
  if (status != CUBLAS_STATUS_SUCCESS) {
    release(plan);
    return status;
  }
  *output = plan;
  return 0;
}

extern "C" int mircuda_cublas_dense_execute(
    void* raw, void* stream, const void* a, const void* b, void* c,
    float alpha, float beta) {
  using namespace mircuda::cublas_dense;
  if (raw == nullptr || stream == nullptr || a == nullptr || b == nullptr ||
      c == nullptr) {
    return -1;
  }
  auto* plan = static_cast<Plan*>(raw);
  if (plan->stream != static_cast<cudaStream_t>(stream)) return -1;
  std::lock_guard<std::mutex> guard(*plan->lease.launch);
  // F32 uses the pedantic compute type so the shared tensor-op math mode can
  // never lower it to TF32.
  const bool single = plan->data_type == DataType::F32;
  const cudaDataType_t type = single ? CUDA_R_32F : CUDA_R_16BF;
  return static_cast<int>(cublasGemmEx(
      plan->lease.handle, CUBLAS_OP_T, CUBLAS_OP_N, plan->n, plan->m, plan->k,
      &alpha, b, type, plan->k, a, type, plan->k, &beta, c, type, plan->n,
      single ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,
      single ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

extern "C" void mircuda_cublas_dense_destroy(void* raw) {
  mircuda::cublas_dense::release(
      static_cast<mircuda::cublas_dense::Plan*>(raw));
}
