#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <new>

namespace mircuda::cublas_dense {

struct Plan {
  int m;
  int n;
  int k;
  cudaStream_t stream;
  cublasHandle_t handle;
};

void release(Plan* plan) {
  if (plan == nullptr) return;
  if (plan->handle != nullptr) cublasDestroy(plan->handle);
  delete plan;
}

int prepare(Plan* plan) {
  cublasStatus_t status = cublasCreate(&plan->handle);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = cublasSetStream(plan->handle, plan->stream);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = cublasSetMathMode(plan->handle, CUBLAS_TENSOR_OP_MATH);
  return static_cast<int>(status);
}

}  // namespace mircuda::cublas_dense

extern "C" int mircuda_cublas_dense_create(
    int m, int n, int k, void* stream, void** output) {
  using namespace mircuda::cublas_dense;
  if (m <= 0 || n <= 0 || k <= 0 || stream == nullptr || output == nullptr) {
    return -1;
  }
  auto* plan = new (std::nothrow)
      Plan{m, n, k, static_cast<cudaStream_t>(stream), nullptr};
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
  return static_cast<int>(cublasGemmEx(
      plan->handle, CUBLAS_OP_T, CUBLAS_OP_N, plan->n, plan->m, plan->k,
      &alpha, b, CUDA_R_16BF, plan->k, a, CUDA_R_16BF, plan->k, &beta, c,
      CUDA_R_16BF, plan->n, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

extern "C" void mircuda_cublas_dense_destroy(void* raw) {
  mircuda::cublas_dense::release(
      static_cast<mircuda::cublas_dense::Plan*>(raw));
}
