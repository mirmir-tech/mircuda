#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <new>

namespace mircuda::cublaslt_fp8 {

constexpr size_t kMaximumWorkspace = 32ULL * 1024ULL * 1024ULL;
constexpr int kCublasStatusBase = 2000;

struct Plan {
  int m;
  int n;
  int k;
  cudaStream_t stream;
  cublasLtHandle_t handle;
  cublasLtMatmulDesc_t operation;
  cublasLtMatrixLayout_t a_layout;
  cublasLtMatrixLayout_t b_layout;
  cublasLtMatrixLayout_t c_layout;
  cublasLtMatmulAlgo_t algorithm;
  void* workspace;
  size_t workspace_bytes;
};

int cublas_status(cublasStatus_t status) {
  return status == CUBLAS_STATUS_SUCCESS
             ? 0
             : kCublasStatusBase + static_cast<int>(status);
}

int set_row_order(cublasLtMatrixLayout_t layout) {
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  return cublas_status(cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
}

void release(Plan* plan) {
  if (plan == nullptr) return;
  if (plan->workspace != nullptr) {
    cudaFreeAsync(plan->workspace, plan->stream);
  }
  if (plan->c_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->c_layout);
  if (plan->b_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->b_layout);
  if (plan->a_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->a_layout);
  if (plan->operation != nullptr) cublasLtMatmulDescDestroy(plan->operation);
  if (plan->handle != nullptr) cublasLtDestroy(plan->handle);
  delete plan;
}

int prepare(Plan* plan) {
  int status = cublas_status(cublasLtCreate(&plan->handle));
  if (status != 0) return status;
  status = cublas_status(cublasLtMatmulDescCreate(
      &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  if (status != 0) return status;
  const cublasOperation_t transpose = CUBLAS_OP_T;
  status = cublas_status(cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
      sizeof(transpose)));
  if (status != 0) return status;
  const int8_t fast_accumulation = 0;
  status = cublas_status(cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
      &fast_accumulation, sizeof(fast_accumulation)));
  if (status != 0) return status;
  status = cublas_status(cublasLtMatrixLayoutCreate(
      &plan->a_layout, CUDA_R_8F_E4M3, plan->m, plan->k, plan->k));
  if (status != 0) return status;
  status = set_row_order(plan->a_layout);
  if (status != 0) return status;
  status = cublas_status(cublasLtMatrixLayoutCreate(
      &plan->b_layout, CUDA_R_8F_E4M3, plan->n, plan->k, plan->k));
  if (status != 0) return status;
  status = set_row_order(plan->b_layout);
  if (status != 0) return status;
  status = cublas_status(cublasLtMatrixLayoutCreate(
      &plan->c_layout, CUDA_R_16BF, plan->m, plan->n, plan->n));
  if (status != 0) return status;
  status = set_row_order(plan->c_layout);
  if (status != 0) return status;

  cublasLtMatmulPreference_t preference = nullptr;
  status = cublas_status(cublasLtMatmulPreferenceCreate(&preference));
  if (status != 0) return status;
  status = cublas_status(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &kMaximumWorkspace, sizeof(kMaximumWorkspace)));
  cublasLtMatmulHeuristicResult_t heuristic{};
  int algorithms = 0;
  if (status == 0) {
    status = cublas_status(cublasLtMatmulAlgoGetHeuristic(
        plan->handle, plan->operation, plan->a_layout, plan->b_layout,
        plan->c_layout, plan->c_layout, preference, 1, &heuristic,
        &algorithms));
  }
  cublasLtMatmulPreferenceDestroy(preference);
  if (status != 0) return status;
  if (algorithms != 1) return kCublasStatusBase + CUBLAS_STATUS_NOT_SUPPORTED;
  plan->algorithm = heuristic.algo;
  plan->workspace_bytes = heuristic.workspaceSize;
  if (plan->workspace_bytes > 0) {
    return static_cast<int>(cudaMallocAsync(
        &plan->workspace, plan->workspace_bytes, plan->stream));
  }
  return 0;
}

}  // namespace mircuda::cublaslt_fp8

extern "C" int mircuda_cublaslt_fp8_create(
    int m, int n, int k, void* stream, void** output) {
  using namespace mircuda::cublaslt_fp8;
  if (m <= 0 || n <= 0 || k <= 0 || stream == nullptr || output == nullptr) {
    return -1;
  }
  auto* plan = new (std::nothrow) Plan{
      m, n, k, static_cast<cudaStream_t>(stream), nullptr, nullptr,
      nullptr, nullptr, nullptr, {}, nullptr, 0};
  if (plan == nullptr) return -2;
  const int status = prepare(plan);
  if (status != 0) {
    release(plan);
    return status;
  }
  *output = plan;
  return 0;
}

extern "C" int mircuda_cublaslt_fp8_execute(
    void* raw, void* stream, const void* a, const void* b,
    const float* a_scale, const float* b_scale, void* c) {
  using namespace mircuda::cublaslt_fp8;
  if (raw == nullptr || stream == nullptr || a == nullptr || b == nullptr ||
      a_scale == nullptr || b_scale == nullptr || c == nullptr) return -1;
  auto* plan = static_cast<Plan*>(raw);
  if (plan->stream != static_cast<cudaStream_t>(stream)) return -1;
  const void* a_scale_pointer = a_scale;
  const void* b_scale_pointer = b_scale;
  int status = cublas_status(cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &a_scale_pointer, sizeof(a_scale_pointer)));
  if (status != 0) return status;
  status = cublas_status(cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &b_scale_pointer, sizeof(b_scale_pointer)));
  if (status != 0) return status;
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  return cublas_status(cublasLtMatmul(
      plan->handle, plan->operation, &alpha, a, plan->a_layout, b,
      plan->b_layout, &beta, nullptr, plan->c_layout, c, plan->c_layout,
      &plan->algorithm, plan->workspace, plan->workspace_bytes, plan->stream));
}

extern "C" void mircuda_cublaslt_fp8_destroy(void* raw) {
  mircuda::cublaslt_fp8::release(
      static_cast<mircuda::cublaslt_fp8::Plan*>(raw));
}
