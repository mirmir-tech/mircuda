// Shared cuBLAS state for plans that enqueue on one stream of one context.
//
// Creating a handle per plan costs a host allocation and a device workspace,
// and destroying one synchronizes the device. Plans therefore lease the handle
// and the cuBLASLt workspace of their stream; the last lease releases them.
#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <mutex>

namespace mircuda::cublas_shared {

constexpr size_t kLtWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;

struct LtLease {
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  // Context current at acquisition; a release may run without one.
  void* context = nullptr;
};

struct BlasLease {
  cublasHandle_t handle = nullptr;
  // cuBLAS handles are not reentrant; launches through a lease hold this.
  std::mutex* launch = nullptr;
  void* context = nullptr;
};

// Each returns a cuBLAS status, or a CUDA runtime error offset by 10000.
int acquire_lt(cudaStream_t stream, LtLease* lease);
void release_lt(const LtLease& lease, cudaStream_t stream);
int acquire_blas(cudaStream_t stream, BlasLease* lease);
void release_blas(const BlasLease& lease, cudaStream_t stream);

}  // namespace mircuda::cublas_shared
