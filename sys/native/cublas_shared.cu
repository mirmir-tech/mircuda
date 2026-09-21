#include "cublas_shared.h"

#include <cuda.h>

#include <map>
#include <utility>

namespace mircuda::cublas_shared {
namespace {

constexpr int kCudaStatusBase = 10000;

using Key = std::pair<CUcontext, cudaStream_t>;

struct LtEntry {
  LtLease lease;
  int leases = 0;
};

struct BlasEntry {
  cublasHandle_t handle = nullptr;
  std::mutex launch;
  int leases = 0;
};

std::mutex registry;
std::map<Key, LtEntry> lt_entries;
std::map<Key, BlasEntry> blas_entries;

Key current_key(cudaStream_t stream) {
  CUcontext context = nullptr;
  cuCtxGetCurrent(&context);
  return {context, stream};
}

}  // namespace

int acquire_lt(cudaStream_t stream, LtLease* lease) {
  std::lock_guard<std::mutex> guard(registry);
  const Key key = current_key(stream);
  LtEntry& entry = lt_entries[key];
  if (entry.leases == 0) {
    const cublasStatus_t status = cublasLtCreate(&entry.lease.handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      lt_entries.erase(key);
      return static_cast<int>(status);
    }
    const cudaError_t allocated = cudaMallocAsync(
        &entry.lease.workspace, kLtWorkspaceBytes, stream);
    if (allocated != cudaSuccess) {
      cublasLtDestroy(entry.lease.handle);
      lt_entries.erase(key);
      return kCudaStatusBase + static_cast<int>(allocated);
    }
  }
  ++entry.leases;
  entry.lease.context = key.first;
  *lease = entry.lease;
  return 0;
}

void release_lt(const LtLease& lease, cudaStream_t stream) {
  std::lock_guard<std::mutex> guard(registry);
  const auto found =
      lt_entries.find({static_cast<CUcontext>(lease.context), stream});
  if (found == lt_entries.end() || --found->second.leases > 0) return;
  cudaFreeAsync(found->second.lease.workspace, stream);
  cublasLtDestroy(found->second.lease.handle);
  lt_entries.erase(found);
}

int acquire_blas(cudaStream_t stream, BlasLease* lease) {
  std::lock_guard<std::mutex> guard(registry);
  const Key key = current_key(stream);
  BlasEntry& entry = blas_entries[key];
  if (entry.leases == 0) {
    cublasStatus_t status = cublasCreate(&entry.handle);
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasSetStream(entry.handle, stream);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasSetMathMode(entry.handle, CUBLAS_TENSOR_OP_MATH);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
      if (entry.handle != nullptr) cublasDestroy(entry.handle);
      blas_entries.erase(key);
      return static_cast<int>(status);
    }
  }
  ++entry.leases;
  lease->handle = entry.handle;
  lease->launch = &entry.launch;
  lease->context = key.first;
  return 0;
}

void release_blas(const BlasLease& lease, cudaStream_t stream) {
  std::lock_guard<std::mutex> guard(registry);
  const auto found =
      blas_entries.find({static_cast<CUcontext>(lease.context), stream});
  if (found == blas_entries.end() || --found->second.leases > 0) return;
  cublasDestroy(found->second.handle);
  blas_entries.erase(found);
}

}  // namespace mircuda::cublas_shared
