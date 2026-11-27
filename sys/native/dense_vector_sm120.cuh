#pragma once

#include <cuda_runtime_api.h>

namespace mircuda::dense_vector {

// Matches the native codes of `DenseVectorKernel`.
enum class Kernel : int { RowGroups = 0, BlockPerRow = 1, SplitK = 2 };

struct Plan {
  int data_type;
  int n;
  int k;
  Kernel kernel;
  // Input slices per row; 1 except for `Kernel::SplitK`.
  int splits;
  cudaStream_t stream;
  // Split-K only: `n` arrival counters followed by `n * splits` FP32
  // partials, allocated with the plan so launches stay capture-safe.
  void* workspace;
  size_t workspace_bytes;
};

}  // namespace mircuda::dense_vector
