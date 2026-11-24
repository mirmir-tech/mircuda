#pragma once

#include <cuda_runtime_api.h>

namespace mircuda::dense_vector {

// Matches the native codes of `DenseVectorKernel`.
enum class Kernel : int { RowGroups = 0, BlockPerRow = 1 };

struct Plan {
  int data_type;
  int n;
  int k;
  Kernel kernel;
  cudaStream_t stream;
};

}  // namespace mircuda::dense_vector
