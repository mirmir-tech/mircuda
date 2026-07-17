#pragma once

#include <cuda_runtime_api.h>

namespace mircuda::dense_vector {

struct Plan {
  int data_type;
  int n;
  int k;
  cudaStream_t stream;
};

}  // namespace mircuda::dense_vector
