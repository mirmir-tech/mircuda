#pragma once

#include <cuda_runtime_api.h>

#define C10_CUDA_CHECK(expression) static_cast<void>(expression)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() static_cast<void>(0)
