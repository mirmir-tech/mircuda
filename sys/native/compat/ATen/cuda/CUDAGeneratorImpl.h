#pragma once

#include <cstdint>

namespace at {

struct PhiloxCudaState {
  std::uint64_t seed = 0;
  std::uint64_t offset = 0;
};

}  // namespace at
