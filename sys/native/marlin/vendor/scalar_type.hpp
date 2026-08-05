#pragma once

#include <cstdint>

namespace vllm {

using ScalarTypeId = int64_t;

struct ScalarType {
  ScalarTypeId value;
  constexpr ScalarTypeId id() const { return value; }
  constexpr bool operator==(ScalarType other) const { return value == other.value; }
  constexpr bool operator!=(ScalarType other) const { return value != other.value; }
  constexpr int size_bits() const {
    return value == 1 || value == 2 || value == 3 ? 4 :
           value == 4 || value == 5 || value == 6 ? 8 : 16;
  }
  static constexpr ScalarType from_id(ScalarTypeId id) { return {id}; }
};

inline constexpr ScalarType kFE2M1f{1};
inline constexpr ScalarType kU4{2};
inline constexpr ScalarType kU4B8{3};
inline constexpr ScalarType kFE4M3fn{4};
inline constexpr ScalarType kFE8M0fnu{5};
inline constexpr ScalarType kS8{6};
inline constexpr ScalarType kFloat16{7};
inline constexpr ScalarType kBFloat16{8};
inline constexpr ScalarType kU8{9};
inline constexpr ScalarType kU8B128{10};
inline constexpr ScalarType kS4{11};

}  // namespace vllm
