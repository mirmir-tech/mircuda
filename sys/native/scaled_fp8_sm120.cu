#include "scaled_fp8_sm120.cuh"

#include <new>

namespace mircuda::scaled_fp8 {
using SmallTile = Shape<_16, _64, _128>;
using WideTile = Shape<_16, _128, _64>;
using LargeTile = Shape<_128, _128, _128>;
using SmallSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using LargeSchedule = cutlass::gemm::collective::KernelScheduleAuto;

template <typename Scale, bool WithBias, bool TensorScale, typename Tile,
          typename Schedule, typename EpilogueTile>
auto arguments(const Plan& plan, const void* input, const void* weight,
               const float* input_scales, const void* weight_scales,
               const void* bias, void* output) {
  using K = Kernel<Scale, WithBias, TensorScale, Tile, Schedule, EpilogueTile>;
  using GemmKernel = typename K::GemmKernel;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  auto shape = make_shape(plan.m, plan.n, plan.k, 1);
  auto stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, make_shape(plan.m, plan.k, 1));
  auto stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, make_shape(plan.n, plan.k, 1));
  auto stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, make_shape(plan.m, plan.n, 1));
  auto visitor = K::Visitor::arguments(
      input_scales, static_cast<const Scale*>(weight_scales),
      static_cast<const Output*>(bias));
  return typename K::Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, shape,
      {static_cast<const Element*>(input), stride_a,
       static_cast<const Element*>(weight), stride_b},
      {visitor, nullptr, stride_d, static_cast<Output*>(output), stride_d}};
}

template <typename Scale, bool WithBias, bool TensorScale, typename Tile,
          typename Schedule, typename EpilogueTile>
int workspace(const Plan& plan, size_t* bytes) {
  using K = Kernel<Scale, WithBias, TensorScale, Tile, Schedule, EpilogueTile>;
  auto args = arguments<Scale, WithBias, TensorScale, Tile, Schedule, EpilogueTile>(
      plan, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
  const auto status = K::Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return 100 + static_cast<int>(status);
  }
  *bytes = K::Gemm::get_workspace_size(args);
  return 0;
}

template <typename Scale, bool WithBias, bool TensorScale, typename Tile,
          typename Schedule, typename EpilogueTile>
int execute(const Plan& plan, const void* input, const void* weight,
            const float* input_scales, const void* weight_scales,
            const void* bias, void* output) {
  using K = Kernel<Scale, WithBias, TensorScale, Tile, Schedule, EpilogueTile>;
  auto args = arguments<Scale, WithBias, TensorScale, Tile, Schedule, EpilogueTile>(
      plan, input, weight, input_scales, weight_scales, bias, output);
  typename K::Gemm gemm;
  const auto status = gemm.run(args, plan.workspace, plan.stream);
  if (status != cutlass::Status::kSuccess) {
    return 100 + static_cast<int>(status);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

template <typename Scale, bool WithBias, bool TensorScale>
int workspace_shape(const Plan& plan, size_t* bytes) {
  if (plan.m <= 16) {
    if (plan.tile == 1) {
      return workspace<Scale, WithBias, TensorScale, WideTile, SmallSchedule,
                       Shape<_16, _64>>(plan, bytes);
    }
    return workspace<Scale, WithBias, TensorScale, SmallTile, SmallSchedule,
                     Shape<_16, _32>>(plan, bytes);
  }
  return workspace<Scale, WithBias, TensorScale, LargeTile, LargeSchedule,
                   cutlass::epilogue::collective::EpilogueTileAuto>(plan,
                                                                    bytes);
}

template <typename Scale, bool WithBias, bool TensorScale>
int execute_shape(const Plan& plan, const void* input, const void* weight,
                  const float* input_scales, const void* weight_scales,
                  const void* bias, void* output) {
  if (plan.m <= 16) {
    if (plan.tile == 1) {
      return execute<Scale, WithBias, TensorScale, WideTile, SmallSchedule,
                     Shape<_16, _64>>(plan, input, weight, input_scales,
                                      weight_scales, bias, output);
    }
    return execute<Scale, WithBias, TensorScale, SmallTile, SmallSchedule,
                   Shape<_16, _32>>(plan, input, weight, input_scales,
                                    weight_scales, bias, output);
  }
  return execute<Scale, WithBias, TensorScale, LargeTile, LargeSchedule,
                 cutlass::epilogue::collective::EpilogueTileAuto>(
      plan, input, weight, input_scales, weight_scales, bias, output);
}

template <typename Scale, bool WithBias>
int workspace_scale(const Plan& plan, size_t* bytes) {
  return plan.tensor_scale ? workspace_shape<Scale, WithBias, true>(plan, bytes)
                           : workspace_shape<Scale, WithBias, false>(plan, bytes);
}

template <typename Scale, bool WithBias>
int execute_scale(const Plan& plan, const void* input, const void* weight,
                  const float* input_scales, const void* weight_scales,
                  const void* bias, void* output) {
  return plan.tensor_scale
             ? execute_shape<Scale, WithBias, true>(
                   plan, input, weight, input_scales, weight_scales, bias, output)
             : execute_shape<Scale, WithBias, false>(
                   plan, input, weight, input_scales, weight_scales, bias, output);
}
}  // namespace mircuda::scaled_fp8

extern "C" int mircuda_scaled_fp8_create(
    int m, int n, int k, int scale_type, int weight_scale_type, int has_bias,
    int tile, void* stream, void** output) {
  using namespace mircuda::scaled_fp8;
  if (m <= 0 || n <= 0 || k <= 0 || n % 16 != 0 || k % 16 != 0 ||
      (scale_type != 0 && scale_type != 1) ||
      (weight_scale_type != 0 && weight_scale_type != 1) ||
      (tile != 0 && tile != 1) || stream == nullptr ||
      output == nullptr) return -1;
  auto* plan = new (std::nothrow) Plan{
      m, n, k, scale_type, weight_scale_type == 0, has_bias != 0, tile,
      static_cast<cudaStream_t>(stream),
      nullptr, 0};
  if (plan == nullptr) return -2;
  int status;
  if (scale_type == 0) {
    status = plan->has_bias ? workspace_scale<float, true>(*plan, &plan->workspace_bytes)
                            : workspace_scale<float, false>(*plan, &plan->workspace_bytes);
  } else {
    status = plan->has_bias
                 ? workspace_scale<cutlass::bfloat16_t, true>(*plan, &plan->workspace_bytes)
                 : workspace_scale<cutlass::bfloat16_t, false>(*plan, &plan->workspace_bytes);
  }
  if (status == 0 && plan->workspace_bytes > 0) {
    status = static_cast<int>(cudaMallocAsync(
        &plan->workspace, plan->workspace_bytes, plan->stream));
  }
  if (status != 0) {
    delete plan;
    return status;
  }
  *output = plan;
  return 0;
}

extern "C" size_t mircuda_scaled_fp8_workspace_bytes(const void* raw) {
  const auto* plan = static_cast<const mircuda::scaled_fp8::Plan*>(raw);
  return plan == nullptr ? 0 : plan->workspace_bytes;
}

extern "C" int mircuda_scaled_fp8_execute(
    const void* raw, void* stream, const void* input, const void* weight,
    const float* input_scales, const void* weight_scales, const void* bias,
    void* output) {
  using namespace mircuda::scaled_fp8;
  if (raw == nullptr || stream == nullptr || input == nullptr ||
      weight == nullptr || input_scales == nullptr || weight_scales == nullptr ||
      output == nullptr) return -1;
  const auto* plan = static_cast<const Plan*>(raw);
  if (static_cast<cudaStream_t>(stream) != plan->stream ||
      (plan->has_bias && bias == nullptr)) return -1;
  if (plan->scale_type == 0) {
    return plan->has_bias
               ? execute_scale<float, true>(*plan, input, weight, input_scales,
                                            weight_scales, bias, output)
               : execute_scale<float, false>(*plan, input, weight, input_scales,
                                             weight_scales, bias, output);
  }
  return plan->has_bias
             ? execute_scale<cutlass::bfloat16_t, true>(
                   *plan, input, weight, input_scales, weight_scales, bias, output)
             : execute_scale<cutlass::bfloat16_t, false>(
                   *plan, input, weight, input_scales, weight_scales, bias, output);
}

extern "C" void mircuda_scaled_fp8_destroy(void* raw) {
  auto* plan = static_cast<mircuda::scaled_fp8::Plan*>(raw);
  if (plan == nullptr) return;
  if (plan->workspace != nullptr) cudaFreeAsync(plan->workspace, plan->stream);
  delete plan;
}
