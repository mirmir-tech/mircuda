#pragma once

#include <cuda_runtime_api.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

namespace mircuda::scaled_fp8 {
using namespace cute;
using Element = cutlass::float_e4m3_t;
using Output = cutlass::bfloat16_t;
using Accumulator = float;

template <typename Scale, typename Tile, bool WithBias, bool TensorScale>
struct Epilogue {
 private:
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0, Tile, float, float, Stride<_1, _0, _0>, 4>;
  using ScaleB = std::conditional_t<
      TensorScale,
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<Scale>,
      cutlass::epilogue::fusion::Sm90RowBroadcast<
          0, Tile, Scale, Scale, Stride<_0, _1, _0>, 128 / sizeof_bits_v<Scale>>>;
  using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0, Tile, Output, Output, Stride<_0, _1, _0>, 8>;
  using MultiplyB = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using ScaledAccum = cutlass::epilogue::fusion::Sm90EVT<
      MultiplyB, ScaleB, Accum>;
  using MultiplyA = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, Output, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using MultiplyAdd = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, Output, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute = std::conditional_t<
      WithBias,
      cutlass::epilogue::fusion::Sm90EVT<
          MultiplyAdd, ScaleA, ScaledAccum, Bias>,
      cutlass::epilogue::fusion::Sm90EVT<
          MultiplyA, ScaleA, ScaledAccum>>;
  using Arguments = typename EVTCompute::Arguments;

  static Arguments arguments(const float* scale_a, const Scale* scale_b,
                             const Output* bias) {
    typename ScaleA::Arguments a{scale_a};
    typename ScaleB::Arguments b{};
    if constexpr (TensorScale) {
      b.scalar_ptrs[0] = scale_b;
    } else {
      b = typename ScaleB::Arguments{scale_b};
    }
    typename ScaledAccum::Arguments scaled{b, {}, {}};
    if constexpr (WithBias) {
      typename Bias::Arguments bias_args{bias};
      return Arguments{a, scaled, bias_args, {}};
    } else {
      return Arguments{a, scaled, {}};
    }
  }
};

template <typename Scale, bool WithBias, bool TensorScale, typename Tile,
          typename KernelSchedule, typename EpilogueTile>
struct Kernel {
  using Cluster = Shape<_1, _1, _1>;
  using Visitor = Epilogue<Scale, Tile, WithBias, TensorScale>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      EpilogueTile, Accumulator, float, void, cutlass::layout::RowMajor, 8,
      Output, cutlass::layout::RowMajor, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      typename Visitor::EVTCompute>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      Element, cutlass::layout::RowMajor, 16,
      Element, cutlass::layout::ColumnMajor, 16,
      Accumulator, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule, void>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

struct Plan {
  int m;
  int n;
  int k;
  int scale_type;
  bool tensor_scale;
  bool has_bias;
  cudaStream_t stream;
  void* workspace;
  size_t workspace_bytes;
};
}  // namespace mircuda::scaled_fp8
