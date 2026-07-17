#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

namespace mircuda::grouped_fp4 {

using namespace cute;
using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
using ElementType = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementA = cutlass::nv_float4_t<ElementType>;
using ElementB = cutlass::nv_float4_t<ElementType>;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ClusterShape = Shape<_1, _1, _1>;
using MmaTileShape = Shape<_128, _128, _128>;
using Fusion = cutlass::epilogue::fusion::LinearCombination<
    ElementC, ElementAccumulator, ElementC, ElementAccumulator>;

static constexpr int AlignmentAB = 32;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutC*, AlignmentC, ElementC, LayoutC*,
        AlignmentC, cutlass::epilogue::collective::EpilogueScheduleAuto,
        Fusion>::CollectiveOp;
using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentAB, ElementB,
        LayoutB*, AlignmentAB, ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using UnderlyingShape = ProblemShape::UnderlyingProblemShape;
using StrideA = GemmKernel::InternalStrideA;
using StrideB = GemmKernel::InternalStrideB;
using StrideC = GemmKernel::InternalStrideC;
using LayoutSFA = CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = CollectiveMainloop::InternalLayoutSFB;
using ScaleConfig = CollectiveMainloop::Sm1xxBlkScaledConfig;

struct Plan {
  int groups;
  int experts;
  int m;
  int n;
  int k;
  bool broadcast_input;
  int device_id;
  int sm_count;
  cudaStream_t stream;
  int32_t* problems;
  ElementType** a_ptrs;
  ElementType** b_ptrs;
  ElementC** c_ptrs;
  ElementSF** a_scale_ptrs;
  ElementSF** b_scale_ptrs;
  float** alpha_ptrs;
  int64_t* a_strides;
  int64_t* b_strides;
  int64_t* c_strides;
  LayoutSFA* a_layouts;
  LayoutSFB* b_layouts;
  Gemm* gemm;
  void* workspace;
  size_t workspace_bytes;
};

GemmKernel::Arguments arguments(Plan* plan);
int allocate_plan(Plan* plan);
void release_plan(Plan* plan);

}  // namespace mircuda::grouped_fp4

extern "C" {
int mircuda_grouped_fp4_create(int groups, int experts, int m, int n, int k,
                               bool broadcast_input, void* stream,
                               void** output);
int mircuda_grouped_fp4_execute(
    void* plan, void* stream, const void* a, const void* a_scales,
    const void* b, const void* b_scales, const void* alphas,
    const unsigned int* selected, void* c);
void mircuda_grouped_fp4_destroy(void* plan);
}
