#include <cuda_runtime_api.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <new>

namespace mircuda::fp4 {

using namespace cute;
using ElementType = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementA = cutlass::nv_float4_t<ElementType>;
using ElementB = cutlass::nv_float4_t<ElementType>;
using ElementC = cutlass::bfloat16_t;
using Accumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using Arch = cutlass::arch::Sm120;
using Operator = cutlass::arch::OpClassBlockScaledTensorOp;
using Tile = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;
using Fusion = cutlass::epilogue::fusion::LinearCombination<
    ElementC, Accumulator, ElementC, Accumulator>;
constexpr int AlignmentAB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, Operator, Tile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, Accumulator, Accumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, Operator, ElementA, LayoutA, AlignmentAB, ElementB, LayoutB,
    AlignmentAB, Accumulator, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = GemmKernel::StrideA;
using StrideB = GemmKernel::StrideB;
using StrideC = GemmKernel::StrideC;
using LayoutSFA = Mainloop::LayoutSFA;
using LayoutSFB = Mainloop::LayoutSFB;
using ScaleConfig = Mainloop::Sm1xxBlkScaledConfig;

struct Plan {
  int m;
  int n;
  int k;
  int device_id;
  int sm_count;
  cudaStream_t stream;
  Gemm* gemm;
  bool initialized;
  void* workspace;
  size_t workspace_bytes;
};

auto arguments(const Plan& plan, const void* a, const void* a_scales,
               const void* b, const void* b_scales, void* c, float alpha) {
  auto stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, {plan.m, plan.k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, {plan.n, plan.k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(
      StrideC{}, {plan.m, plan.n, 1});
  auto layout_a = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(plan.m, plan.n, plan.k, 1));
  auto layout_b = ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(plan.m, plan.n, plan.k, 1));
  cutlass::KernelHardwareInfo hardware;
  hardware.device_id = plan.device_id;
  hardware.sm_count = plan.sm_count;
  return Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {plan.m, plan.n, plan.k, 1},
      {static_cast<const ElementType*>(a), stride_a,
       static_cast<const ElementType*>(b), stride_b,
       static_cast<const ElementSF*>(a_scales), layout_a,
       static_cast<const ElementSF*>(b_scales), layout_b},
      {{alpha, 0.0f}, static_cast<const ElementC*>(c), stride_c,
       static_cast<ElementC*>(c), stride_c},
      hardware};
}

int prepare(Plan* plan) {
  auto args = arguments(*plan, nullptr, nullptr, nullptr, nullptr, nullptr, 1.0f);
  const auto implementation = Gemm::can_implement(args);
  if (implementation != cutlass::Status::kSuccess) {
    return 100 + static_cast<int>(implementation);
  }
  plan->workspace_bytes = Gemm::get_workspace_size(args);
  if (plan->workspace_bytes > 0) {
    const int status = static_cast<int>(cudaMallocAsync(
        &plan->workspace, plan->workspace_bytes, plan->stream));
    if (status != 0) return status;
  }
  plan->gemm = new (std::nothrow) Gemm{};
  return plan->gemm == nullptr ? -2 : 0;
}

}  // namespace mircuda::fp4

extern "C" int mircuda_fp4_create(int m, int n, int k, void* stream,
                                    void** output) {
  using namespace mircuda::fp4;
  if (m <= 0 || n <= 0 || k <= 0 || k % 64 != 0 || stream == nullptr ||
      output == nullptr) {
    return -1;
  }
  auto* plan = new (std::nothrow) Plan{m, n, k, 0, 0,
                                      static_cast<cudaStream_t>(stream),
                                      nullptr, false, nullptr, 0};
  if (plan == nullptr) return -2;
  int status = static_cast<int>(cudaGetDevice(&plan->device_id));
  if (status == 0) {
    status = static_cast<int>(cudaDeviceGetAttribute(
        &plan->sm_count, cudaDevAttrMultiProcessorCount, plan->device_id));
  }
  if (status == 0) status = prepare(plan);
  if (status != 0) {
    if (plan->workspace != nullptr) cudaFreeAsync(plan->workspace, plan->stream);
    delete plan;
    return status;
  }
  *output = plan;
  return 0;
}

extern "C" size_t mircuda_fp4_workspace_bytes(const void* raw) {
  const auto* plan = static_cast<const mircuda::fp4::Plan*>(raw);
  return plan == nullptr ? 0 : plan->workspace_bytes;
}

extern "C" int mircuda_fp4_execute(
    void* raw, void* stream, const void* a, const void* a_scales,
    const void* b, const void* b_scales, void* c, float alpha) {
  using namespace mircuda::fp4;
  if (raw == nullptr || stream == nullptr || a == nullptr ||
      a_scales == nullptr || b == nullptr || b_scales == nullptr ||
      c == nullptr) {
    return -1;
  }
  auto* plan = static_cast<Plan*>(raw);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (cuda_stream != plan->stream) return -1;
  auto args = arguments(*plan, a, a_scales, b, b_scales, c, alpha);
  auto status = plan->initialized
                    ? plan->gemm->update(args, plan->workspace)
                    : plan->gemm->initialize(args, plan->workspace, cuda_stream);
  if (status != cutlass::Status::kSuccess) return 100 + static_cast<int>(status);
  plan->initialized = true;
  status = plan->gemm->run(cuda_stream);
  if (status != cutlass::Status::kSuccess) return 100 + static_cast<int>(status);
  return static_cast<int>(cudaPeekAtLastError());
}

extern "C" void mircuda_fp4_destroy(void* raw) {
  auto* plan = static_cast<mircuda::fp4::Plan*>(raw);
  if (plan == nullptr) return;
  delete plan->gemm;
  if (plan->workspace != nullptr) cudaFreeAsync(plan->workspace, plan->stream);
  delete plan;
}
