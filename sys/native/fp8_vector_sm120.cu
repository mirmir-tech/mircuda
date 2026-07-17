#include <cuda_runtime_api.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <new>

namespace mircuda::fp8_vector {
using namespace cute;
using Element = cutlass::float_e4m3_t;
using Output = cutlass::bfloat16_t;
using Accumulator = float;
using Tile = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;
using ScaleConfig = cutlass::detail::Sm1xxBlockwiseScaleConfig<1, 128, 128>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
using Fusion = cutlass::epilogue::fusion::LinearCombination<
    Output, Accumulator, void, Accumulator>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Tile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, Accumulator, Accumulator,
    void, cutlass::layout::RowMajor, 0, Output, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    Element, cute::tuple<cutlass::layout::RowMajor, LayoutSFA>, 128,
    Element, cute::tuple<cutlass::layout::ColumnMajor, LayoutSFB>, 128,
    Accumulator, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, Mainloop, Epilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using StrideA = Kernel::StrideA;
using StrideB = Kernel::StrideB;
using StrideD = Kernel::StrideD;

struct Plan {
  int n;
  int k;
  cudaStream_t stream;
  Gemm* gemm;
  bool initialized;
  void* workspace;
  size_t workspace_bytes;
};

auto arguments(const Plan& plan, const void* input, const void* input_scales,
               const void* weight, const void* weight_scales, void* output) {
  auto shape = cute::make_shape(1, plan.n, plan.k, 1);
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {1, plan.k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {plan.n, plan.k, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {1, plan.n, 1});
  auto layout_a = ScaleConfig::tile_atom_to_shape_SFA(shape);
  auto layout_b = ScaleConfig::tile_atom_to_shape_SFB(shape);
  auto arguments = Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, shape,
      {static_cast<const Element*>(input), stride_a,
       static_cast<const Element*>(weight), stride_b,
       static_cast<const Accumulator*>(input_scales), layout_a,
       static_cast<const Accumulator*>(weight_scales), layout_b},
      {{}, nullptr, stride_d, static_cast<Output*>(output), stride_d}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;
  return arguments;
}

int prepare(Plan* plan) {
  auto args = arguments(*plan, nullptr, nullptr, nullptr, nullptr, nullptr);
  auto status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) return 100 + static_cast<int>(status);
  plan->workspace_bytes = Gemm::get_workspace_size(args);
  if (plan->workspace_bytes > 0) {
    int error = static_cast<int>(cudaMallocAsync(
        &plan->workspace, plan->workspace_bytes, plan->stream));
    if (error != 0) return error;
  }
  plan->gemm = new (std::nothrow) Gemm{};
  return plan->gemm == nullptr ? -2 : 0;
}
}  // namespace mircuda::fp8_vector

extern "C" int mircuda_fp8_vector_create(int n, int k, void* stream,
                                            void** output) {
  using namespace mircuda::fp8_vector;
  if (n <= 0 || k <= 0 || n % 128 != 0 || k % 128 != 0 ||
      stream == nullptr || output == nullptr) return -1;
  auto* plan = new (std::nothrow) Plan{
      n, k, static_cast<cudaStream_t>(stream), nullptr, false, nullptr, 0};
  if (plan == nullptr) return -2;
  int status = prepare(plan);
  if (status != 0) {
    if (plan->workspace != nullptr) cudaFreeAsync(plan->workspace, plan->stream);
    delete plan;
    return status;
  }
  *output = plan;
  return 0;
}

extern "C" size_t mircuda_fp8_vector_workspace_bytes(const void* raw) {
  auto* plan = static_cast<const mircuda::fp8_vector::Plan*>(raw);
  return plan == nullptr ? 0 : plan->workspace_bytes;
}

extern "C" int mircuda_fp8_vector_execute(
    void* raw, void* stream, const void* input, const void* input_scales,
    const void* weight, const void* weight_scales, void* output) {
  using namespace mircuda::fp8_vector;
  if (raw == nullptr || stream == nullptr || input == nullptr ||
      input_scales == nullptr || weight == nullptr || weight_scales == nullptr ||
      output == nullptr) return -1;
  auto* plan = static_cast<Plan*>(raw);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (cuda_stream != plan->stream) return -1;
  auto args = arguments(*plan, input, input_scales, weight, weight_scales, output);
  auto status = plan->initialized
      ? plan->gemm->update(args, plan->workspace)
      : plan->gemm->initialize(args, plan->workspace, cuda_stream);
  if (status != cutlass::Status::kSuccess) return 100 + static_cast<int>(status);
  plan->initialized = true;
  status = plan->gemm->run(cuda_stream);
  if (status != cutlass::Status::kSuccess) return 100 + static_cast<int>(status);
  return static_cast<int>(cudaPeekAtLastError());
}

extern "C" void mircuda_fp8_vector_destroy(void* raw) {
  auto* plan = static_cast<mircuda::fp8_vector::Plan*>(raw);
  if (plan == nullptr) return;
  delete plan->gemm;
  if (plan->workspace != nullptr) cudaFreeAsync(plan->workspace, plan->stream);
  delete plan;
}
