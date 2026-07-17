#include "grouped_fp4_sm120.cuh"

#include <new>

namespace mircuda::grouped_fp4 {

template <typename T>
int allocate(Plan* plan, T** pointer, size_t count) {
  return static_cast<int>(
      cudaMallocAsync(pointer, count * sizeof(T), plan->stream));
}

template <typename T>
void release(Plan* plan, T* pointer) {
  if (pointer != nullptr) cudaFreeAsync(pointer, plan->stream);
}

__global__ void initialize_metadata(Plan plan) {
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= plan.groups) return;
  plan.problems[group * 3] = plan.m;
  plan.problems[group * 3 + 1] = plan.n;
  plan.problems[group * 3 + 2] = plan.k;
  plan.a_strides[group] = plan.k;
  plan.b_strides[group] = plan.k;
  plan.c_strides[group] = plan.n;
}

__global__ void setup(
    Plan plan, const ElementType* a, const ElementSF* a_scales,
    const ElementType* b, const ElementSF* b_scales, const float* alphas,
  const unsigned int* selected, ElementC* c) {
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= plan.groups) return;
  const unsigned int expert = selected[group];
  if (expert >= static_cast<unsigned int>(plan.experts)) return;
  const int packed_k = plan.k / 2;
  const int a_scale_stride = ((plan.m + 127) / 128) * 128 *
                             ((plan.k / 16 + 3) / 4) * 4;
  const int b_scale_stride = ((plan.n + 127) / 128) * 128 *
                             ((plan.k / 16 + 3) / 4) * 4;
  const int input_group = plan.broadcast_input ? 0 : group;
  plan.a_ptrs[group] =
      const_cast<ElementType*>(a) + input_group * plan.m * packed_k;
  plan.b_ptrs[group] =
      const_cast<ElementType*>(b) + expert * plan.n * packed_k;
  plan.c_ptrs[group] = c + group * plan.m * plan.n;
  plan.a_scale_ptrs[group] =
      const_cast<ElementSF*>(a_scales) + input_group * a_scale_stride;
  plan.b_scale_ptrs[group] =
      const_cast<ElementSF*>(b_scales) + expert * b_scale_stride;
  plan.alpha_ptrs[group] = const_cast<float*>(alphas) + expert;
  plan.a_layouts[group] = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(plan.m, plan.n, plan.k, 1));
  plan.b_layouts[group] = ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(plan.m, plan.n, plan.k, 1));
}

GemmKernel::Arguments arguments(Plan* plan) {
  typename GemmKernel::MainloopArguments mainloop{
      const_cast<const ElementType**>(plan->a_ptrs),
      reinterpret_cast<StrideA*>(plan->a_strides),
      const_cast<const ElementType**>(plan->b_ptrs),
      reinterpret_cast<StrideB*>(plan->b_strides),
      const_cast<const ElementSF**>(plan->a_scale_ptrs), plan->a_layouts,
      const_cast<const ElementSF**>(plan->b_scale_ptrs), plan->b_layouts};
  typename GemmKernel::EpilogueArguments epilogue{
      {}, nullptr, reinterpret_cast<StrideC*>(plan->c_strides), plan->c_ptrs,
      reinterpret_cast<StrideC*>(plan->c_strides)};
  epilogue.thread.alpha_ptr_array = plan->alpha_ptrs;
  epilogue.thread.dAlpha = {_0{}, _0{}, 1};
  epilogue.thread.beta = 0.0f;
  cutlass::KernelHardwareInfo hardware;
  hardware.device_id = plan->device_id;
  hardware.sm_count = plan->sm_count;
  typename GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order =
      cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
  return {cutlass::gemm::GemmUniversalMode::kGrouped,
          {plan->groups, reinterpret_cast<UnderlyingShape*>(plan->problems),
           nullptr}, mainloop, epilogue,
          hardware, scheduler};
}

int allocate_plan(Plan* plan) {
  const size_t groups = static_cast<size_t>(plan->groups);
  int status = allocate(plan, &plan->problems, groups * 3);
  if (status == 0) status = allocate(plan, &plan->a_ptrs, groups);
  if (status == 0) status = allocate(plan, &plan->b_ptrs, groups);
  if (status == 0) status = allocate(plan, &plan->c_ptrs, groups);
  if (status == 0) status = allocate(plan, &plan->a_scale_ptrs, groups);
  if (status == 0) status = allocate(plan, &plan->b_scale_ptrs, groups);
  if (status == 0) status = allocate(plan, &plan->alpha_ptrs, groups);
  if (status == 0) status = allocate(plan, &plan->a_strides, groups);
  if (status == 0) status = allocate(plan, &plan->b_strides, groups);
  if (status == 0) status = allocate(plan, &plan->c_strides, groups);
  if (status == 0) status = allocate(plan, &plan->a_layouts, groups);
  if (status == 0) status = allocate(plan, &plan->b_layouts, groups);
  if (status != 0) return status;
  constexpr int threads = 256;
  const int blocks = (plan->groups + threads - 1) / threads;
  initialize_metadata<<<blocks, threads, 0, plan->stream>>>(*plan);
  status = static_cast<int>(cudaPeekAtLastError());
  if (status != 0) return status;
  Gemm gemm;
  auto args = arguments(plan);
  const auto implementation = Gemm::can_implement(args);
  if (implementation != cutlass::Status::kSuccess) {
    return 100 + static_cast<int>(implementation);
  }
  plan->workspace_bytes = Gemm::get_workspace_size(args);
  if (plan->workspace_bytes > 0) {
    status = static_cast<int>(
        cudaMallocAsync(&plan->workspace, plan->workspace_bytes, plan->stream));
    if (status != 0) return status;
  }
  plan->gemm = new (std::nothrow) Gemm{};
  if (plan->gemm == nullptr) return -2;
  const auto initialized =
      plan->gemm->initialize(args, plan->workspace, plan->stream);
  return initialized == cutlass::Status::kSuccess
             ? 0
             : 100 + static_cast<int>(initialized);
}

void release_plan(Plan* plan) {
  release(plan, plan->problems);
  release(plan, plan->a_ptrs);
  release(plan, plan->b_ptrs);
  release(plan, plan->c_ptrs);
  release(plan, plan->a_scale_ptrs);
  release(plan, plan->b_scale_ptrs);
  release(plan, plan->alpha_ptrs);
  release(plan, plan->a_strides);
  release(plan, plan->b_strides);
  release(plan, plan->c_strides);
  release(plan, plan->a_layouts);
  release(plan, plan->b_layouts);
  release(plan, static_cast<unsigned char*>(plan->workspace));
  delete plan->gemm;
}

}  // namespace mircuda::grouped_fp4

extern "C" int mircuda_grouped_fp4_create(int groups, int experts, int m,
                                           int n, int k, bool broadcast_input, void* stream,
                                           void** output) {
  using namespace mircuda::grouped_fp4;
  if (groups <= 0 || experts <= 0 || m <= 0 || n <= 0 || k <= 0 ||
      k % 64 != 0 ||
      stream == nullptr || output == nullptr) return -1;
  auto* plan = new (std::nothrow) Plan{};
  if (plan == nullptr) return -2;
  plan->groups = groups;
  plan->experts = experts;
  plan->m = m;
  plan->n = n;
  plan->k = k;
  plan->broadcast_input = broadcast_input;
  plan->stream = static_cast<cudaStream_t>(stream);
  int status = static_cast<int>(cudaGetDevice(&plan->device_id));
  if (status == 0) status = static_cast<int>(cudaDeviceGetAttribute(
      &plan->sm_count, cudaDevAttrMultiProcessorCount, plan->device_id));
  if (status != 0) {
    delete plan;
    return status;
  }
  status = allocate_plan(plan);
  if (status != 0) {
    release_plan(plan);
    delete plan;
    return status;
  }
  *output = plan;
  return 0;
}

extern "C" int mircuda_grouped_fp4_execute(
    void* raw, void* stream, const void* a, const void* a_scales,
    const void* b, const void* b_scales, const void* alphas,
    const unsigned int* selected, void* c) {
  using namespace mircuda::grouped_fp4;
  if (raw == nullptr || stream == nullptr) return -1;
  auto* plan = static_cast<Plan*>(raw);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (cuda_stream != plan->stream) return -1;
  constexpr int threads = 256;
  const int blocks = (plan->groups + threads - 1) / threads;
  setup<<<blocks, threads, 0, cuda_stream>>>(
      *plan, static_cast<const ElementType*>(a),
      static_cast<const ElementSF*>(a_scales),
      static_cast<const ElementType*>(b),
      static_cast<const ElementSF*>(b_scales),
      static_cast<const float*>(alphas), selected, static_cast<ElementC*>(c));
  const int setup_status = static_cast<int>(cudaPeekAtLastError());
  if (setup_status != 0) return setup_status;
  auto status = plan->gemm->run(cuda_stream);
  if (status != cutlass::Status::kSuccess) return 100 + static_cast<int>(status);
  return static_cast<int>(cudaPeekAtLastError());
}

extern "C" void mircuda_grouped_fp4_destroy(void* raw) {
  auto* plan = static_cast<mircuda::grouped_fp4::Plan*>(raw);
  if (plan == nullptr) return;
  mircuda::grouped_fp4::release_plan(plan);
  delete plan;
}
