#include "variable_grouped_bf16_sm120.cuh"

#include <new>

namespace mircuda::variable_grouped_bf16 {

template <typename T>
int allocate(Plan* plan, T** pointer, size_t count) {
  return static_cast<int>(
      cudaMallocAsync(pointer, count * sizeof(T), plan->stream));
}

template <typename T>
void release(Plan* plan, T* pointer) {
  if (pointer != nullptr) cudaFreeAsync(pointer, plan->stream);
}

__global__ void setup(
    Plan plan, const Element* input, const Element* weights,
    const unsigned int* rows, const unsigned int* offsets, Element* output) {
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= plan.groups) return;
  const int row_count = min(static_cast<int>(rows[group]), plan.max_rows);
  const size_t offset = offsets[group];
  plan.problems[group] = {row_count, plan.n, plan.k};
  plan.a_ptrs[group] = const_cast<Element*>(input) + offset * plan.k;
  plan.b_ptrs[group] =
      const_cast<Element*>(weights) + group * plan.n * plan.k;
  plan.c_ptrs[group] = output + offset * plan.n;
  plan.lda[group] = plan.k;
  plan.ldb[group] = plan.k;
  plan.ldc[group] = plan.n;
}

template <int Kind>
typename Grouped<Kind>::Gemm::Arguments arguments(Plan* plan, float beta) {
  using Gemm = typename Grouped<Kind>::Gemm;
  typename Gemm::EpilogueOutputOp::Params epilogue(1.0f, beta);
  return typename Gemm::Arguments(
      plan->problems, plan->groups, plan->threadblocks, epilogue,
      plan->a_ptrs, plan->b_ptrs, plan->c_ptrs, plan->c_ptrs,
      plan->lda, plan->ldb, plan->ldc, plan->ldc, nullptr);
}

template <int Kind>
int prepare(Plan* plan) {
  using Gemm = typename Grouped<Kind>::Gemm;
  plan->threadblocks = Gemm::sufficient(nullptr, plan->groups, plan->sm_count);
  if (plan->threadblocks <= 0) return -3;
  auto args = arguments<Kind>(plan, 0.0f);
  const auto implementation = Gemm::can_implement(args);
  if (implementation != cutlass::Status::kSuccess) {
    return 100 + static_cast<int>(implementation);
  }
  plan->workspace_bytes = Gemm::get_workspace_size(args);
  if (plan->workspace_bytes > 0) {
    const int status = static_cast<int>(
        cudaMallocAsync(&plan->workspace, plan->workspace_bytes, plan->stream));
    if (status != 0) return status;
  }
  auto* gemm = new (std::nothrow) Gemm{};
  if (gemm == nullptr) return -2;
  plan->gemm = gemm;
  return 0;
}

template <int Kind>
int execute(Plan* plan, cudaStream_t stream, float beta) {
  using Gemm = typename Grouped<Kind>::Gemm;
  auto* gemm = static_cast<Gemm*>(plan->gemm);
  auto status =
      gemm->initialize(arguments<Kind>(plan, beta), plan->workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return 200 + static_cast<int>(status);
  }
  status = gemm->run(stream);
  if (status != cutlass::Status::kSuccess) {
    return 300 + static_cast<int>(status);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

template <int Kind>
void release_gemm(Plan* plan) {
  using Gemm = typename Grouped<Kind>::Gemm;
  delete static_cast<Gemm*>(plan->gemm);
}

int allocate_plan(Plan* plan) {
  const size_t groups = static_cast<size_t>(plan->groups);
  int status = allocate(plan, &plan->problems, groups);
  if (status == 0) status = allocate(plan, &plan->a_ptrs, groups);
  if (status == 0) status = allocate(plan, &plan->b_ptrs, groups);
  if (status == 0) status = allocate(plan, &plan->c_ptrs, groups);
  if (status == 0) status = allocate(plan, &plan->lda, groups);
  if (status == 0) status = allocate(plan, &plan->ldb, groups);
  if (status == 0) status = allocate(plan, &plan->ldc, groups);
  if (status != 0) return status;
  return plan->kind == Short ? prepare<Short>(plan) : prepare<Bulk>(plan);
}

void release_plan(Plan* plan) {
  release(plan, plan->problems);
  release(plan, plan->a_ptrs);
  release(plan, plan->b_ptrs);
  release(plan, plan->c_ptrs);
  release(plan, plan->lda);
  release(plan, plan->ldb);
  release(plan, plan->ldc);
  release(plan, static_cast<unsigned char*>(plan->workspace));
  if (plan->gemm != nullptr) {
    plan->kind == Short ? release_gemm<Short>(plan) : release_gemm<Bulk>(plan);
  }
}

int create_plan(int groups, int max_rows, int n, int k, void* stream,
                Plan** output) {
  if (groups <= 0 || max_rows <= 0 || n <= 0 || k <= 0 ||
      n % 8 != 0 || k % 8 != 0 || stream == nullptr || output == nullptr) {
    return -1;
  }
  auto* plan = new (std::nothrow) Plan{};
  if (plan == nullptr) return -2;
  plan->groups = groups;
  plan->max_rows = max_rows;
  plan->n = n;
  plan->k = k;
  plan->kind = max_rows <= 256 ? Short : Bulk;
  plan->stream = static_cast<cudaStream_t>(stream);
  int status = static_cast<int>(cudaGetDevice(&plan->device_id));
  if (status == 0) status = static_cast<int>(cudaDeviceGetAttribute(
      &plan->sm_count, cudaDevAttrMultiProcessorCount, plan->device_id));
  if (status == 0) status = allocate_plan(plan);
  if (status != 0) {
    release_plan(plan);
    delete plan;
    return status;
  }
  *output = plan;
  return 0;
}

int execute_plan(Plan* plan, cudaStream_t stream, float beta) {
  return plan->kind == Short ? execute<Short>(plan, stream, beta)
                             : execute<Bulk>(plan, stream, beta);
}

}  // namespace mircuda::variable_grouped_bf16

extern "C" int mircuda_variable_grouped_bf16_create(
    int groups, int max_rows, int n, int k, void* stream, void** output) {
  using namespace mircuda::variable_grouped_bf16;
  Plan* plan = nullptr;
  const int status = create_plan(groups, max_rows, n, k, stream, &plan);
  if (status == 0) *output = plan;
  return status;
}

extern "C" int mircuda_variable_grouped_bf16_execute(
    void* raw, void* stream, const void* input, const void* weights,
    const unsigned int* rows, const unsigned int* offsets, void* output,
    float beta) {
  using namespace mircuda::variable_grouped_bf16;
  if (raw == nullptr || stream == nullptr || input == nullptr ||
      weights == nullptr || rows == nullptr || offsets == nullptr ||
      output == nullptr) return -1;
  auto* plan = static_cast<Plan*>(raw);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (cuda_stream != plan->stream) return -1;
  constexpr int threads = 256;
  const int blocks = (plan->groups + threads - 1) / threads;
  setup<<<blocks, threads, 0, cuda_stream>>>(
      *plan, static_cast<const Element*>(input),
      static_cast<const Element*>(weights), rows, offsets,
      static_cast<Element*>(output));
  const int status = static_cast<int>(cudaPeekAtLastError());
  return status == 0 ? execute_plan(plan, cuda_stream, beta) : status;
}

extern "C" void mircuda_variable_grouped_bf16_destroy(void* raw) {
  auto* plan = static_cast<mircuda::variable_grouped_bf16::Plan*>(raw);
  if (plan == nullptr) return;
  mircuda::variable_grouped_bf16::release_plan(plan);
  delete plan;
}
