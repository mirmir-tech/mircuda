#include "variable_grouped_fp4_sm120.cuh"

namespace mircuda::variable_grouped_fp4 {

__global__ void setup_pair(
    Plan plan,
    const ElementType* left_a, const ElementSF* left_a_scales,
    const ElementType* left_b, const ElementSF* left_b_scales,
    const float* left_alphas,
    const ElementType* right_a, const ElementSF* right_a_scales,
    const ElementType* right_b, const ElementSF* right_b_scales,
    const float* right_alphas,
    const unsigned int* indices, const unsigned int* rows,
    const unsigned int* offsets, const unsigned int* scale_offsets,
    ElementC* left_c, ElementC* right_c) {
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= plan.groups) return;
  const int logical = group / 2;
  const bool right = group % 2 != 0;
  const unsigned int matrix = indices[logical];
  if (matrix >= static_cast<unsigned int>(plan.matrices)) return;
  const int row_count = min(static_cast<int>(rows[logical]), plan.max_rows);
  const size_t offset = offsets[logical];
  const int packed_k = plan.k / 2;
  const int b_scale_stride = ((plan.n + 127) / 128) * 128 *
                             ((plan.k / 16 + 3) / 4) * 4;
  const size_t a_scale_offset =
      static_cast<size_t>(scale_offsets[logical] / 128) *
      (plan.k / 64) * 512;
  const ElementType* a = right ? right_a : left_a;
  const ElementSF* a_scales = right ? right_a_scales : left_a_scales;
  const ElementType* b = right ? right_b : left_b;
  const ElementSF* b_scales = right ? right_b_scales : left_b_scales;
  const float* alphas = right ? right_alphas : left_alphas;
  ElementC* c = right ? right_c : left_c;
  plan.rows[group] = row_count;
  plan.a_ptrs[group] = const_cast<ElementType*>(b) + matrix * plan.n * packed_k;
  plan.b_ptrs[group] = const_cast<ElementType*>(a) + offset * packed_k;
  plan.c_ptrs[group] = c + offset * plan.n;
  plan.a_scale_ptrs[group] =
      const_cast<ElementSF*>(b_scales) + matrix * b_scale_stride;
  plan.b_scale_ptrs[group] =
      const_cast<ElementSF*>(a_scales) + a_scale_offset;
  plan.alpha_ptrs[group] = const_cast<float*>(alphas) + matrix;
  plan.a_layouts[group] = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(plan.n, row_count, plan.k, 1));
  plan.b_layouts[group] = ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(plan.n, row_count, plan.k, 1));
}

}  // namespace mircuda::variable_grouped_fp4

extern "C" int mircuda_paired_variable_grouped_fp4_create(
    int groups, int matrices, int max_m, int n, int k, void* stream,
    void** output) {
  using namespace mircuda::variable_grouped_fp4;
  if (groups > INT_MAX / 2) return -1;
  Plan* plan = nullptr;
  const int status = create_plan(
      groups * 2, matrices, max_m, n, k, stream, &plan);
  if (status == 0) *output = plan;
  return status;
}

extern "C" int mircuda_paired_variable_grouped_fp4_execute(
    void* raw, void* stream,
    const void* left_a, const void* left_a_scales,
    const void* left_b, const void* left_b_scales, const void* left_alphas,
    const void* right_a, const void* right_a_scales,
    const void* right_b, const void* right_b_scales, const void* right_alphas,
    const unsigned int* indices, const unsigned int* rows,
    const unsigned int* offsets, const unsigned int* scale_offsets,
    void* left_c, void* right_c) {
  using namespace mircuda::variable_grouped_fp4;
  if (raw == nullptr || stream == nullptr) return -1;
  auto* plan = static_cast<Plan*>(raw);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (cuda_stream != plan->stream || plan->groups % 2 != 0) return -1;
  constexpr int threads = 256;
  const int blocks = (plan->groups + threads - 1) / threads;
  setup_pair<<<blocks, threads, 0, cuda_stream>>>(*plan,
      static_cast<const ElementType*>(left_a),
      static_cast<const ElementSF*>(left_a_scales),
      static_cast<const ElementType*>(left_b),
      static_cast<const ElementSF*>(left_b_scales),
      static_cast<const float*>(left_alphas),
      static_cast<const ElementType*>(right_a),
      static_cast<const ElementSF*>(right_a_scales),
      static_cast<const ElementType*>(right_b),
      static_cast<const ElementSF*>(right_b_scales),
      static_cast<const float*>(right_alphas), indices, rows, offsets, scale_offsets,
      static_cast<ElementC*>(left_c), static_cast<ElementC*>(right_c));
  const int status = static_cast<int>(cudaPeekAtLastError());
  return status == 0 ? execute_plan(plan, cuda_stream) : status;
}

extern "C" void mircuda_paired_variable_grouped_fp4_destroy(void* raw) {
  auto* plan = static_cast<mircuda::variable_grouped_fp4::Plan*>(raw);
  if (plan == nullptr) return;
  mircuda::variable_grouped_fp4::release_plan(plan);
  delete plan;
}
