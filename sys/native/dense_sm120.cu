#include "dense_sm120.cuh"

#include <new>

namespace mircuda::dense {

template <typename Element, typename Output, int Kind>
auto arguments(const Plan& plan, const void* a, const void* b, void* c,
               float alpha, float beta) {
  using K = Kernel<Element, Output, Kind>;
  return typename K::Gemm::Arguments{
      {plan.m, plan.n, plan.k},
      {static_cast<const Element*>(a), plan.k},
      {static_cast<const Element*>(b), plan.k},
      {static_cast<const Output*>(c), plan.n},
      {static_cast<Output*>(c), plan.n},
      {alpha, beta}};
}

template <typename Element, typename Output, int Kind>
int prepare(Plan* plan) {
  using Gemm = typename Kernel<Element, Output, Kind>::Gemm;
  auto args = arguments<Element, Output, Kind>(
      *plan, nullptr, nullptr, nullptr, 1.0f, 0.0f);
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

template <typename Element, typename Output, int Kind>
int execute(Plan* plan, const void* a, const void* b, void* c, float alpha,
            float beta) {
  using Gemm = typename Kernel<Element, Output, Kind>::Gemm;
  auto args = arguments<Element, Output, Kind>(*plan, a, b, c, alpha, beta);
  auto* gemm = static_cast<Gemm*>(plan->gemm);
  auto status = plan->initialized
                    ? gemm->update(args, plan->workspace)
                    : gemm->initialize(args, plan->workspace, plan->stream);
  if (status != cutlass::Status::kSuccess) return 100 + static_cast<int>(status);
  plan->initialized = true;
  status = gemm->run(plan->stream);
  if (status != cutlass::Status::kSuccess) return 100 + static_cast<int>(status);
  return static_cast<int>(cudaPeekAtLastError());
}

template <typename Element, typename Output, int Kind>
void release(Plan* plan) {
  using Gemm = typename Kernel<Element, Output, Kind>::Gemm;
  delete static_cast<Gemm*>(plan->gemm);
}

template <typename Element, typename Output>
int prepare_selected(Plan* plan) {
  if (plan->kind == ShortTensorCore) {
    return prepare<Element, Output, ShortTensorCore>(plan);
  }
  if (plan->kind == ProfiledWideTensorCore) {
    return prepare<Element, Output, ProfiledWideTensorCore>(plan);
  }
  if (plan->kind == BulkTensorCore) {
    return prepare<Element, Output, BulkTensorCore>(plan);
  }
  return prepare<Element, Output, Simt>(plan);
}

template <typename Element, typename Output>
int execute_selected(Plan* plan, const void* a, const void* b, void* c,
                     float alpha, float beta) {
  if (plan->kind == ShortTensorCore) {
    return execute<Element, Output, ShortTensorCore>(plan, a, b, c, alpha, beta);
  }
  if (plan->kind == ProfiledWideTensorCore) {
    return execute<Element, Output, ProfiledWideTensorCore>(plan, a, b, c, alpha, beta);
  }
  if (plan->kind == BulkTensorCore) {
    return execute<Element, Output, BulkTensorCore>(plan, a, b, c, alpha, beta);
  }
  return execute<Element, Output, Simt>(plan, a, b, c, alpha, beta);
}

template <typename Element, typename Output>
void release_selected(Plan* plan) {
  if (plan->kind == ShortTensorCore) {
    return release<Element, Output, ShortTensorCore>(plan);
  }
  if (plan->kind == ProfiledWideTensorCore) {
    return release<Element, Output, ProfiledWideTensorCore>(plan);
  }
  if (plan->kind == BulkTensorCore) {
    return release<Element, Output, BulkTensorCore>(plan);
  }
  release<Element, Output, Simt>(plan);
}

template <typename Element>
int prepare_output(Plan* plan) {
  return plan->output_type == 2
      ? prepare_selected<Element, float>(plan)
      : prepare_selected<Element, Element>(plan);
}

template <typename Element>
int execute_output(Plan* plan, const void* a, const void* b, void* c,
                   float alpha, float beta) {
  return plan->output_type == 2
      ? execute_selected<Element, float>(plan, a, b, c, alpha, beta)
      : execute_selected<Element, Element>(plan, a, b, c, alpha, beta);
}

template <typename Element>
void release_output(Plan* plan) {
  if (plan->output_type == 2) {
    release_selected<Element, float>(plan);
  } else {
    release_selected<Element, Element>(plan);
  }
}

}  // namespace mircuda::dense

extern "C" int mircuda_dense_create(int input_type, int output_type, int m,
                                      int n, int k, void* stream, void** output) {
  using namespace mircuda::dense;
  if ((input_type != 0 && input_type != 1) ||
      (output_type != input_type && output_type != 2) ||
      m <= 0 || n <= 0 || k <= 0 ||
      stream == nullptr || output == nullptr) return -1;
  const bool aligned = k % 8 == 0 && n % 8 == 0;
  const int kind = !aligned
                       ? Simt
                       : (m > 32 ? BulkTensorCore
                                 : (n >= 8192 ? ProfiledWideTensorCore
                                              : ShortTensorCore));
  auto* plan = new (std::nothrow) Plan{input_type, output_type, m, n, k, kind,
                                      static_cast<cudaStream_t>(stream),
                                      false, nullptr, nullptr, 0};
  if (plan == nullptr) return -2;
  const int status = input_type == 0 ? prepare_output<cutlass::half_t>(plan)
                                     : prepare_output<cutlass::bfloat16_t>(plan);
  if (status != 0) {
    if (plan->workspace != nullptr) cudaFreeAsync(plan->workspace, plan->stream);
    delete plan;
    return status;
  }
  *output = plan;
  return 0;
}

extern "C" size_t mircuda_dense_workspace_bytes(const void* raw) {
  const auto* plan = static_cast<const mircuda::dense::Plan*>(raw);
  return plan == nullptr ? 0 : plan->workspace_bytes;
}

extern "C" int mircuda_dense_execute(void* raw, void* stream, const void* a,
                                       const void* b, void* c, float alpha,
                                       float beta) {
  using namespace mircuda::dense;
  if (raw == nullptr || stream == nullptr || a == nullptr || b == nullptr ||
      c == nullptr) return -1;
  auto* plan = static_cast<Plan*>(raw);
  if (static_cast<cudaStream_t>(stream) != plan->stream) return -1;
  return plan->input_type == 0
             ? execute_output<cutlass::half_t>(plan, a, b, c, alpha, beta)
             : execute_output<cutlass::bfloat16_t>(plan, a, b, c, alpha, beta);
}

extern "C" void mircuda_dense_destroy(void* raw) {
  using namespace mircuda::dense;
  auto* plan = static_cast<Plan*>(raw);
  if (plan == nullptr) return;
  plan->input_type == 0 ? release_output<cutlass::half_t>(plan)
                        : release_output<cutlass::bfloat16_t>(plan);
  if (plan->workspace != nullptr) cudaFreeAsync(plan->workspace, plan->stream);
  delete plan;
}
