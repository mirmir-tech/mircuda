#pragma once

#include <cuda_runtime_api.h>

#include <cutlass/arch/arch.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>

namespace mircuda::variable_grouped_bf16 {

using Element = cutlass::bfloat16_t;
using Accumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    Element, 128 / cutlass::sizeof_bits<Element>::value,
    Accumulator, Accumulator>;

enum KernelKind { Short = 0, Bulk = 1 };

template <int Kind>
struct Kernel;

template <>
struct Kernel<Short> {
  using Threadblock = cutlass::gemm::GemmShape<16, 128, 32>;
  using Warp = cutlass::gemm::GemmShape<16, 64, 32>;
};

template <>
struct Kernel<Bulk> {
  using Threadblock = cutlass::gemm::GemmShape<128, 128, 32>;
  using Warp = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <int Kind>
struct Grouped {
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      Element, LayoutA, cutlass::ComplexTransform::kNone, 8,
      Element, LayoutB, cutlass::ComplexTransform::kNone, 8,
      Element, LayoutC, Accumulator, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, typename Kernel<Kind>::Threadblock,
      typename Kernel<Kind>::Warp, cutlass::gemm::GemmShape<16, 8, 16>,
      Epilogue, cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      Kind == Bulk ? 3 : 4,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;
};

struct Plan {
  int groups;
  int max_rows;
  int n;
  int k;
  int kind;
  int threadblocks;
  int device_id;
  int sm_count;
  cudaStream_t stream;
  cutlass::gemm::GemmCoord* problems;
  Element** a_ptrs;
  Element** b_ptrs;
  Element** c_ptrs;
  int64_t* lda;
  int64_t* ldb;
  int64_t* ldc;
  void* gemm;
  void* workspace;
  size_t workspace_bytes;
};

int allocate_plan(Plan* plan);
int create_plan(int groups, int max_rows, int n, int k, void* stream,
                Plan** output);
int execute_plan(Plan* plan, cudaStream_t stream, float beta);
void release_plan(Plan* plan);

}  // namespace mircuda::variable_grouped_bf16

extern "C" {
int mircuda_variable_grouped_bf16_create(
    int groups, int max_rows, int n, int k, void* stream, void** output);
int mircuda_variable_grouped_bf16_execute(
    void* plan, void* stream, const void* input, const void* weights,
    const unsigned int* rows, const unsigned int* offsets, void* output,
    float beta);
void mircuda_variable_grouped_bf16_destroy(void* plan);
}
