#pragma once

#include <cuda_runtime_api.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

namespace mircuda::dense {

enum KernelKind {
  ShortTensorCore = 0,
  BulkTensorCore = 1,
  Simt = 2,
  ProfiledWideTensorCore = 3
};

template <typename Element, typename Output, int Kind>
struct Kernel;

template <typename Element, typename Output>
struct TensorCoreBase {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using Accumulator = float;
  using Epilogue = cutlass::epilogue::thread::LinearCombination<
      Output, 128 / cutlass::sizeof_bits<Output>::value,
      Accumulator, Accumulator>;
};

template <typename Element, typename Output>
struct Kernel<Element, Output, ShortTensorCore>
    : TensorCoreBase<Element, Output> {
  using Base = TensorCoreBase<Element, Output>;
  static constexpr int Alignment = 128 / cutlass::sizeof_bits<Element>::value;
  using Gemm = cutlass::gemm::device::Gemm<
      Element, typename Base::LayoutA, Element, typename Base::LayoutB,
      Output, typename Base::LayoutC, typename Base::Accumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<16, 128, 32>,
      cutlass::gemm::GemmShape<16, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>, typename Base::Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4,
      Alignment, Alignment>;
};

template <typename Element, typename Output>
struct Kernel<Element, Output, BulkTensorCore>
    : TensorCoreBase<Element, Output> {
  using Base = TensorCoreBase<Element, Output>;
  static constexpr int Alignment = 128 / cutlass::sizeof_bits<Element>::value;
  using Gemm = cutlass::gemm::device::Gemm<
      Element, typename Base::LayoutA, Element, typename Base::LayoutB,
      Output, typename Base::LayoutC, typename Base::Accumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>, typename Base::Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
      Alignment, Alignment>;
};

template <typename Element, typename Output>
struct Kernel<Element, Output, ProfiledWideTensorCore>
    : TensorCoreBase<Element, Output> {
  using Base = TensorCoreBase<Element, Output>;
  static constexpr int Alignment = 128 / cutlass::sizeof_bits<Element>::value;
  using Gemm = cutlass::gemm::device::Gemm<
      Element, typename Base::LayoutA, Element, typename Base::LayoutB,
      Output, typename Base::LayoutC, typename Base::Accumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<32, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>, typename Base::Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 5,
      Alignment, Alignment>;
};

template <typename Element, typename Output>
struct Kernel<Element, Output, Simt> {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using Accumulator = float;
  using Epilogue = cutlass::epilogue::thread::LinearCombination<
      Output, 1, Accumulator, Accumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      Element, LayoutA, Element, LayoutB, Output, LayoutC, Accumulator,
      cutlass::arch::OpClassSimt, cutlass::arch::Sm50,
      cutlass::gemm::GemmShape<64, 64, 8>,
      cutlass::gemm::GemmShape<32, 32, 8>,
      cutlass::gemm::GemmShape<1, 1, 1>, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1>;
};

struct Plan {
  int input_type;
  int output_type;
  int m;
  int n;
  int k;
  int kind;
  cudaStream_t stream;
  bool initialized;
  void* gemm;
  void* workspace;
  size_t workspace_bytes;
};

}  // namespace mircuda::dense
