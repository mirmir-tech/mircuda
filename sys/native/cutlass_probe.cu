#include <cutlass/arch/arch.h>
#include <cutlass/arch/config.h>
#include <cutlass/cutlass.h>
#include <cutlass/version.h>

#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && \
    !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#error "CUTLASS lacks SM12x support"
#endif

#if defined(__CUDA_ARCH__) && !defined(CUTLASS_ARCH_MMA_SM120_ENABLED) && \
    !defined(CUTLASS_ARCH_MMA_SM121_ENABLED)
#error "CUDA target does not enable SM12x matrix instructions"
#endif

extern "C" unsigned int mircuda_cutlass_version() {
  return CUTLASS_MAJOR * 10000u + CUTLASS_MINOR * 100u + CUTLASS_PATCH;
}
