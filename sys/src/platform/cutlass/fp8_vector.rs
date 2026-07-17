use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::CudaStream;

use super::super::{
    driver::{Context, Stream},
    memory::{DeviceBuffer, ensure_stream},
};
use crate::{Error, Result};

unsafe extern "C" {
    fn mircuda_fp8_vector_create(
        n: i32,
        k: i32,
        stream: *mut c_void,
        output: *mut *mut c_void,
    ) -> i32;
    fn mircuda_fp8_vector_workspace_bytes(plan: *const c_void) -> usize;
    fn mircuda_fp8_vector_execute(
        plan: *mut c_void,
        stream: *mut c_void,
        input: *const c_void,
        input_scales: *const c_void,
        weight: *const c_void,
        weight_scales: *const c_void,
        output: *mut c_void,
    ) -> i32;
    fn mircuda_fp8_vector_destroy(plan: *mut c_void);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockwiseFp8VectorSpec {
    pub n: usize,
    pub k: usize,
}

#[derive(Debug)]
pub struct BlockwiseFp8VectorPlan {
    raw: NonNull<c_void>,
    stream: Arc<CudaStream>,
    spec: BlockwiseFp8VectorSpec,
}

// SAFETY: the plan retains its CUDA stream, binds the context before native
// use or destruction, and execution requires exclusive mutable access.
unsafe impl Send for BlockwiseFp8VectorPlan {}

impl Context {
    pub fn create_blockwise_fp8_vector_plan(
        &self,
        stream: &Stream,
        spec: BlockwiseFp8VectorSpec,
    ) -> Result<BlockwiseFp8VectorPlan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        self.inner.bind_to_thread()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: output is writable and the safe facade validated geometry.
        let status = unsafe {
            mircuda_fp8_vector_create(
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
                stream.inner.cu_stream().cast(),
                &raw mut raw,
            )
        };
        check(status)?;
        Ok(BlockwiseFp8VectorPlan {
            raw: NonNull::new(raw).ok_or(Error::NullAllocation)?,
            stream: stream.inner.clone(),
            spec,
        })
    }
}

impl BlockwiseFp8VectorPlan {
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        // SAFETY: raw remains live for the lifetime of this plan.
        unsafe { mircuda_fp8_vector_workspace_bytes(self.raw.as_ptr()) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer,
        input_scales: &DeviceBuffer,
        weight: &DeviceBuffer,
        weight_scales: &DeviceBuffer,
        output: &DeviceBuffer,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        for buffer in [input, input_scales, weight, weight_scales, output] {
            ensure_stream(buffer, stream)?;
        }
        validate(self.spec, input, input_scales, weight, weight_scales, output)?;
        self.stream.context().bind_to_thread()?;
        // SAFETY: shape, byte lengths, stream, and context were checked above.
        check(unsafe {
            mircuda_fp8_vector_execute(
                self.raw.as_ptr(),
                stream.inner.cu_stream().cast(),
                input.pointer() as *const c_void,
                input_scales.pointer() as *const c_void,
                weight.pointer() as *const c_void,
                weight_scales.pointer() as *const c_void,
                output.pointer() as *mut c_void,
            )
        })
    }
}

impl Drop for BlockwiseFp8VectorPlan {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        // SAFETY: this value uniquely owns the native plan.
        unsafe { mircuda_fp8_vector_destroy(self.raw.as_ptr()) };
    }
}

fn validate(
    spec: BlockwiseFp8VectorSpec,
    input: &DeviceBuffer,
    input_scales: &DeviceBuffer,
    weight: &DeviceBuffer,
    weight_scales: &DeviceBuffer,
    output: &DeviceBuffer,
) -> Result<()> {
    let scale_blocks = spec.k / 128;
    let weight_scale_count = (spec.n / 128).checked_mul(scale_blocks);
    let valid = input.bytes() == spec.k
        && input_scales.bytes() == scale_blocks * size_of::<f32>()
        && weight.bytes() == spec.n.checked_mul(spec.k).unwrap_or(0)
        && weight_scales.bytes() == weight_scale_count.unwrap_or(0) * size_of::<f32>()
        && output.bytes() == spec.n * size_of::<u16>();
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidMatmulBuffer)
    }
}

const fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Cutlass(status))
    }
}
