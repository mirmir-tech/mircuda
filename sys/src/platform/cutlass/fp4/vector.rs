use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::CudaStream;

use super::super::super::{
    driver::{Context, Stream},
    memory::{DeviceBuffer, ensure_stream},
};
use crate::{Error, Result};

unsafe extern "C" {
    fn mircuda_fp4_vector_create(
        n: i32,
        k: i32,
        stream: *mut c_void,
        output: *mut *mut c_void,
    ) -> i32;
    fn mircuda_fp4_vector_execute(
        plan: *mut c_void,
        stream: *mut c_void,
        input: *const c_void,
        input_scales: *const c_void,
        weight: *const c_void,
        weight_scales: *const c_void,
        output: *mut c_void,
        alpha: f32,
    ) -> i32;
    fn mircuda_fp4_vector_destroy(plan: *mut c_void);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockScaledFp4VectorSpec {
    pub n: usize,
    pub k: usize,
}

#[derive(Debug)]
pub struct BlockScaledFp4VectorPlan {
    raw: NonNull<c_void>,
    stream: Arc<CudaStream>,
    spec: BlockScaledFp4VectorSpec,
}

// SAFETY: the plan retains its stream and execute requires exclusive access.
unsafe impl Send for BlockScaledFp4VectorPlan {}

impl Context {
    pub fn create_block_scaled_fp4_vector_plan(
        &self,
        stream: &Stream,
        spec: BlockScaledFp4VectorSpec,
    ) -> Result<BlockScaledFp4VectorPlan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        self.inner.bind_to_thread()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: the output pointer is writable and the safe caller validates geometry.
        check(unsafe {
            mircuda_fp4_vector_create(
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
                stream.inner.cu_stream().cast(),
                &raw mut raw,
            )
        })?;
        Ok(BlockScaledFp4VectorPlan {
            raw: NonNull::new(raw).ok_or(Error::NullAllocation)?,
            stream: stream.inner.clone(),
            spec,
        })
    }
}

impl BlockScaledFp4VectorPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer,
        input_scales: &DeviceBuffer,
        weight: &DeviceBuffer,
        weight_scales: &DeviceBuffer,
        output: &DeviceBuffer,
        alpha: f32,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        for buffer in [input, input_scales, weight, weight_scales, output] {
            ensure_stream(buffer, stream)?;
        }
        validate(self.spec, input, input_scales, weight, weight_scales, output)?;
        self.stream.context().bind_to_thread()?;
        // SAFETY: sizes, context, retained stream, and exclusive plan access were checked.
        check(unsafe {
            mircuda_fp4_vector_execute(
                self.raw.as_ptr(),
                stream.inner.cu_stream().cast(),
                input.pointer() as *const c_void,
                input_scales.pointer() as *const c_void,
                weight.pointer() as *const c_void,
                weight_scales.pointer() as *const c_void,
                output.pointer() as *mut c_void,
                alpha,
            )
        })
    }
}

impl Drop for BlockScaledFp4VectorPlan {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        // SAFETY: this value uniquely owns the native plan.
        unsafe { mircuda_fp4_vector_destroy(self.raw.as_ptr()) };
    }
}

fn validate(
    spec: BlockScaledFp4VectorSpec,
    input: &DeviceBuffer,
    input_scales: &DeviceBuffer,
    weight: &DeviceBuffer,
    weight_scales: &DeviceBuffer,
    output: &DeviceBuffer,
) -> Result<()> {
    let valid = input.bytes() == spec.k / 2
        && input_scales.bytes() == scale_bytes(1, spec.k).unwrap_or(0)
        && weight.bytes() == spec.n.checked_mul(spec.k).unwrap_or(0) / 2
        && weight_scales.bytes() == scale_bytes(spec.n, spec.k).unwrap_or(0)
        && output.bytes() == spec.n * size_of::<u16>();
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidMatmulBuffer)
    }
}

fn scale_bytes(rows: usize, columns: usize) -> Option<usize> {
    rows.div_ceil(128).checked_mul(columns.checked_div(64)?)?.checked_mul(512)
}

const fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Cutlass(status))
    }
}
