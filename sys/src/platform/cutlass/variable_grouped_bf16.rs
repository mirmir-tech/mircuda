use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::CudaStream;

use super::super::{
    driver::{Context, Stream},
    memory::{DeviceBuffer, ensure_stream},
};
use crate::{Error, Result};

unsafe extern "C" {
    fn mircuda_variable_grouped_bf16_create(
        groups: i32,
        max_rows: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
        output: *mut *mut c_void,
    ) -> i32;
    fn mircuda_variable_grouped_bf16_execute(
        plan: *mut c_void,
        stream: *mut c_void,
        input: *const c_void,
        weights: *const c_void,
        rows: *const u32,
        offsets: *const u32,
        output: *mut c_void,
        beta: f32,
    ) -> i32;
    fn mircuda_variable_grouped_bf16_destroy(plan: *mut c_void);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariableGroupedBf16Spec {
    pub groups: usize,
    pub max_rows: usize,
    pub n: usize,
    pub k: usize,
    pub capacity_rows: usize,
}

#[derive(Debug)]
pub struct VariableGroupedBf16Plan {
    raw: NonNull<c_void>,
    stream: Arc<CudaStream>,
    spec: VariableGroupedBf16Spec,
}

// SAFETY: the plan retains its stream and execution requires exclusive access.
unsafe impl Send for VariableGroupedBf16Plan {}

impl Context {
    pub fn create_variable_grouped_bf16_plan(
        &self,
        stream: &Stream,
        spec: VariableGroupedBf16Spec,
    ) -> Result<VariableGroupedBf16Plan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        self.inner.bind_to_thread()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: output is writable and the safe facade validated dimensions.
        let status = unsafe {
            mircuda_variable_grouped_bf16_create(
                i32::try_from(spec.groups)?,
                i32::try_from(spec.max_rows)?,
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
                stream.inner.cu_stream().cast(),
                &raw mut raw,
            )
        };
        check(status)?;
        Ok(VariableGroupedBf16Plan {
            raw: NonNull::new(raw).ok_or(Error::NullAllocation)?,
            stream: stream.inner.clone(),
            spec,
        })
    }
}

impl VariableGroupedBf16Plan {
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer,
        weights: &DeviceBuffer,
        rows: &DeviceBuffer,
        offsets: &DeviceBuffer,
        output: &DeviceBuffer,
        beta: f32,
    ) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        for buffer in [input, weights, rows, offsets, output] {
            ensure_stream(buffer, stream)?;
        }
        validate_sizes(self.spec, input, weights, rows, offsets, output)?;
        // SAFETY: exact-size stream-bound buffers outlive the queued operation.
        let status = unsafe {
            mircuda_variable_grouped_bf16_execute(
                self.raw.as_ptr(),
                stream.inner.cu_stream().cast(),
                input.pointer() as *const c_void,
                weights.pointer() as *const c_void,
                rows.pointer() as *const u32,
                offsets.pointer() as *const u32,
                output.pointer() as *mut c_void,
                beta,
            )
        };
        check(status)
    }
}

impl Drop for VariableGroupedBf16Plan {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        // SAFETY: this owns the plan returned by the matching constructor.
        unsafe { mircuda_variable_grouped_bf16_destroy(self.raw.as_ptr()) };
    }
}

fn validate_sizes(
    spec: VariableGroupedBf16Spec,
    input: &DeviceBuffer,
    weights: &DeviceBuffer,
    rows: &DeviceBuffer,
    offsets: &DeviceBuffer,
    output: &DeviceBuffer,
) -> Result<()> {
    let bf16 = size_of::<u16>();
    let metadata = product(spec.groups, size_of::<u32>())?;
    let expected = [
        product(product(spec.capacity_rows, spec.k)?, bf16)?,
        product(product(product(spec.groups, spec.k)?, spec.n)?, bf16)?,
        metadata,
        metadata,
        product(product(spec.capacity_rows, spec.n)?, bf16)?,
    ];
    let actual = [input.bytes(), weights.bytes(), rows.bytes(), offsets.bytes(), output.bytes()];
    if expected == actual {
        Ok(())
    } else {
        Err(Error::InvalidMatmulBuffer)
    }
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or(Error::InvalidMatmulBuffer)
}

const fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Cutlass(status))
    }
}
