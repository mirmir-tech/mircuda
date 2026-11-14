use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::CudaStream;

use super::{VariableGroupedFp4Spec, check, validate_sizes};
use crate::{
    Error, Result,
    platform::{
        driver::{Context, Stream},
        memory::{DeviceBuffer, ensure_stream},
    },
};

unsafe extern "C" {
    fn mircuda_paired_variable_grouped_fp4_create(
        groups: i32,
        matrices: i32,
        max_m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
        output: *mut *mut c_void,
    ) -> i32;
    fn mircuda_paired_variable_grouped_fp4_execute(
        plan: *mut c_void,
        stream: *mut c_void,
        left_a: *const c_void,
        left_a_scales: *const c_void,
        left_b: *const c_void,
        left_b_scales: *const c_void,
        left_alphas: *const c_void,
        right_a: *const c_void,
        right_a_scales: *const c_void,
        right_b: *const c_void,
        right_b_scales: *const c_void,
        right_alphas: *const c_void,
        indices: *const u32,
        rows: *const u32,
        offsets: *const u32,
        scale_offsets: *const u32,
        left_c: *mut c_void,
        right_c: *mut c_void,
    ) -> i32;
    fn mircuda_paired_variable_grouped_fp4_destroy(plan: *mut c_void);
}

#[derive(Debug)]
pub struct PairedVariableGroupedFp4Plan {
    raw: NonNull<c_void>,
    stream: Arc<CudaStream>,
    spec: VariableGroupedFp4Spec,
}

// SAFETY: the plan retains its CUDA stream, binds the context before native
// use or destruction, and execution requires exclusive mutable access.
unsafe impl Send for PairedVariableGroupedFp4Plan {}

impl Context {
    pub fn create_paired_variable_grouped_fp4_plan(
        &self,
        stream: &Stream,
        spec: VariableGroupedFp4Spec,
    ) -> Result<PairedVariableGroupedFp4Plan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        self.inner.bind_to_thread()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: output is writable and the safe facade validated all dimensions.
        let status = unsafe {
            mircuda_paired_variable_grouped_fp4_create(
                i32::try_from(spec.groups)?,
                i32::try_from(spec.matrices)?,
                i32::try_from(spec.max_m)?,
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
                stream.inner.cu_stream().cast(),
                &raw mut raw,
            )
        };
        check(status)?;
        Ok(PairedVariableGroupedFp4Plan {
            raw: NonNull::new(raw).ok_or(Error::NullAllocation)?,
            stream: stream.inner.clone(),
            spec,
        })
    }
}

impl PairedVariableGroupedFp4Plan {
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        stream: &Stream,
        left_a: &DeviceBuffer,
        left_a_scales: &DeviceBuffer,
        left_b: &DeviceBuffer,
        left_b_scales: &DeviceBuffer,
        left_alphas: &DeviceBuffer,
        right_a: &DeviceBuffer,
        right_a_scales: &DeviceBuffer,
        right_b: &DeviceBuffer,
        right_b_scales: &DeviceBuffer,
        right_alphas: &DeviceBuffer,
        indices: &DeviceBuffer,
        rows: &DeviceBuffer,
        offsets: &DeviceBuffer,
        scale_offsets: &DeviceBuffer,
        left_c: &DeviceBuffer,
        right_c: &DeviceBuffer,
    ) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        let buffers = [
            left_a, left_a_scales, left_b, left_b_scales, left_alphas, right_a, right_a_scales,
            right_b, right_b_scales, right_alphas, indices, rows, offsets, scale_offsets, left_c,
            right_c,
        ];
        for buffer in buffers {
            ensure_stream(buffer, stream)?;
        }
        validate_sizes(
            self.spec, left_a, left_a_scales, left_b, left_b_scales, left_alphas, indices, rows,
            offsets, scale_offsets, left_c,
        )?;
        validate_sizes(
            self.spec, right_a, right_a_scales, right_b, right_b_scales, right_alphas, indices,
            rows, offsets, scale_offsets, right_c,
        )?;
        // SAFETY: validated buffers are stream-bound and outlive asynchronous execution.
        check(unsafe {
            mircuda_paired_variable_grouped_fp4_execute(
                self.raw.as_ptr(),
                stream.inner.cu_stream().cast(),
                left_a.pointer() as *const c_void,
                left_a_scales.pointer() as *const c_void,
                left_b.pointer() as *const c_void,
                left_b_scales.pointer() as *const c_void,
                left_alphas.pointer() as *const c_void,
                right_a.pointer() as *const c_void,
                right_a_scales.pointer() as *const c_void,
                right_b.pointer() as *const c_void,
                right_b_scales.pointer() as *const c_void,
                right_alphas.pointer() as *const c_void,
                indices.pointer() as *const u32,
                rows.pointer() as *const u32,
                offsets.pointer() as *const u32,
                scale_offsets.pointer() as *const u32,
                left_c.pointer() as *mut c_void,
                right_c.pointer() as *mut c_void,
            )
        })
    }
}

impl Drop for PairedVariableGroupedFp4Plan {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        // SAFETY: this owns the matching native paired plan.
        unsafe { mircuda_paired_variable_grouped_fp4_destroy(self.raw.as_ptr()) };
    }
}
