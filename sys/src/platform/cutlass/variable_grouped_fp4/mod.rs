use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::CudaStream;

use super::super::{
    driver::{Context, Stream},
    memory::{DeviceBuffer, ensure_stream},
};
use crate::{Error, Result};

mod paired;

pub use paired::PairedVariableGroupedFp4Plan;

unsafe extern "C" {
    fn mircuda_variable_grouped_fp4_create(
        groups: i32,
        matrices: i32,
        max_m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
        output: *mut *mut c_void,
    ) -> i32;
    fn mircuda_variable_grouped_fp4_execute(
        plan: *mut c_void,
        stream: *mut c_void,
        a: *const c_void,
        a_scales: *const c_void,
        b: *const c_void,
        b_scales: *const c_void,
        alphas: *const c_void,
        indices: *const u32,
        rows: *const u32,
        offsets: *const u32,
        scale_offsets: *const u32,
        c: *mut c_void,
    ) -> i32;
    fn mircuda_variable_grouped_fp4_destroy(plan: *mut c_void);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariableGroupedFp4Spec {
    pub groups: usize,
    pub matrices: usize,
    pub max_m: usize,
    pub n: usize,
    pub k: usize,
    pub capacity_rows: usize,
}

#[derive(Debug)]
pub struct VariableGroupedFp4Plan {
    raw: NonNull<c_void>,
    stream: Arc<CudaStream>,
    spec: VariableGroupedFp4Spec,
}

// SAFETY: the plan retains its CUDA stream, binds the context before native
// use or destruction, and execution requires exclusive mutable access.
unsafe impl Send for VariableGroupedFp4Plan {}

impl Context {
    pub fn create_variable_grouped_fp4_plan(
        &self,
        stream: &Stream,
        spec: VariableGroupedFp4Spec,
    ) -> Result<VariableGroupedFp4Plan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        self.inner.bind_to_thread()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: output is writable and dimensions were checked by the safe facade.
        let status = unsafe {
            mircuda_variable_grouped_fp4_create(
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
        Ok(VariableGroupedFp4Plan {
            raw: NonNull::new(raw).ok_or(Error::NullAllocation)?,
            stream: stream.inner.clone(),
            spec,
        })
    }
}

impl VariableGroupedFp4Plan {
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        stream: &Stream,
        a: &DeviceBuffer,
        a_scales: &DeviceBuffer,
        b: &DeviceBuffer,
        b_scales: &DeviceBuffer,
        alphas: &DeviceBuffer,
        indices: &DeviceBuffer,
        rows: &DeviceBuffer,
        offsets: &DeviceBuffer,
        scale_offsets: &DeviceBuffer,
        c: &DeviceBuffer,
    ) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        for buffer in [a, a_scales, b, b_scales, alphas, indices, rows, offsets, scale_offsets, c] {
            ensure_stream(buffer, stream)?;
        }
        validate_sizes(
            self.spec, a, a_scales, b, b_scales, alphas, indices, rows, offsets, scale_offsets, c,
        )?;
        // SAFETY: exact-size stream-bound buffers outlive the asynchronous operation.
        let status = unsafe {
            mircuda_variable_grouped_fp4_execute(
                self.raw.as_ptr(),
                stream.inner.cu_stream().cast(),
                a.pointer() as *const c_void,
                a_scales.pointer() as *const c_void,
                b.pointer() as *const c_void,
                b_scales.pointer() as *const c_void,
                alphas.pointer() as *const c_void,
                indices.pointer() as *const u32,
                rows.pointer() as *const u32,
                offsets.pointer() as *const u32,
                scale_offsets.pointer() as *const u32,
                c.pointer() as *mut c_void,
            )
        };
        check(status)
    }
}

impl Drop for VariableGroupedFp4Plan {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        // SAFETY: this owns the plan returned by the matching constructor.
        unsafe { mircuda_variable_grouped_fp4_destroy(self.raw.as_ptr()) };
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_sizes(
    spec: VariableGroupedFp4Spec,
    a: &DeviceBuffer,
    a_scales: &DeviceBuffer,
    b: &DeviceBuffer,
    b_scales: &DeviceBuffer,
    alphas: &DeviceBuffer,
    indices: &DeviceBuffer,
    rows: &DeviceBuffer,
    offsets: &DeviceBuffer,
    scale_offsets: &DeviceBuffer,
    c: &DeviceBuffer,
) -> Result<()> {
    let a_scales_bytes = scale_bytes(scale_capacity_rows(spec)?, spec.k)?;
    let b_scales_bytes = scale_bytes(spec.n, spec.k)?
        .checked_mul(spec.matrices)
        .ok_or(Error::InvalidMatmulBuffer)?;
    let metadata_bytes =
        spec.groups.checked_mul(size_of::<u32>()).ok_or(Error::InvalidMatmulBuffer)?;
    let expected = [
        packed_bytes(spec.capacity_rows, spec.k)?,
        a_scales_bytes,
        packed_bytes(product(spec.matrices, spec.n)?, spec.k)?,
        b_scales_bytes,
        product(spec.matrices, size_of::<f32>())?,
        metadata_bytes,
        metadata_bytes,
        metadata_bytes,
        metadata_bytes,
        product(product(spec.capacity_rows, spec.n)?, size_of::<u16>())?,
    ];
    let actual = [
        a.bytes(),
        a_scales.bytes(),
        b.bytes(),
        b_scales.bytes(),
        alphas.bytes(),
        indices.bytes(),
        rows.bytes(),
        offsets.bytes(),
        scale_offsets.bytes(),
        c.bytes(),
    ];
    if expected == actual {
        Ok(())
    } else {
        Err(Error::InvalidMatmulBuffer)
    }
}

fn packed_bytes(rows: usize, k: usize) -> Result<usize> {
    rows.checked_mul(k / 2).ok_or(Error::InvalidMatmulBuffer)
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or(Error::InvalidMatmulBuffer)
}

fn scale_bytes(rows: usize, k: usize) -> Result<usize> {
    rows.div_ceil(128)
        .checked_mul(k.div_ceil(64))
        .and_then(|tiles| tiles.checked_mul(512))
        .ok_or(Error::InvalidMatmulBuffer)
}

fn scale_capacity_rows(spec: VariableGroupedFp4Spec) -> Result<usize> {
    let rows = spec
        .groups
        .checked_mul(127)
        .and_then(|padding| spec.capacity_rows.checked_add(padding))
        .ok_or(Error::InvalidMatmulBuffer)?;
    Ok(rows / 128 * 128)
}

pub(super) const fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Cutlass(status))
    }
}
