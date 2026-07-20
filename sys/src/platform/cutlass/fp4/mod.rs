use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::CudaStream;

use super::super::{
    driver::{Context, Stream},
    memory::{DeviceBuffer, ensure_stream},
};
use crate::{Error, Result};

mod vector;

pub use vector::{BlockScaledFp4VectorPlan, BlockScaledFp4VectorSpec};

unsafe extern "C" {
    fn mircuda_fp4_create(
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
        output: *mut *mut c_void,
    ) -> i32;
    fn mircuda_fp4_workspace_bytes(plan: *const c_void) -> usize;
    fn mircuda_fp4_execute(
        plan: *mut c_void,
        stream: *mut c_void,
        a: *const c_void,
        a_scales: *const c_void,
        b: *const c_void,
        b_scales: *const c_void,
        c: *mut c_void,
        alpha: f32,
    ) -> i32;
    fn mircuda_fp4_destroy(plan: *mut c_void);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockScaledFp4Spec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug)]
pub struct BlockScaledFp4Plan {
    raw: NonNull<c_void>,
    stream: Arc<CudaStream>,
    spec: BlockScaledFp4Spec,
}

// SAFETY: the plan retains its CUDA stream, binds the context before native
// use or destruction, and execution requires exclusive mutable access.
unsafe impl Send for BlockScaledFp4Plan {}

impl Context {
    pub fn create_block_scaled_fp4_plan(
        &self,
        stream: &Stream,
        spec: BlockScaledFp4Spec,
    ) -> Result<BlockScaledFp4Plan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        self.inner.bind_to_thread()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: output is writable and dimensions were validated by the safe caller.
        let status = unsafe {
            mircuda_fp4_create(
                i32::try_from(spec.m)?,
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
                stream.inner.cu_stream().cast(),
                &raw mut raw,
            )
        };
        check(status)?;
        Ok(BlockScaledFp4Plan {
            raw: NonNull::new(raw).ok_or(Error::NullAllocation)?,
            stream: stream.inner.clone(),
            spec,
        })
    }
}

impl BlockScaledFp4Plan {
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        // SAFETY: raw remains live for the lifetime of this plan.
        unsafe { mircuda_fp4_workspace_bytes(self.raw.as_ptr()) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        stream: &Stream,
        a: &DeviceBuffer,
        a_scales: &DeviceBuffer,
        b: &DeviceBuffer,
        b_scales: &DeviceBuffer,
        c: &DeviceBuffer,
        alpha: f32,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        for buffer in [a, a_scales, b, b_scales, c] {
            ensure_stream(buffer, stream)?;
        }
        validate_sizes(self.spec, a, a_scales, b, b_scales, c)?;
        self.stream.context().bind_to_thread()?;
        // SAFETY: fixed shape, buffer sizes, context, and retained stream were checked.
        let status = unsafe {
            mircuda_fp4_execute(
                self.raw.as_ptr(),
                stream.inner.cu_stream().cast(),
                a.pointer() as *const c_void,
                a_scales.pointer() as *const c_void,
                b.pointer() as *const c_void,
                b_scales.pointer() as *const c_void,
                c.pointer() as *mut c_void,
                alpha,
            )
        };
        check(status)
    }
}

impl Drop for BlockScaledFp4Plan {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        // SAFETY: this is the sole owner of the native plan.
        unsafe { mircuda_fp4_destroy(self.raw.as_ptr()) };
    }
}

fn validate_sizes(
    spec: BlockScaledFp4Spec,
    a: &DeviceBuffer,
    a_scales: &DeviceBuffer,
    b: &DeviceBuffer,
    b_scales: &DeviceBuffer,
    c: &DeviceBuffer,
) -> Result<()> {
    let valid = packed_bytes(spec.m, spec.k) == Some(a.bytes())
        && scale_bytes(spec.m, spec.k) == Some(a_scales.bytes())
        && packed_bytes(spec.n, spec.k) == Some(b.bytes())
        && scale_bytes(spec.n, spec.k) == Some(b_scales.bytes())
        && output_bytes(spec.m, spec.n) == Some(c.bytes());
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidMatmulBuffer)
    }
}

fn packed_bytes(rows: usize, columns: usize) -> Option<usize> {
    rows.checked_mul(columns)?.checked_div(2)
}

fn scale_bytes(rows: usize, columns: usize) -> Option<usize> {
    rows.div_ceil(128).checked_mul(columns.checked_div(64)?)?.checked_mul(512)
}

fn output_bytes(rows: usize, columns: usize) -> Option<usize> {
    rows.checked_mul(columns)?.checked_mul(size_of::<u16>())
}

const fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Cutlass(status))
    }
}
