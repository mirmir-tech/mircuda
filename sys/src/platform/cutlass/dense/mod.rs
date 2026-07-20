mod vector;

use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::CudaStream;
pub use vector::{DenseVectorPlan, DenseVectorSpec};

use super::super::{
    driver::{Context, Stream},
    memory::{DeviceBuffer, ensure_stream},
};
use crate::{Error, Result};

unsafe extern "C" {
    fn mircuda_dense_create(
        input_type: i32,
        output_type: i32,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
        output: *mut *mut c_void,
    ) -> i32;
    fn mircuda_dense_workspace_bytes(plan: *const c_void) -> usize;
    fn mircuda_dense_execute(
        plan: *mut c_void,
        stream: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        alpha: f32,
        beta: f32,
    ) -> i32;
    fn mircuda_dense_destroy(plan: *mut c_void);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseMatmulDataType {
    F16,
    Bf16,
    F32,
}

impl DenseMatmulDataType {
    pub(super) const fn native(self) -> i32 {
        match self {
            Self::F16 => 0,
            Self::Bf16 => 1,
            Self::F32 => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseMatmulSpec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub input_type: DenseMatmulDataType,
    pub output_type: DenseMatmulDataType,
}

#[derive(Debug)]
pub struct DenseMatmulPlan {
    raw: NonNull<c_void>,
    stream: Arc<CudaStream>,
    spec: DenseMatmulSpec,
}

// SAFETY: the plan retains its CUDA stream, binds the context before native
// use or destruction, and execution requires exclusive mutable access.
unsafe impl Send for DenseMatmulPlan {}

impl Context {
    pub fn create_dense_matmul_plan(
        &self,
        stream: &Stream,
        spec: DenseMatmulSpec,
    ) -> Result<DenseMatmulPlan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        self.inner.bind_to_thread()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: output is writable and dimensions are checked by the safe caller.
        let status = unsafe {
            mircuda_dense_create(
                spec.input_type.native(),
                spec.output_type.native(),
                i32::try_from(spec.m)?,
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
                stream.inner.cu_stream().cast(),
                &raw mut raw,
            )
        };
        check(status)?;
        Ok(DenseMatmulPlan {
            raw: NonNull::new(raw).ok_or(Error::NullAllocation)?,
            stream: stream.inner.clone(),
            spec,
        })
    }
}

impl DenseMatmulPlan {
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        // SAFETY: raw remains live for the lifetime of this plan.
        unsafe { mircuda_dense_workspace_bytes(self.raw.as_ptr()) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        stream: &Stream,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
        c: &DeviceBuffer,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        for buffer in [a, b, c] {
            ensure_stream(buffer, stream)?;
        }
        validate_sizes(self.spec, a, b, c)?;
        self.stream.context().bind_to_thread()?;
        // SAFETY: all buffers share the retained stream and satisfy the fixed shape.
        let status = unsafe {
            mircuda_dense_execute(
                self.raw.as_ptr(),
                stream.inner.cu_stream().cast(),
                a.pointer() as *const c_void,
                b.pointer() as *const c_void,
                c.pointer() as *mut c_void,
                alpha,
                beta,
            )
        };
        check(status)
    }
}

impl Drop for DenseMatmulPlan {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        // SAFETY: this is the sole owner of the native plan.
        unsafe { mircuda_dense_destroy(self.raw.as_ptr()) };
    }
}

fn validate_sizes(
    spec: DenseMatmulSpec,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    c: &DeviceBuffer,
) -> Result<()> {
    let input_bytes = match spec.input_type {
        DenseMatmulDataType::F16 | DenseMatmulDataType::Bf16 => size_of::<u16>(),
        DenseMatmulDataType::F32 => size_of::<f32>(),
    };
    let output_bytes = match spec.output_type {
        DenseMatmulDataType::F16 | DenseMatmulDataType::Bf16 => size_of::<u16>(),
        DenseMatmulDataType::F32 => size_of::<f32>(),
    };
    let valid = checked_bytes(spec.m, spec.k, input_bytes) == Some(a.bytes())
        && checked_bytes(spec.n, spec.k, input_bytes) == Some(b.bytes())
        && checked_bytes(spec.m, spec.n, output_bytes) == Some(c.bytes());
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidMatmulBuffer)
    }
}

const fn checked_bytes(rows: usize, columns: usize, bytes: usize) -> Option<usize> {
    match rows.checked_mul(columns) {
        Some(elements) => elements.checked_mul(bytes),
        None => None,
    }
}

const fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Cutlass(status))
    }
}
