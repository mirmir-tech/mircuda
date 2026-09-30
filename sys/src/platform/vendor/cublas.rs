use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::CudaStream;

use super::super::{
    driver::{Context, Stream},
    memory::{DeviceBuffer, ensure_stream},
};
use crate::{Error, Result};

unsafe extern "C" {
    fn mircuda_cublas_dense_create(
        m: i32,
        n: i32,
        k: i32,
        data_type: i32,
        stream: *mut c_void,
        output: *mut *mut c_void,
    ) -> i32;
    fn mircuda_cublas_dense_execute(
        plan: *mut c_void,
        stream: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        alpha: f32,
        beta: f32,
    ) -> i32;
    fn mircuda_cublas_dense_destroy(plan: *mut c_void);
}

/// Element type of every operand of a classic cuBLAS dense plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CublasDataType {
    Bf16,
    F32,
}

impl CublasDataType {
    pub(super) const fn native(self) -> i32 {
        match self {
            Self::Bf16 => 0,
            Self::F32 => 1,
        }
    }

    pub(super) const fn bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CublasDenseSpec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub data_type: CublasDataType,
}

#[derive(Debug)]
pub struct CublasDensePlan {
    raw: NonNull<c_void>,
    stream: Arc<CudaStream>,
    spec: CublasDenseSpec,
}

// SAFETY: the plan retains its CUDA stream, binds its context before native
// use or destruction, and execution requires exclusive mutable access.
unsafe impl Send for CublasDensePlan {}

impl Context {
    pub fn create_cublas_dense_plan(
        &self,
        stream: &Stream,
        spec: CublasDenseSpec,
    ) -> Result<CublasDensePlan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        self.inner.bind_to_thread()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: output is writable and dimensions are checked by the safe caller.
        let status = unsafe {
            mircuda_cublas_dense_create(
                i32::try_from(spec.m)?,
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
                spec.data_type.native(),
                stream.inner.cu_stream().cast(),
                &raw mut raw,
            )
        };
        check(status)?;
        Ok(CublasDensePlan {
            raw: NonNull::new(raw).ok_or(Error::NullAllocation)?,
            stream: stream.inner.clone(),
            spec,
        })
    }
}

impl CublasDensePlan {
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
        // SAFETY: buffers share the retained stream and satisfy the fixed shape and type.
        check(unsafe {
            mircuda_cublas_dense_execute(
                self.raw.as_ptr(),
                stream.inner.cu_stream().cast(),
                a.pointer() as *const c_void,
                b.pointer() as *const c_void,
                c.pointer() as *mut c_void,
                alpha,
                beta,
            )
        })
    }
}

impl Drop for CublasDensePlan {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        // SAFETY: this is the sole owner of the native plan.
        unsafe { mircuda_cublas_dense_destroy(self.raw.as_ptr()) };
    }
}

fn validate_sizes(
    spec: CublasDenseSpec,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    c: &DeviceBuffer,
) -> Result<()> {
    let bytes = spec.data_type.bytes();
    let valid = checked_bytes(spec.m, spec.k, bytes) == Some(a.bytes())
        && checked_bytes(spec.n, spec.k, bytes) == Some(b.bytes())
        && checked_bytes(spec.m, spec.n, bytes) == Some(c.bytes());
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
        Err(Error::Cublas(status))
    }
}
