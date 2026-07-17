use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::CudaStream;

use super::{
    super::{
        driver::{Context, Stream},
        memory::{DeviceBuffer, ensure_stream},
    },
    DenseMatmulDataType,
};
use crate::{Error, Result};

unsafe extern "C" {
    fn mircuda_dense_vector_create(
        data_type: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
        output: *mut *mut c_void,
    ) -> i32;
    fn mircuda_dense_vector_execute(
        plan: *mut c_void,
        stream: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        alpha: f32,
        beta: f32,
    ) -> i32;
    fn mircuda_dense_vector_destroy(plan: *mut c_void);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseVectorSpec {
    pub n: usize,
    pub k: usize,
    pub data_type: DenseMatmulDataType,
}

#[derive(Debug)]
pub struct DenseVectorPlan {
    raw: NonNull<c_void>,
    stream: Arc<CudaStream>,
    spec: DenseVectorSpec,
}

// SAFETY: the plan retains its CUDA stream, binds the context before native
// use or destruction, and execution requires exclusive mutable access.
unsafe impl Send for DenseVectorPlan {}

impl Context {
    pub fn create_dense_vector_plan(
        &self,
        stream: &Stream,
        spec: DenseVectorSpec,
    ) -> Result<DenseVectorPlan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        self.inner.bind_to_thread()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: output is writable and dimensions are checked by the safe caller.
        let status = unsafe {
            mircuda_dense_vector_create(
                spec.data_type.native(),
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
                stream.inner.cu_stream().cast(),
                &raw mut raw,
            )
        };
        check(status)?;
        Ok(DenseVectorPlan {
            raw: NonNull::new(raw).ok_or(Error::NullAllocation)?,
            stream: stream.inner.clone(),
            spec,
        })
    }
}

impl DenseVectorPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer,
        weight: &DeviceBuffer,
        output: &DeviceBuffer,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        for buffer in [input, weight, output] {
            ensure_stream(buffer, stream)?;
        }
        validate_sizes(self.spec, input, weight, output)?;
        self.stream.context().bind_to_thread()?;
        // SAFETY: buffers share the retained stream and satisfy the fixed shape.
        let status = unsafe {
            mircuda_dense_vector_execute(
                self.raw.as_ptr(),
                stream.inner.cu_stream().cast(),
                input.pointer() as *const c_void,
                weight.pointer() as *const c_void,
                output.pointer() as *mut c_void,
                alpha,
                beta,
            )
        };
        check(status)
    }
}

impl Drop for DenseVectorPlan {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        // SAFETY: this is the sole owner of the native plan.
        unsafe { mircuda_dense_vector_destroy(self.raw.as_ptr()) };
    }
}

fn validate_sizes(
    spec: DenseVectorSpec,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    output: &DeviceBuffer,
) -> Result<()> {
    let bytes = size_of::<u16>();
    let valid = spec.k.checked_mul(bytes) == Some(input.bytes())
        && spec.n.checked_mul(spec.k).and_then(|size| size.checked_mul(bytes))
            == Some(weight.bytes())
        && spec.n.checked_mul(bytes) == Some(output.bytes());
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
