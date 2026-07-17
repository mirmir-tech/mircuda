use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::CudaStream;

use super::super::{
    driver::{Context, Stream},
    memory::{DeviceBuffer, ensure_stream},
};
use crate::{Error, Result};

unsafe extern "C" {
    fn mircuda_grouped_fp4_create(
        groups: i32,
        experts: i32,
        m: i32,
        n: i32,
        k: i32,
        broadcast_input: bool,
        stream: *mut c_void,
        output: *mut *mut c_void,
    ) -> i32;
    fn mircuda_grouped_fp4_execute(
        plan: *mut c_void,
        stream: *mut c_void,
        a: *const c_void,
        a_scales: *const c_void,
        b: *const c_void,
        b_scales: *const c_void,
        alphas: *const c_void,
        selected: *const u32,
        c: *mut c_void,
    ) -> i32;
    fn mircuda_grouped_fp4_destroy(plan: *mut c_void);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexedGroupedFp4Spec {
    pub groups: usize,
    pub matrices: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub broadcast_input: bool,
}

#[derive(Debug)]
pub struct IndexedGroupedFp4Plan {
    raw: NonNull<c_void>,
    stream: Arc<CudaStream>,
    spec: IndexedGroupedFp4Spec,
}

// SAFETY: the plan retains its CUDA stream, binds the context before native
// use or destruction, and execution requires exclusive mutable access.
unsafe impl Send for IndexedGroupedFp4Plan {}

impl Context {
    pub fn create_indexed_grouped_fp4_plan(
        &self,
        stream: &Stream,
        spec: IndexedGroupedFp4Spec,
    ) -> Result<IndexedGroupedFp4Plan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        self.inner.bind_to_thread()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: output is writable and all dimensions were checked by the caller.
        let status = unsafe {
            mircuda_grouped_fp4_create(
                i32::try_from(spec.groups)?,
                i32::try_from(spec.matrices)?,
                i32::try_from(spec.m)?,
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
                spec.broadcast_input,
                stream.inner.cu_stream().cast(),
                &raw mut raw,
            )
        };
        check(status)?;
        Ok(IndexedGroupedFp4Plan {
            raw: NonNull::new(raw).ok_or(Error::NullAllocation)?,
            stream: stream.inner.clone(),
            spec,
        })
    }
}

impl IndexedGroupedFp4Plan {
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
        c: &DeviceBuffer,
    ) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        for buffer in [a, a_scales, b, b_scales, alphas, indices, c] {
            ensure_stream(buffer, stream)?;
        }
        validate_sizes(self.spec, a, a_scales, b, b_scales, alphas, indices, c)?;
        // SAFETY: validated buffers outlive the asynchronous launch and share the stream.
        let status = unsafe {
            mircuda_grouped_fp4_execute(
                self.raw.as_ptr(),
                stream.inner.cu_stream().cast(),
                a.pointer() as *const c_void,
                a_scales.pointer() as *const c_void,
                b.pointer() as *const c_void,
                b_scales.pointer() as *const c_void,
                alphas.pointer() as *const c_void,
                indices.pointer() as *const u32,
                c.pointer() as *mut c_void,
            )
        };
        check(status)
    }
}

impl Drop for IndexedGroupedFp4Plan {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        // SAFETY: this is the sole owner of a plan returned by the matching constructor.
        unsafe { mircuda_grouped_fp4_destroy(self.raw.as_ptr()) };
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_sizes(
    spec: IndexedGroupedFp4Spec,
    a: &DeviceBuffer,
    a_scales: &DeviceBuffer,
    b: &DeviceBuffer,
    b_scales: &DeviceBuffer,
    alphas: &DeviceBuffer,
    indices: &DeviceBuffer,
    c: &DeviceBuffer,
) -> Result<()> {
    let a_scale_stride = spec
        .m
        .div_ceil(128)
        .checked_mul(spec.k.div_ceil(64))
        .and_then(|tiles| tiles.checked_mul(512));
    let b_scale_stride = spec
        .n
        .div_ceil(128)
        .checked_mul(spec.k.div_ceil(64))
        .and_then(|tiles| tiles.checked_mul(512));
    let b_bytes = spec
        .matrices
        .checked_mul(spec.n)
        .and_then(|elements| elements.checked_mul(spec.k / 2))
        .ok_or(Error::InvalidMatmulBuffer)?;
    let c_bytes = spec
        .groups
        .checked_mul(spec.m)
        .and_then(|elements| elements.checked_mul(spec.n))
        .and_then(|elements| elements.checked_mul(size_of::<u16>()))
        .ok_or(Error::InvalidMatmulBuffer)?;
    let input_groups = if spec.broadcast_input {
        1
    } else {
        spec.groups
    };
    let expected = [
        input_groups
            .checked_mul(spec.m)
            .and_then(|elements| elements.checked_mul(spec.k / 2)),
        input_groups.checked_mul(a_scale_stride.ok_or(Error::InvalidMatmulBuffer)?),
        Some(b_bytes),
        spec.matrices.checked_mul(b_scale_stride.ok_or(Error::InvalidMatmulBuffer)?),
        spec.matrices.checked_mul(size_of::<f32>()),
        spec.groups.checked_mul(size_of::<u32>()),
        Some(c_bytes),
    ];
    let actual = [
        a.bytes(),
        a_scales.bytes(),
        b.bytes(),
        b_scales.bytes(),
        alphas.bytes(),
        indices.bytes(),
        c.bytes(),
    ];
    if expected.into_iter().zip(actual).all(|(left, right)| left == Some(right)) {
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
