use std::{ffi::c_void, sync::Arc};

use super::MarlinNvFp4RepackSpec;
use crate::{
    Error, Result,
    platform::{
        driver::{Context, Stream},
        memory::{DeviceBuffer, ensure_stream},
    },
};

unsafe extern "C" {
    fn mircuda_marlin_nvfp4_repack(
        stream: *mut c_void,
        input: *const c_void,
        output: *mut c_void,
        experts: i32,
        size_n: i32,
        size_k: i32,
    ) -> i32;
    fn mircuda_marlin_nvfp4_repack_pair(
        stream: *mut c_void,
        left: *const c_void,
        right: *const c_void,
        output: *mut c_void,
        experts: i32,
        size_n: i32,
        size_k: i32,
    ) -> i32;
    fn mircuda_marlin_nvfp4_prepare_scales(
        stream: *mut c_void,
        left: *const c_void,
        right: *const c_void,
        globals: *const c_void,
        output: *mut c_void,
        output_globals: *mut c_void,
        maximum: *mut c_void,
        experts: i32,
        size_n: i32,
        size_k: i32,
    ) -> i32;
}

impl Context {
    pub fn marlin_repack_nvfp4(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4RepackSpec,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
    ) -> Result<()> {
        self.repack(stream, spec, input, None, output)
    }

    pub fn marlin_repack_nvfp4_pair(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4RepackSpec,
        left: &DeviceBuffer,
        right: &DeviceBuffer,
        output: &DeviceBuffer,
    ) -> Result<()> {
        self.repack(stream, spec, left, Some(right), output)
    }

    fn repack(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4RepackSpec,
        left: &DeviceBuffer,
        right: Option<&DeviceBuffer>,
        output: &DeviceBuffer,
    ) -> Result<()> {
        validate_context(self, stream, &[left, output])?;
        if let Some(right) = right {
            validate_context(self, stream, &[right])?;
        }
        let output_bytes = matrix_bytes(spec)?;
        let input_bytes = if right.is_some() {
            output_bytes / 2
        } else {
            output_bytes
        };
        if left.bytes() != input_bytes
            || right.is_some_and(|value| value.bytes() != input_bytes)
            || output.bytes() != output_bytes
        {
            return Err(Error::InvalidMatmulBuffer);
        }
        self.inner.bind_to_thread()?;
        // SAFETY: stream ownership and exact buffer extents were checked above.
        let status = unsafe {
            if let Some(right) = right {
                mircuda_marlin_nvfp4_repack_pair(
                    stream.inner.cu_stream().cast(),
                    left.pointer() as *const c_void,
                    right.pointer() as *const c_void,
                    output.pointer() as *mut c_void,
                    i32::try_from(spec.experts)?,
                    i32::try_from(spec.n)?,
                    i32::try_from(spec.k)?,
                )
            } else {
                mircuda_marlin_nvfp4_repack(
                    stream.inner.cu_stream().cast(),
                    left.pointer() as *const c_void,
                    output.pointer() as *mut c_void,
                    i32::try_from(spec.experts)?,
                    i32::try_from(spec.n)?,
                    i32::try_from(spec.k)?,
                )
            }
        };
        native_status(status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn marlin_prepare_nvfp4_scales(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4RepackSpec,
        left: &DeviceBuffer,
        right: Option<&DeviceBuffer>,
        globals: &DeviceBuffer,
        output: &DeviceBuffer,
        output_globals: &DeviceBuffer,
        maximum: &DeviceBuffer,
    ) -> Result<()> {
        validate_context(self, stream, &[left, globals, output, output_globals, maximum])?;
        if let Some(right) = right {
            validate_context(self, stream, &[right])?;
        }
        let scales = matrix_elements(spec)? / 16;
        let input_scales = if right.is_some() {
            scales / 2
        } else {
            scales
        };
        if left.bytes() != input_scales
            || right.is_some_and(|value| value.bytes() != input_scales)
            || output.bytes() != scales
            || globals.bytes() != spec.experts * size_of::<f32>()
            || output_globals.bytes() != globals.bytes()
            || maximum.bytes() != size_of::<u32>()
        {
            return Err(Error::InvalidMatmulBuffer);
        }
        self.inner.bind_to_thread()?;
        // SAFETY: buffer ownership and exact byte lengths were checked above.
        let status = unsafe {
            mircuda_marlin_nvfp4_prepare_scales(
                stream.inner.cu_stream().cast(),
                left.pointer() as *const c_void,
                right.map_or(std::ptr::null(), |value| value.pointer() as *const c_void),
                globals.pointer() as *const c_void,
                output.pointer() as *mut c_void,
                output_globals.pointer() as *mut c_void,
                maximum.pointer() as *mut c_void,
                i32::try_from(spec.experts)?,
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
            )
        };
        native_status(status)
    }
}

pub(super) fn validate_context(
    context: &Context,
    stream: &Stream,
    buffers: &[&DeviceBuffer],
) -> Result<()> {
    if !Arc::ptr_eq(&context.inner, stream.inner.context()) {
        return Err(Error::ContextMismatch);
    }
    for buffer in buffers {
        ensure_stream(buffer, stream)?;
    }
    Ok(())
}

pub(super) const fn native_status(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Cutlass(status))
    }
}

fn matrix_elements(spec: MarlinNvFp4RepackSpec) -> Result<usize> {
    spec.experts
        .checked_mul(spec.n)
        .and_then(|value| value.checked_mul(spec.k))
        .ok_or(Error::InvalidMatmulBuffer)
}

fn matrix_bytes(spec: MarlinNvFp4RepackSpec) -> Result<usize> {
    matrix_elements(spec)?.checked_div(2).ok_or(Error::InvalidMatmulBuffer)
}
