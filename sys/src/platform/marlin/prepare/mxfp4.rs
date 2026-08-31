use std::ffi::c_void;

use super::{native_status, validate_context};
use crate::{
    Error, Result,
    platform::{
        driver::{Context, Stream},
        marlin::MarlinMxFp4RepackSpec,
        memory::DeviceBuffer,
    },
};

unsafe extern "C" {
    fn mircuda_marlin_mxfp4_repack(
        stream: *mut c_void,
        input: *const c_void,
        output: *mut c_void,
        experts: i32,
        size_n: i32,
        size_k: i32,
        logical_n: i32,
        logical_k: i32,
        interleaved_gate_up: bool,
    ) -> i32;
    fn mircuda_marlin_mxfp4_prepare_scales(
        stream: *mut c_void,
        input: *const c_void,
        output: *mut c_void,
        experts: i32,
        size_n: i32,
        size_k: i32,
        logical_n: i32,
        logical_k: i32,
        interleaved_gate_up: bool,
    ) -> i32;
}

impl Context {
    pub fn marlin_repack_mxfp4(
        &self,
        stream: &Stream,
        spec: MarlinMxFp4RepackSpec,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        interleaved_gate_up: bool,
    ) -> Result<()> {
        validate_context(self, stream, &[input, output])?;
        if input.bytes() != logical_elements(spec)? / 2 || output.bytes() != elements(spec)? / 2 {
            return Err(Error::InvalidMatmulBuffer);
        }
        self.inner.bind_to_thread()?;
        // SAFETY: stream ownership and exact packed matrix extents were checked above.
        let status = unsafe {
            mircuda_marlin_mxfp4_repack(
                stream.inner.cu_stream().cast(),
                input.pointer() as *const c_void,
                output.pointer() as *mut c_void,
                i32::try_from(spec.experts)?,
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
                i32::try_from(spec.logical_n)?,
                i32::try_from(spec.logical_k)?,
                interleaved_gate_up,
            )
        };
        native_status(status)
    }

    pub fn marlin_prepare_mxfp4_scales(
        &self,
        stream: &Stream,
        spec: MarlinMxFp4RepackSpec,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        interleaved_gate_up: bool,
    ) -> Result<()> {
        validate_context(self, stream, &[input, output])?;
        if input.bytes() != logical_elements(spec)? / 32 || output.bytes() != elements(spec)? / 32 {
            return Err(Error::InvalidMatmulBuffer);
        }
        self.inner.bind_to_thread()?;
        // SAFETY: ownership and exact OCP MXFP4 scale extents were checked above.
        let status = unsafe {
            mircuda_marlin_mxfp4_prepare_scales(
                stream.inner.cu_stream().cast(),
                input.pointer() as *const c_void,
                output.pointer() as *mut c_void,
                i32::try_from(spec.experts)?,
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
                i32::try_from(spec.logical_n)?,
                i32::try_from(spec.logical_k)?,
                interleaved_gate_up,
            )
        };
        native_status(status)
    }
}

fn elements(spec: MarlinMxFp4RepackSpec) -> Result<usize> {
    spec.experts
        .checked_mul(spec.n)
        .and_then(|value| value.checked_mul(spec.k))
        .ok_or(Error::InvalidMatmulBuffer)
}

fn logical_elements(spec: MarlinMxFp4RepackSpec) -> Result<usize> {
    spec.experts
        .checked_mul(spec.logical_n)
        .and_then(|value| value.checked_mul(spec.logical_k))
        .ok_or(Error::InvalidMatmulBuffer)
}
