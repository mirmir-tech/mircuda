use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::CudaStream;

use super::super::{
    driver::{Context, Stream},
    memory::{DeviceBuffer, ensure_stream},
};
use crate::{Error, Result};

unsafe extern "C" {
    fn mircuda_scaled_fp8_create(
        m: i32,
        n: i32,
        k: i32,
        scale_type: i32,
        weight_scale_type: i32,
        has_bias: i32,
        tile: i32,
        stream: *mut c_void,
        output: *mut *mut c_void,
    ) -> i32;
    fn mircuda_scaled_fp8_workspace_bytes(plan: *const c_void) -> usize;
    fn mircuda_scaled_fp8_execute(
        plan: *const c_void,
        stream: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        input_scales: *const f32,
        weight_scales: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
    ) -> i32;
    fn mircuda_scaled_fp8_destroy(plan: *mut c_void);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaledFp8ScaleType {
    F32,
    Bf16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaledFp8WeightScaleType {
    Tensor,
    OutputChannel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaledFp8Tile {
    M16N64K128,
    M16N128K64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScaledFp8Spec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub scale_type: ScaledFp8ScaleType,
    pub weight_scale_type: ScaledFp8WeightScaleType,
    pub has_bias: bool,
    pub tile: ScaledFp8Tile,
}

#[derive(Debug)]
pub struct ScaledFp8Plan {
    raw: NonNull<c_void>,
    stream: Arc<CudaStream>,
    spec: ScaledFp8Spec,
}

// SAFETY: execution is confined to the retained stream and the native plan is immutable.
unsafe impl Send for ScaledFp8Plan {}
// SAFETY: the plan is immutable after construction and all launches use its retained stream.
unsafe impl Sync for ScaledFp8Plan {}

impl Context {
    pub fn create_scaled_fp8_plan(
        &self,
        stream: &Stream,
        spec: ScaledFp8Spec,
    ) -> Result<ScaledFp8Plan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        self.inner.bind_to_thread()?;
        let mut raw = std::ptr::null_mut();
        let scale_type = i32::from(spec.scale_type == ScaledFp8ScaleType::Bf16);
        let weight_scale_type =
            i32::from(spec.weight_scale_type == ScaledFp8WeightScaleType::OutputChannel);
        let tile = i32::from(spec.tile == ScaledFp8Tile::M16N128K64);
        // SAFETY: the returned pointer is checked and uniquely owned.
        check(unsafe {
            mircuda_scaled_fp8_create(
                i32::try_from(spec.m)?,
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
                scale_type,
                weight_scale_type,
                i32::from(spec.has_bias),
                tile,
                stream.inner.cu_stream().cast(),
                &raw mut raw,
            )
        })?;
        Ok(ScaledFp8Plan {
            raw: NonNull::new(raw).ok_or(Error::NullAllocation)?,
            stream: stream.inner.clone(),
            spec,
        })
    }
}

impl ScaledFp8Plan {
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        // SAFETY: raw remains live until Drop.
        unsafe { mircuda_scaled_fp8_workspace_bytes(self.raw.as_ptr()) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer,
        weight: &DeviceBuffer,
        input_scales: &DeviceBuffer,
        weight_scales: &DeviceBuffer,
        bias: Option<&DeviceBuffer>,
        output: &DeviceBuffer,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        for buffer in [input, weight, input_scales, weight_scales, output] {
            ensure_stream(buffer, stream)?;
        }
        if let Some(bias) = bias {
            ensure_stream(bias, stream)?;
        }
        validate(self.spec, input, weight, input_scales, weight_scales, bias, output)?;
        self.stream.context().bind_to_thread()?;
        // SAFETY: all buffers and fixed dimensions were validated.
        check(unsafe {
            mircuda_scaled_fp8_execute(
                self.raw.as_ptr(),
                stream.inner.cu_stream().cast(),
                input.pointer() as *const c_void,
                weight.pointer() as *const c_void,
                input_scales.pointer() as *const f32,
                weight_scales.pointer() as *const c_void,
                bias.map_or(std::ptr::null(), |value| value.pointer() as *const c_void),
                output.pointer() as *mut c_void,
            )
        })
    }
}

impl Drop for ScaledFp8Plan {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        // SAFETY: this value uniquely owns raw.
        unsafe { mircuda_scaled_fp8_destroy(self.raw.as_ptr()) };
    }
}

fn validate(
    spec: ScaledFp8Spec,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    input_scales: &DeviceBuffer,
    weight_scales: &DeviceBuffer,
    bias: Option<&DeviceBuffer>,
    output: &DeviceBuffer,
) -> Result<()> {
    let scale_bytes = match spec.scale_type {
        ScaledFp8ScaleType::F32 => 4,
        ScaledFp8ScaleType::Bf16 => 2,
    };
    let weight_scale_elements = match spec.weight_scale_type {
        ScaledFp8WeightScaleType::Tensor => 1,
        ScaledFp8WeightScaleType::OutputChannel => spec.n,
    };
    let valid = input.bytes() == spec.m * spec.k
        && weight.bytes() == spec.n * spec.k
        && input_scales.bytes() == spec.m * 4
        && weight_scales.bytes() == weight_scale_elements * scale_bytes
        && bias.map_or(!spec.has_bias, |value| spec.has_bias && value.bytes() == spec.n * 2)
        && output.bytes() == spec.m * spec.n * 2;
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
