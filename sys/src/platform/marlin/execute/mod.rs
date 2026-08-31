use std::ffi::c_void;

use super::{
    MarlinNvFp4MoeSpec,
    prepare::{native_status, validate_context},
};
use crate::{
    Error, Result,
    platform::{
        driver::{Context, Stream},
        memory::DeviceBuffer,
    },
};

mod dense;
mod mxfp4;

unsafe extern "C" {
    fn mircuda_marlin_prepare_moe_routes(
        stream: *mut c_void,
        selected: *const c_void,
        routing: *const c_void,
        sorted: *mut c_void,
        expert_ids: *mut c_void,
        padded: *mut c_void,
        offsets: *mut c_void,
        routing_f32: *mut c_void,
        assignments: i32,
        experts: i32,
        block_size: i32,
    ) -> i32;
    fn mircuda_marlin_nvfp4_moe_execute(
        stream: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        temporary: *mut c_void,
        scales: *const c_void,
        global_scales: *const c_void,
        sorted: *const c_void,
        expert_ids: *const c_void,
        padded: *const c_void,
        routing: *const c_void,
        top_k: i32,
        multiply_routing: bool,
        tokens: i32,
        output_features: i32,
        input_features: i32,
        thread_config: i32,
        locks: *mut c_void,
        atomic_reduce: bool,
    ) -> i32;
}

impl Context {
    #[allow(clippy::too_many_arguments)]
    pub fn marlin_prepare_moe_routes(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4MoeSpec,
        selected: &DeviceBuffer,
        routing: &DeviceBuffer,
        sorted: &DeviceBuffer,
        expert_ids: &DeviceBuffer,
        padded: &DeviceBuffer,
        offsets: &DeviceBuffer,
        routing_f32: &DeviceBuffer,
    ) -> Result<()> {
        validate_context(
            self,
            stream,
            &[selected, routing, sorted, expert_ids, padded, offsets, routing_f32],
        )?;
        let assignments = assignments(spec)?;
        let capacity = padded_capacity(spec)?;
        let block_size = spec.thread_config.moe_block_size();
        if selected.bytes() != assignments * size_of::<u32>()
            || routing.bytes() != assignments * size_of::<u16>()
            || sorted.bytes() != capacity * size_of::<i32>()
            || expert_ids.bytes() != capacity.div_ceil(block_size) * size_of::<i32>()
            || padded.bytes() != size_of::<i32>()
            || offsets.bytes() != spec.experts * size_of::<i32>()
            || routing_f32.bytes() != assignments * size_of::<f32>()
        {
            return Err(Error::InvalidMatmulBuffer);
        }
        self.inner.bind_to_thread()?;
        // SAFETY: every stream-bound buffer has its exact required extent.
        let status = unsafe {
            mircuda_marlin_prepare_moe_routes(
                stream.inner.cu_stream().cast(),
                selected.pointer() as *const c_void,
                routing.pointer() as *const c_void,
                sorted.pointer() as *mut c_void,
                expert_ids.pointer() as *mut c_void,
                padded.pointer() as *mut c_void,
                offsets.pointer() as *mut c_void,
                routing_f32.pointer() as *mut c_void,
                i32::try_from(assignments)?,
                i32::try_from(spec.experts)?,
                i32::try_from(block_size)?,
            )
        };
        native_status(status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn marlin_nvfp4_moe(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4MoeSpec,
        input: &DeviceBuffer,
        weight: &DeviceBuffer,
        scales: &DeviceBuffer,
        globals: &DeviceBuffer,
        routing: &DeviceBuffer,
        sorted: &DeviceBuffer,
        expert_ids: &DeviceBuffer,
        padded: &DeviceBuffer,
        temporary: &DeviceBuffer,
        locks: &DeviceBuffer,
        output: &DeviceBuffer,
        multiply_routing: bool,
        atomic_reduce: bool,
    ) -> Result<()> {
        validate_context(
            self,
            stream,
            &[
                input, weight, scales, globals, routing, sorted, expert_ids, padded, temporary,
                locks, output,
            ],
        )?;
        validate_execution(spec, input, weight, scales, globals, routing, output)?;
        self.inner.bind_to_thread()?;
        // SAFETY: ownership and projection extents were validated above; scratch is conservatively
        // sized by the public plan and the native kernel bounds its use by the padded route count.
        let status = unsafe {
            mircuda_marlin_nvfp4_moe_execute(
                stream.inner.cu_stream().cast(),
                input.pointer() as *const c_void,
                weight.pointer() as *const c_void,
                output.pointer() as *mut c_void,
                temporary.pointer() as *mut c_void,
                scales.pointer() as *const c_void,
                globals.pointer() as *const c_void,
                sorted.pointer() as *const c_void,
                expert_ids.pointer() as *const c_void,
                padded.pointer() as *const c_void,
                routing.pointer() as *const c_void,
                i32::try_from(spec.top_k)?,
                multiply_routing,
                i32::try_from(spec.tokens)?,
                i32::try_from(spec.n)?,
                i32::try_from(spec.k)?,
                spec.thread_config as i32,
                locks.pointer() as *mut c_void,
                atomic_reduce,
            )
        };
        native_status(status)
    }
}

fn validate_execution(
    spec: MarlinNvFp4MoeSpec,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    scales: &DeviceBuffer,
    globals: &DeviceBuffer,
    routing: &DeviceBuffer,
    output: &DeviceBuffer,
) -> Result<()> {
    let assignments = assignments(spec)?;
    let matrix = spec
        .experts
        .checked_mul(spec.n)
        .and_then(|value| value.checked_mul(spec.k))
        .ok_or(Error::InvalidMatmulBuffer)?;
    if input.bytes() != spec.tokens * spec.k * size_of::<u16>()
        || weight.bytes() != matrix / 2
        || scales.bytes() != matrix / 16
        || globals.bytes() != spec.experts * size_of::<f32>()
        || routing.bytes() != assignments * size_of::<f32>()
        || output.bytes() != assignments * spec.n * size_of::<u16>()
    {
        return Err(Error::InvalidMatmulBuffer);
    }
    Ok(())
}

pub(super) fn assignments(spec: MarlinNvFp4MoeSpec) -> Result<usize> {
    spec.tokens.checked_mul(spec.top_k).ok_or(Error::InvalidMatmulBuffer)
}

fn padded_capacity(spec: MarlinNvFp4MoeSpec) -> Result<usize> {
    assignments(spec)?
        .checked_add(
            spec.experts
                .checked_mul(spec.thread_config.moe_block_size() - 1)
                .ok_or(Error::InvalidMatmulBuffer)?,
        )
        .ok_or(Error::InvalidMatmulBuffer)
}
