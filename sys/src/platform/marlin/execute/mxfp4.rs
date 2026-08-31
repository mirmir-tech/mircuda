use std::ffi::c_void;

use super::{assignments, native_status, validate_context};
use crate::{
    Error, Result,
    platform::{
        driver::{Context, Stream},
        marlin::MarlinNvFp4MoeSpec,
        memory::DeviceBuffer,
    },
};

unsafe extern "C" {
    fn mircuda_marlin_mxfp4_moe_execute(
        stream: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        temporary: *mut c_void,
        scales: *const c_void,
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
    pub fn marlin_mxfp4_moe(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4MoeSpec,
        input: &DeviceBuffer,
        weight: &DeviceBuffer,
        scales: &DeviceBuffer,
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
            &[input, weight, scales, routing, sorted, expert_ids, padded, temporary, locks, output],
        )?;
        validate_execution(spec, input, weight, scales, routing, output)?;
        self.inner.bind_to_thread()?;
        // SAFETY: exact OCP MXFP4 matrix and route extents were validated above.
        let status = unsafe {
            mircuda_marlin_mxfp4_moe_execute(
                stream.inner.cu_stream().cast(),
                input.pointer() as *const c_void,
                weight.pointer() as *const c_void,
                output.pointer() as *mut c_void,
                temporary.pointer() as *mut c_void,
                scales.pointer() as *const c_void,
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
        || scales.bytes() != matrix / 32
        || routing.bytes() != assignments * size_of::<f32>()
        || output.bytes() != assignments * spec.n * size_of::<u16>()
    {
        return Err(Error::InvalidMatmulBuffer);
    }
    Ok(())
}
