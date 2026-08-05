use std::ffi::c_void;

use super::{MarlinNvFp4MoeSpec, native_status, validate_context};
use crate::{
    Error, Result,
    platform::{
        driver::{Context, Stream},
        memory::DeviceBuffer,
    },
};

unsafe extern "C" {
    fn mircuda_marlin_nvfp4_dense_execute(
        stream: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        temporary: *mut c_void,
        scales: *const c_void,
        global_scale: *const f32,
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
    pub fn marlin_nvfp4_dense(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4MoeSpec,
        input: &DeviceBuffer,
        weight: &DeviceBuffer,
        scales: &DeviceBuffer,
        global_scale: &DeviceBuffer,
        temporary: &DeviceBuffer,
        locks: &DeviceBuffer,
        output: &DeviceBuffer,
        atomic_reduce: bool,
    ) -> Result<()> {
        validate_context(
            self,
            stream,
            &[input, weight, scales, global_scale, temporary, locks, output],
        )?;
        validate(spec, self, input, weight, scales, global_scale, temporary, locks, output)?;
        self.inner.bind_to_thread()?;
        let status = unsafe {
            mircuda_marlin_nvfp4_dense_execute(
                stream.inner.cu_stream().cast(),
                input.pointer() as *const c_void,
                weight.pointer() as *const c_void,
                output.pointer() as *mut c_void,
                temporary.pointer() as *mut c_void,
                scales.pointer() as *const c_void,
                global_scale.pointer() as *const f32,
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

#[allow(clippy::too_many_arguments)]
fn validate(
    spec: MarlinNvFp4MoeSpec,
    context: &Context,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    scales: &DeviceBuffer,
    global_scale: &DeviceBuffer,
    temporary: &DeviceBuffer,
    locks: &DeviceBuffer,
    output: &DeviceBuffer,
) -> Result<()> {
    let sms = usize::try_from(context.device_info()?.multiprocessor_count)?;
    let matrix = spec.n.checked_mul(spec.k).ok_or(Error::InvalidMatmulBuffer)?;
    let temporary_elements = sms.checked_mul(16 * 256).ok_or(Error::InvalidMatmulBuffer)?;
    if spec.experts != 1
        || spec.top_k != 1
        || spec.tokens > 8
        || input.bytes() != spec.tokens * spec.k * size_of::<u16>()
        || weight.bytes() != matrix / 2
        || scales.bytes() != matrix / 16
        || global_scale.bytes() != size_of::<f32>()
        || temporary.bytes() < temporary_elements * size_of::<f32>()
        || locks.bytes() < sms * size_of::<i32>()
        || output.bytes() != spec.tokens * spec.n * size_of::<u16>()
    {
        return Err(Error::InvalidMatmulBuffer);
    }
    Ok(())
}
