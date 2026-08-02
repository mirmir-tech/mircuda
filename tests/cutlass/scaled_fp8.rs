#![cfg(target_os = "linux")]

use mircuda::{
    Context, DeviceBuffer, DeviceElement, Driver, MemoryPool, ScaledFp8Plan, ScaledFp8Scale,
    ScaledFp8Spec, ScaledFp8WeightScale, Stream, bf16,
};

const E4M3_ONE: u8 = 0x38;

#[test]
fn per_token_and_channel_scales_with_bias() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let spec = ScaledFp8Spec::new(
        1,
        16,
        16,
        ScaledFp8Scale::F32,
        ScaledFp8WeightScale::OutputChannel,
        true,
    )?;
    let input = copy(&context, &stream, &pool, &[E4M3_ONE; 16])?;
    let weight = copy(&context, &stream, &pool, &[E4M3_ONE; 256])?;
    let input_scales = copy(&context, &stream, &pool, &[0.5_f32])?;
    let channel_scales = (1..=16_u8).map(|value| f32::from(value) / 16.0).collect::<Vec<_>>();
    let weight_scales = copy(&context, &stream, &pool, &channel_scales)?;
    let bias_values = [bf16::from_f32(1.0); 16];
    let bias = copy(&context, &stream, &pool, &bias_values)?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, 16)?;
    ScaledFp8Plan::new(&context, &stream, spec)?.execute_f32_scales(
        &stream,
        &input,
        &weight,
        &input_scales,
        &weight_scales,
        Some(&bias),
        &mut output,
    )?;
    let actual = read(&context, &stream, &output)?;
    let expected = channel_scales
        .into_iter()
        .map(|scale| bf16::from_f32(8.0_f32.mul_add(scale, 1.0)))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn per_token_and_tensor_scale() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let spec =
        ScaledFp8Spec::new(1, 16, 16, ScaledFp8Scale::F32, ScaledFp8WeightScale::Tensor, false)?;
    let input = copy(&context, &stream, &pool, &[E4M3_ONE; 16])?;
    let weight = copy(&context, &stream, &pool, &[E4M3_ONE; 256])?;
    let input_scales = copy(&context, &stream, &pool, &[0.5_f32])?;
    let weight_scales = copy(&context, &stream, &pool, &[0.25_f32])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, 16)?;
    ScaledFp8Plan::new(&context, &stream, spec)?.execute_f32_scales(
        &stream, &input, &weight, &input_scales, &weight_scales, None, &mut output,
    )?;
    let actual = read(&context, &stream, &output)?;
    assert_eq!(actual, [bf16::from_f32(2.0); 16]);
    Ok(())
}

fn copy<T: DeviceElement>(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    values: &[T],
) -> mircuda::Result<DeviceBuffer<T>> {
    let mut host = context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = pool.allocate::<T>(stream, values.len())?;
    stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement>(
    context: &Context,
    stream: &Stream,
    values: &DeviceBuffer<T>,
) -> mircuda::Result<Vec<T>> {
    let mut host = context.allocate_pinned::<T>(values.len())?;
    stream.copy_to_host(values, &mut host)?;
    stream.synchronize()?;
    host.to_vec()
}
