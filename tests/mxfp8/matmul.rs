#[cfg(feature = "cutlass")]
use mircuda::MxFp8TensorCore;
use mircuda::{Driver, MxFp8Matmul, MxFp8Spec, bf16};

use super::{copy_device, read_device};

#[test]
fn executes_pinned_mxfp8_words_and_e8m0_scales() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let input = (1_u8..=5)
        .flat_map(|value| [bf16::from_f32(f32::from(value)); 32])
        .collect::<Vec<_>>();
    let input = copy_device(&context, &stream, &pool, &input)?;
    let words = [0x3838_3838_u32; 8].into_iter().chain([0xb0b0_b0b0_u32; 8]).collect::<Vec<_>>();
    let weight = copy_device(&context, &stream, &pool, &words)?;
    let scales = copy_device(&context, &stream, &pool, &[127_u8, 128])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, 10)?;
    let projection =
        MxFp8Matmul::compile(&mircuda::Compiler::new(context.clone())?, MxFp8Spec::new(5, 32, 2)?)?;
    projection.execute(&stream, &input, &weight, &scales, &mut output)?;
    let expected = (1_u8..=5)
        .flat_map(|value| {
            let value = 32.0 * f32::from(value);
            [bf16::from_f32(value), bf16::from_f32(-value)]
        })
        .collect::<Vec<_>>();
    assert_eq!(read_device(&context, &stream, &output)?, expected);
    Ok(())
}

#[test]
#[cfg(feature = "cutlass")]
fn executes_mxfp8_tensor_core_with_device_quantized_activations() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let spec = MxFp8Spec::new(4, 128, 128)?;
    let operation = MxFp8TensorCore::new(
        &mircuda::Compiler::new(context.clone())?,
        &context,
        &pool,
        &stream,
        spec,
    )?;
    let input = copy_device(&context, &stream, &pool, &[bf16::from_f32(1.0); 512])?;
    let weight = copy_device(&context, &stream, &pool, &[0x3838_3838_u32; 4_096])?;
    let scales = copy_device(&context, &stream, &pool, &[127_u8; 512])?;
    let scales = operation.swizzle_weight_scales(&pool, &stream, &scales)?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, 512)?;
    operation.execute(&stream, &input, &weight, &scales, &mut output)?;
    assert!(
        read_device(&context, &stream, &output)?
            .iter()
            .all(|value| *value == bf16::from_f32(128.0))
    );
    Ok(())
}
