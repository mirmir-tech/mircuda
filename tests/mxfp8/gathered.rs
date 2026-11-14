use mircuda::{Driver, MxFp8Gathered, MxFp8GatheredOperands, MxFp8GatheredSpec, bf16};

use super::{copy_device, read_device};

#[test]
fn gathers_mxfp8_matrix_banks_with_input_fanout() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let input = copy_device(&context, &stream, &pool, &[bf16::ONE; 32])?;
    let words = [0x3838_3838_u32; 8]
        .into_iter()
        .chain([0xb0b0_b0b0; 8])
        .chain([0x3838_3838; 8])
        .chain([0xb0b0_b0b0; 8])
        .collect::<Vec<_>>();
    let weight = copy_device(&context, &stream, &pool, &words)?;
    let scales = copy_device(&context, &stream, &pool, &[127_u8, 127, 128, 128])?;
    let selected = copy_device(&context, &stream, &pool, &[1_u32, 0])?;
    let bias = copy_device(
        &context,
        &stream,
        &pool,
        &[bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0), bf16::from_f32(4.0)],
    )?;
    let compiler = mircuda::Compiler::new(context.clone())?;
    let spec = MxFp8GatheredSpec::new_routed(1, 2, 2, 32, 2)?;
    let expected =
        [bf16::from_f32(64.0), bf16::from_f32(-32.0), bf16::from_f32(32.0), bf16::from_f32(-16.0)];
    for warps in [4, 8] {
        let mut output = pool.allocate_zeroed::<bf16>(&stream, 4)?;
        let operation = MxFp8Gathered::compile_warps(&compiler, spec, warps)?;
        operation.execute(
            &stream,
            &mut MxFp8GatheredOperands {
                input: &input,
                weight: &weight,
                scales: &scales,
                bias: None,
                selected: &selected,
                output: &mut output,
            },
        )?;
        assert_eq!(read_device(&context, &stream, &output)?, expected);
        operation.execute(
            &stream,
            &mut MxFp8GatheredOperands {
                input: &input,
                weight: &weight,
                scales: &scales,
                bias: Some(&bias),
                selected: &selected,
                output: &mut output,
            },
        )?;
        assert_eq!(
            read_device(&context, &stream, &output)?,
            [
                bf16::from_f32(67.0),
                bf16::from_f32(-28.0),
                bf16::from_f32(33.0),
                bf16::from_f32(-14.0)
            ]
        );
    }
    Ok(())
}
