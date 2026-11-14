use mircuda::{
    CompileOptions, DeviceBuffer, Driver, LaunchConfig, MxFp8Embedding, MxFp8EmbeddingOperands,
    MxFp8EmbeddingSpec, TypedKernel, bf16, cuda_export, cuda_kernel_file,
};

use super::{copy_device, read_device};

cuda_export!(
    Quantize = "mircuda_mxfp8_quantize_bf16"(
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        rows: u32,
        columns: u32,
    )
);

#[test]
fn quantizes_mxfp8_with_ceil_ue8m0_scale() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let compiler = mircuda::Compiler::new(context.clone())?;
    let module = compiler
        .compile(cuda_kernel_file!("../../kernels/mxfp8.cu"), &CompileOptions::default())?;
    let quantize: TypedKernel<Quantize> = module.kernel()?;
    let input = copy_device(&context, &stream, &pool, &[bf16::from_f32(1.0); 32])?;
    let mut output = pool.allocate_zeroed::<u8>(&stream, 32)?;
    let mut scales = pool.allocate_zeroed::<u8>(&stream, 512)?;
    quantize.launch(
        &stream,
        LaunchConfig {
            grid: (1, 1, 1),
            block: (32, 1, 1),
            shared_memory_bytes: 0,
        },
        (&input, &mut output, &mut scales, 1, 32),
    )?;
    assert_eq!(read_device(&context, &stream, &output)?, [0x78; 32]);
    let scales = read_device(&context, &stream, &scales)?;
    assert_eq!(scales[0], 119);
    assert!(scales[1..].iter().all(|scale| *scale == 0));
    Ok(())
}

#[test]
fn dequantizes_only_selected_mxfp8_embedding_rows() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let words = [0x3838_3838_u32; 8].into_iter().chain([0xb0b0_b0b0_u32; 8]).collect::<Vec<_>>();
    let weight = copy_device(&context, &stream, &pool, &words)?;
    let scales = copy_device(&context, &stream, &pool, &[127_u8, 128])?;
    let selected = copy_device(&context, &stream, &pool, &[1_u32, 0])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, 64)?;
    let embedding = MxFp8Embedding::compile(
        &mircuda::Compiler::new(context.clone())?,
        MxFp8EmbeddingSpec::new(2, 32, 1.0)?,
    )?;
    embedding.execute(
        &stream,
        MxFp8EmbeddingOperands {
            weight: &weight,
            scales: &scales,
            selected: &selected,
            output: &mut output,
        },
        0,
        2,
    )?;
    let values = read_device(&context, &stream, &output)?;
    assert!(values[..32].iter().all(|value| *value == bf16::from_f32(-1.0)));
    assert!(values[32..].iter().all(|value| *value == bf16::from_f32(1.0)));
    Ok(())
}
