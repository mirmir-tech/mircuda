#[cfg(all(target_os = "linux", feature = "cutlass"))]
use mircuda::MxFp8TensorCore;
#[cfg(target_os = "linux")]
use mircuda::{
    CompileOptions, Context, DeviceBuffer, DeviceElement, Driver, LaunchConfig, MemoryPool,
    MxFp8Embedding, MxFp8EmbeddingOperands, MxFp8Gathered, MxFp8GatheredOperands, MxFp8Matmul,
    Stream, TypedKernel, bf16, cuda_export, cuda_kernel_file,
};
use mircuda::{MxFp8EmbeddingSpec, MxFp8GatheredSpec, MxFp8Spec};

#[cfg(target_os = "linux")]
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
fn validates_mxfp8_geometry_without_a_device() -> mircuda::Result<()> {
    let spec = MxFp8Spec::new(2, 32, 3)?;
    assert_eq!(spec.tokens(), 2);
    assert_eq!(spec.input_features(), 32);
    assert_eq!(spec.output_features(), 3);
    assert_eq!(spec.weight_words()?, 24);
    assert_eq!(spec.scale_elements()?, 3);
    assert!(MxFp8Spec::new(1, 31, 1).is_err());
    assert_eq!(MxFp8EmbeddingSpec::new(3, 32, 1.0)?.weight_words()?, 24);
    assert!(MxFp8GatheredSpec::new_routed(1, 2, 2, 32, 3).is_ok());
    Ok(())
}

#[test]
#[cfg(target_os = "linux")]
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

#[test]
#[cfg(target_os = "linux")]
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
#[cfg(all(target_os = "linux", feature = "cutlass"))]
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

#[test]
#[cfg(target_os = "linux")]
fn quantizes_mxfp8_with_ceil_ue8m0_scale() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let compiler = mircuda::Compiler::new(context.clone())?;
    let module =
        compiler.compile(cuda_kernel_file!("../kernels/mxfp8.cu"), &CompileOptions::default())?;
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
#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
fn copy_device<T: DeviceElement>(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    values: &[T],
) -> mircuda::Result<DeviceBuffer<T>> {
    let mut host = context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = pool.allocate::<T>(stream, values.len())?;
    stream.copy_to_device(&mut host, &mut device)?;
    stream.synchronize()?;
    Ok(device)
}

#[cfg(target_os = "linux")]
fn read_device<T: DeviceElement>(
    context: &Context,
    stream: &Stream,
    device: &DeviceBuffer<T>,
) -> mircuda::Result<Vec<T>> {
    let mut host = context.allocate_pinned::<T>(device.len())?;
    stream.copy_to_host(device, &mut host)?;
    stream.synchronize()?;
    host.to_vec()
}
