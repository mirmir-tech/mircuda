#![cfg(all(target_os = "linux", feature = "cutlass"))]

use mircuda::{
    Context, DenseMatmulPlan, DenseMatmulSpec, DenseVectorPlan, DenseVectorSpec, DeviceBuffer,
    DeviceElement, Driver, MemoryPool, Stream, bf16, f16,
};
#[cfg(feature = "cublaslt")]
use mircuda::{CublasBf16Plan, CublasBf16Spec, CublasLtBf16Plan, CublasLtBf16Spec};

mod vector;

const N: usize = 128;
const K: usize = 128;

#[test]
fn dense_bf16_supports_decode_and_prefill() -> mircuda::Result<()> {
    run_bf16(1)?;
    run_bf16(128)
}

#[test]
fn dense_bf16_supports_profiled_wide_decode() -> mircuda::Result<()> {
    const OUTPUTS: usize = 8_192;
    let (context, stream, pool) = environment()?;
    let input = copy_device(&context, &stream, &pool, &[bf16::ONE; K])?;
    let weight = copy_device(&context, &stream, &pool, &vec![bf16::ONE; OUTPUTS * K])?;
    let mut output = copy_device(&context, &stream, &pool, &[bf16::ZERO; OUTPUTS])?;
    DenseMatmulPlan::new(&context, &stream, DenseMatmulSpec::new(1, OUTPUTS, K)?)?
        .execute(&stream, &input, &weight, &mut output, 1.0, 0.0)?;
    let actual = read_device(&context, &stream, &output)?;
    assert!(actual.iter().all(|value| *value == bf16::from_f32(128.0)));
    Ok(())
}

#[test]
fn dense_bf16_retains_fp32_output() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let input = copy_device(&context, &stream, &pool, &[bf16::from_f32(0.1); K])?;
    let weight = copy_device(&context, &stream, &pool, &vec![bf16::from_f32(0.1); N * K])?;
    let mut output = copy_device(&context, &stream, &pool, &[f32::NAN; N])?;
    DenseMatmulPlan::<bf16, f32>::new(&context, &stream, DenseMatmulSpec::new(1, N, K)?)?
        .execute(&stream, &input, &weight, &mut output, 1.0, 0.0)?;
    let actual = read_device(&context, &stream, &output)?;
    let rounded = bf16::from_f32(actual[0]).to_f32();
    assert!(
        actual
            .iter()
            .all(|value| value.is_finite() && (*value - actual[0]).abs() < f32::EPSILON)
    );
    assert!((actual[0] - rounded).abs() > f32::EPSILON, "FP32 result was rounded to BF16");
    Ok(())
}

#[test]
fn dense_bf16_supports_unaligned_shapes() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let a = copy_device(&context, &stream, &pool, &[bf16::ONE; 6])?;
    let b = copy_device(&context, &stream, &pool, &[bf16::ONE; 12])?;
    let mut c = copy_device(&context, &stream, &pool, &[bf16::ONE; 8])?;
    let mut plan = DenseMatmulPlan::<bf16>::new(&context, &stream, DenseMatmulSpec::new(2, 4, 3)?)?;
    plan.execute(&stream, &a, &b, &mut c, 1.0, 1.0)?;
    drop(plan);
    let actual = read_device(&context, &stream, &c)?;
    assert!(actual.iter().all(|value| *value == bf16::from_f32(4.0)));
    Ok(())
}

#[test]
fn dense_f16_supports_decode() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let a = copy_device(&context, &stream, &pool, &vec![f16::ONE; K])?;
    let b = copy_device(&context, &stream, &pool, &vec![f16::ONE; N * K])?;
    let mut c = copy_device(&context, &stream, &pool, &vec![f16::ONE; N])?;
    let mut plan = DenseMatmulPlan::<f16>::new(&context, &stream, DenseMatmulSpec::new(1, N, K)?)?;
    plan.execute(&stream, &a, &b, &mut c, 1.0, 1.0)?;
    drop(plan);
    let actual = read_device(&context, &stream, &c)?;
    let expected = f16::from_f32(129.0);
    assert!(actual.iter().all(|value| *value == expected));
    Ok(())
}

#[test]
fn dense_f16_vector_supports_alpha_and_beta() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let input = copy_device(&context, &stream, &pool, &[f16::ONE; K])?;
    let weight = copy_device(&context, &stream, &pool, &vec![f16::ONE; N * K])?;
    let mut output = copy_device(&context, &stream, &pool, &[f16::ONE; N])?;
    DenseVectorPlan::new(&context, &stream, DenseVectorSpec::new(N, K)?)?
        .execute(&stream, &input, &weight, &mut output, 0.5, 1.0)?;
    let actual = read_device(&context, &stream, &output)?;
    assert!(actual.iter().all(|value| *value == f16::from_f32(65.0)));
    Ok(())
}

#[cfg(feature = "cublaslt")]
#[test]
fn cublaslt_bf16_matches_tensor_core_gemm() -> mircuda::Result<()> {
    const OUTPUTS: usize = 257;
    const FEATURES: usize = 2_816;
    let (context, stream, pool) = environment()?;
    let input = (0..FEATURES)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 31)?) / 32.0 - 0.5)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    let weight = (0..OUTPUTS * FEATURES)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 17)?) / 64.0 - 0.125)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    let input = copy_device(&context, &stream, &pool, &input)?;
    let weight = copy_device(&context, &stream, &pool, &weight)?;
    let mut expected = copy_device(&context, &stream, &pool, &[bf16::ZERO; OUTPUTS])?;
    let mut actual = copy_device(&context, &stream, &pool, &[bf16::NAN; OUTPUTS])?;
    DenseMatmulPlan::new(&context, &stream, DenseMatmulSpec::new(1, OUTPUTS, FEATURES)?)?
        .execute(&stream, &input, &weight, &mut expected, 1.0, 0.0)?;
    CublasLtBf16Plan::new(&context, &stream, CublasLtBf16Spec::new(1, OUTPUTS, FEATURES)?)?
        .execute(&stream, &input, &weight, &mut actual, 1.0, 0.0)?;
    let expected = read_device(&context, &stream, &expected)?;
    let actual = read_device(&context, &stream, &actual)?;
    assert!(actual.iter().all(|value| value.to_f32().is_finite()));
    let error = expected
        .iter()
        .zip(&actual)
        .map(|(left, right)| (left.to_f32() - right.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(error <= 0.125, "maximum cuBLASLt BF16 difference: {error}");
    Ok(())
}

#[cfg(feature = "cublaslt")]
#[test]
fn cublas_bf16_matches_tensor_core_gemm() -> mircuda::Result<()> {
    const OUTPUTS: usize = 257;
    const FEATURES: usize = 2_816;
    let (context, stream, pool) = environment()?;
    let input = (0..FEATURES)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 31)?) / 32.0 - 0.5)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    let weight = (0..OUTPUTS * FEATURES)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 17)?) / 64.0 - 0.125)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    let input = copy_device(&context, &stream, &pool, &input)?;
    let weight = copy_device(&context, &stream, &pool, &weight)?;
    let mut expected = copy_device(&context, &stream, &pool, &[bf16::ZERO; OUTPUTS])?;
    let mut actual = copy_device(&context, &stream, &pool, &[bf16::NAN; OUTPUTS])?;
    DenseMatmulPlan::new(&context, &stream, DenseMatmulSpec::new(1, OUTPUTS, FEATURES)?)?
        .execute(&stream, &input, &weight, &mut expected, 1.0, 0.0)?;
    CublasBf16Plan::new(&context, &stream, CublasBf16Spec::new(1, OUTPUTS, FEATURES)?)?
        .execute(&stream, &input, &weight, &mut actual, 1.0, 0.0)?;
    let expected = read_device(&context, &stream, &expected)?;
    let actual = read_device(&context, &stream, &actual)?;
    assert!(actual.iter().all(|value| value.to_f32().is_finite()));
    let error = expected
        .iter()
        .zip(&actual)
        .map(|(left, right)| (left.to_f32() - right.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(error <= 0.125, "maximum cuBLAS BF16 difference: {error}");
    Ok(())
}

fn run_bf16(m: usize) -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let a = copy_device(&context, &stream, &pool, &vec![bf16::ONE; m * K])?;
    let b = copy_device(&context, &stream, &pool, &vec![bf16::ONE; N * K])?;
    let mut c = copy_device(&context, &stream, &pool, &vec![bf16::ONE; m * N])?;
    let mut plan = DenseMatmulPlan::<bf16>::new(&context, &stream, DenseMatmulSpec::new(m, N, K)?)?;
    plan.execute(&stream, &a, &b, &mut c, 0.5, 1.0)?;
    assert_eq!(plan.spec(), DenseMatmulSpec::new(m, N, K)?);
    drop(plan);
    let actual = read_device(&context, &stream, &c)?;
    let expected = bf16::from_f32(65.0);
    assert!(actual.iter().all(|value| *value == expected));
    Ok(())
}

fn environment() -> mircuda::Result<(Context, Stream, MemoryPool)> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    Ok((context, stream, pool))
}

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
