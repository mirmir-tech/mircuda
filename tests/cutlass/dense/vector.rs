use mircuda::{DenseMatmulPlan, DenseMatmulSpec, DenseVectorPlan, DenseVectorSpec, bf16};

use super::{copy_device, environment, read_device};

#[test]
fn matches_tensor_core_gemm() -> mircuda::Result<()> {
    const OUTPUTS: usize = 257;
    const FEATURES: usize = 2_816;
    let input = (0..FEATURES)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 31)?) / 32.0 - 0.5)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    let weight = (0..OUTPUTS * FEATURES)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 17)?) / 64.0 - 0.125)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    compare(OUTPUTS, FEATURES, &input, &weight, 0.125)
}

#[test]
#[ignore = "allocates the full Qwen2 output-head geometry"]
fn matches_tensor_core_gemm_for_qwen2_output_head() -> mircuda::Result<()> {
    const OUTPUTS: usize = 151_936;
    const FEATURES: usize = 896;
    let input = (0..FEATURES)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 29)?) / 32.0 - 0.4375)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    let row = (0..FEATURES)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 23)?) / 64.0 - 0.171_875)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    let mut weight = row.repeat(OUTPUTS);
    for (index, values) in weight.as_chunks_mut::<FEATURES>().0.iter_mut().enumerate() {
        values[0] = bf16::from_f32(f32::from(u8::try_from(index % 127)?) / 256.0);
    }
    compare(OUTPUTS, FEATURES, &input, &weight, 0.125)
}

fn compare(
    outputs: usize,
    features: usize,
    input: &[bf16],
    weight: &[bf16],
    tolerance: f32,
) -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let input = copy_device(&context, &stream, &pool, input)?;
    let weight = copy_device(&context, &stream, &pool, weight)?;
    let mut expected = copy_device(&context, &stream, &pool, &vec![bf16::ZERO; outputs])?;
    let mut actual = copy_device(&context, &stream, &pool, &vec![bf16::NAN; outputs])?;
    DenseMatmulPlan::new(&context, &stream, DenseMatmulSpec::new(1, outputs, features)?)?
        .execute(&stream, &input, &weight, &mut expected, 1.0, 0.0)?;
    DenseVectorPlan::new(&context, &stream, DenseVectorSpec::new(outputs, features)?)?
        .execute(&stream, &input, &weight, &mut actual, 1.0, 0.0)?;
    let expected = read_device(&context, &stream, &expected)?;
    let actual = read_device(&context, &stream, &actual)?;
    assert!(actual.iter().all(|value| value.to_f32().is_finite()));
    assert_eq!(maximum(&actual), maximum(&expected));
    let error = expected
        .iter()
        .zip(&actual)
        .map(|(left, right)| (left.to_f32() - right.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(error <= tolerance, "maximum BF16 GEMV difference: {error}");
    Ok(())
}

fn maximum(values: &[bf16]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.to_f32().total_cmp(&right.1.to_f32()))
        .map(|(index, _)| index)
}
