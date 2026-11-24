use std::num::NonZeroU16;

use mircuda::{
    DenseMatmulPlan, DenseMatmulSpec, DenseVectorKernel, DenseVectorPlan, DenseVectorSpec, bf16,
};

use super::{copy_device, environment, read_device};

mod split;

/// Row-group and block-per-row kernels plus split-K at slice counts that
/// divide the router widths evenly (2, 4, 16) and unevenly (3, 7, 18).
fn kernels() -> impl Iterator<Item = DenseVectorKernel> {
    [DenseVectorKernel::RowGroups, DenseVectorKernel::BlockPerRow]
        .into_iter()
        .chain(
            [2, 3, 4, 7, 16, 18]
                .into_iter()
                .filter_map(NonZeroU16::new)
                .map(|splits| DenseVectorKernel::SplitK { splits }),
        )
}

/// Every kernel matches the tensor-core GEMM for router-sized and wide
/// outputs; inputs not divisible by 8 admit only the row-group kernel.
#[test]
fn matches_tensor_core_gemm() -> mircuda::Result<()> {
    for (outputs, features) in [(32, 2_880), (257, 2_816), (1_031, 2_816), (64, 2_050), (33, 999)] {
        compare_patterned(outputs, features)?;
    }
    Ok(())
}

/// Split-K matches the tensor-core GEMM at every benchmarked router shape
/// and at an input width whose 8-element vectors split unevenly.
#[test]
fn split_k_matches_tensor_core_gemm_for_router_shapes() -> mircuda::Result<()> {
    for outputs in [32, 128, 256] {
        for features in [2_048, 4_096, 7_168] {
            compare_patterned(outputs, features)?;
        }
    }
    compare_patterned(48, 2_056)
}

fn compare_patterned(outputs: usize, features: usize) -> mircuda::Result<()> {
    let input = (0..features)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 31)?) / 32.0 - 0.5)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    let weight = (0..outputs * features)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 17)?) / 64.0 - 0.125)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    compare(outputs, features, &input, &weight, 0.125)
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
    DenseMatmulPlan::new(&context, &stream, DenseMatmulSpec::new(1, outputs, features)?)?
        .execute(&stream, &input, &weight, &mut expected, 1.0, 0.0)?;
    let expected = read_device(&context, &stream, &expected)?;
    let shape = DenseVectorSpec::new(outputs, features)?;
    for kernel in kernels() {
        let Ok(spec) = shape.with_kernel(kernel) else {
            assert!(kernel != DenseVectorKernel::RowGroups && !features.is_multiple_of(8));
            continue;
        };
        let mut actual = copy_device(&context, &stream, &pool, &vec![bf16::NAN; outputs])?;
        DenseVectorPlan::new(&context, &stream, spec)?
            .execute(&stream, &input, &weight, &mut actual, 1.0, 0.0)?;
        let actual = read_device(&context, &stream, &actual)?;
        assert!(actual.iter().all(|value| value.to_f32().is_finite()));
        assert_eq!(maximum(&actual), maximum(&expected), "{kernel:?} argmax");
        let error = expected
            .iter()
            .zip(&actual)
            .map(|(left, right)| (left.to_f32() - right.to_f32()).abs())
            .fold(0.0_f32, f32::max);
        assert!(error <= tolerance, "maximum {kernel:?} BF16 GEMV difference: {error}");
    }
    Ok(())
}

fn maximum(values: &[bf16]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.to_f32().total_cmp(&right.1.to_f32()))
        .map(|(index, _)| index)
}
