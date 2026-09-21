use mircuda::{CublasBf16Plan, CublasBf16Spec, CublasLtBf16Plan, CublasLtBf16Spec, bf16};

use super::{copy_device, environment, read_device};

const OUTPUTS: usize = 129;
const FEATURES: usize = 512;

/// Plans lease one handle and workspace per stream; dropping some of them must
/// leave the others usable and their results unchanged.
#[test]
fn leased_cublas_state_outlives_dropped_plans() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let input = (0..4 * FEATURES)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 29)?) / 32.0 - 0.4)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    let weight = (0..OUTPUTS * FEATURES)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 19)?) / 64.0 - 0.125)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    let input = copy_device(&context, &stream, &pool, &input)?;
    let weight = copy_device(&context, &stream, &pool, &weight)?;
    let lt = |rows| {
        CublasLtBf16Plan::new(&context, &stream, CublasLtBf16Spec::new(rows, OUTPUTS, FEATURES)?)
    };
    let blas = |rows| {
        CublasBf16Plan::new(&context, &stream, CublasBf16Spec::new(rows, OUTPUTS, FEATURES)?)
    };

    let mut before = copy_device(&context, &stream, &pool, &[bf16::NAN; 4 * OUTPUTS])?;
    let mut after = copy_device(&context, &stream, &pool, &[bf16::NAN; 4 * OUTPUTS])?;
    let (mut first, mut kept) = (lt(4)?, lt(4)?);
    first.execute(&stream, &input, &weight, &mut before, 1.0, 0.0)?;
    drop(first);
    drop(lt(2)?);
    kept.execute(&stream, &input, &weight, &mut after, 1.0, 0.0)?;
    let expected = read_device(&context, &stream, &before)?;
    assert!(expected.iter().any(|value| value.to_f32().abs() > 0.25));
    assert_eq!(read_device(&context, &stream, &after)?, expected);

    let (mut first, mut kept) = (blas(4)?, blas(4)?);
    first.execute(&stream, &input, &weight, &mut before, 1.0, 0.0)?;
    drop(first);
    drop(blas(2)?);
    kept.execute(&stream, &input, &weight, &mut after, 1.0, 0.0)?;
    let expected = read_device(&context, &stream, &before)?;
    assert!(expected.iter().any(|value| value.to_f32().abs() > 0.25));
    assert_eq!(read_device(&context, &stream, &after)?, expected);
    Ok(())
}
