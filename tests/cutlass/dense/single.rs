use mircuda::{CublasDenseSpec, CublasF32Plan};

use super::{copy_device, environment, read_device};

/// Values with more mantissa than TF32 keeps, so a TF32 product would show.
fn pattern(count: usize, seed: u32) -> mircuda::Result<Vec<f32>> {
    (0..count)
        .map(|index| {
            let step = u32::try_from(index)? * 2_654_435_761_u32.wrapping_add(seed);
            Ok(f32::from(u16::try_from(step % 10_007)?) / 10_007.0 - 0.5)
        })
        .collect()
}

#[test]
fn cublas_f32_matches_a_double_precision_reference() -> mircuda::Result<()> {
    const ROWS: usize = 37;
    const OUTPUTS: usize = 65;
    const FEATURES: usize = 1_031;
    let (context, stream, pool) = environment()?;
    let (a, b) = (pattern(ROWS * FEATURES, 1)?, pattern(OUTPUTS * FEATURES, 2)?);
    let initial = pattern(ROWS * OUTPUTS, 3)?;
    let a_device = copy_device(&context, &stream, &pool, &a)?;
    let b_device = copy_device(&context, &stream, &pool, &b)?;
    let mut c_device = copy_device(&context, &stream, &pool, &initial)?;
    CublasF32Plan::new(&context, &stream, CublasDenseSpec::new(ROWS, OUTPUTS, FEATURES)?)?
        .execute(&stream, &a_device, &b_device, &mut c_device, 1.0, 1.0)?;
    let actual = read_device(&context, &stream, &c_device)?;
    for row in 0..ROWS {
        for column in 0..OUTPUTS {
            let product: f64 = (0..FEATURES)
                .map(|k| f64::from(a[row * FEATURES + k]) * f64::from(b[column * FEATURES + k]))
                .sum();
            let expected = product + f64::from(initial[row * OUTPUTS + column]);
            let difference = (f64::from(actual[row * OUTPUTS + column]) - expected).abs();
            assert!(difference < 2e-4, "{row},{column}: difference {difference}");
        }
    }
    Ok(())
}
