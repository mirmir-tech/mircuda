use mircuda::{
    CublasElement, CublasGemmOffsets, CublasGemmOperand, CublasGemmPlan, CublasGemmSpec, bf16,
};

use super::{copy_device, environment, read_device};

const LENGTH: usize = 45;
const HEADS: usize = 3;
const DIM: usize = 16;
const WIDTH: usize = 3 * HEADS * DIM;

fn pattern(count: usize, seed: u32) -> mircuda::Result<Vec<f32>> {
    (0..count)
        .map(|index| {
            let step = u32::try_from(index)? * 2_654_435_761_u32.wrapping_add(seed);
            Ok(f32::from(u16::try_from(step % 10_007)?) / 10_007.0 - 0.5)
        })
        .collect()
}

/// Per-head `Q · Kᵀ` of a fused `[length, 3 × heads × dim]` projection,
/// read straight from the projection with no copy.
fn scores_spec() -> mircuda::Result<CublasGemmSpec> {
    let fused = |transposed| CublasGemmOperand { leading: WIDTH, stride: DIM, transposed };
    let scores = CublasGemmOperand {
        leading: LENGTH,
        stride: LENGTH * LENGTH,
        transposed: false,
    };
    CublasGemmSpec::new((LENGTH, LENGTH, DIM, HEADS), fused(false), fused(true), scores)
}

fn check_scores<I: CublasElement>(
    round: impl Fn(f32) -> (I, f64),
    tolerance: f64,
) -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let qkv = pattern(LENGTH * WIDTH, 7)?;
    let (input, rounded): (Vec<I>, Vec<f64>) = qkv.iter().copied().map(round).unzip();
    let qkv_device = copy_device(&context, &stream, &pool, &input)?;
    let mut scores =
        copy_device(&context, &stream, &pool, &vec![0.0_f32; HEADS * LENGTH * LENGTH])?;
    let scale = 0.25;
    let offsets = CublasGemmOffsets { left: 0, right: HEADS * DIM, output: 0 };
    CublasGemmPlan::<I, f32>::new(&context, &stream, scores_spec()?)?.execute(
        &stream,
        (&qkv_device, &qkv_device, &mut scores),
        offsets,
        (scale, 0.0),
    )?;
    let actual = read_device(&context, &stream, &scores)?;
    for head in 0..HEADS {
        for query in 0..LENGTH {
            for key in 0..LENGTH {
                let dot: f64 = (0..DIM)
                    .map(|d| {
                        rounded[query * WIDTH + head * DIM + d]
                            * rounded[key * WIDTH + (HEADS + head) * DIM + d]
                    })
                    .sum();
                let got = actual[(head * LENGTH + query) * LENGTH + key];
                let difference = dot.mul_add(-f64::from(scale), f64::from(got)).abs();
                assert!(difference < tolerance, "{head},{query},{key}: difference {difference}");
            }
        }
    }
    Ok(())
}

#[test]
fn strided_batched_scores_match_a_reference_in_f32() -> mircuda::Result<()> {
    check_scores(|value| (value, f64::from(value)), 1e-5)
}

#[test]
fn strided_batched_scores_match_a_reference_in_bf16() -> mircuda::Result<()> {
    check_scores(|value| (bf16::from_f32(value), bf16::from_f32(value).to_f64()), 1e-4)
}

#[test]
fn strided_batched_values_write_into_a_head_interleaved_output() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let qkv = pattern(LENGTH * WIDTH, 11)?;
    let weights = pattern(HEADS * LENGTH * LENGTH, 13)?;
    let qkv_device = copy_device(&context, &stream, &pool, &qkv)?;
    let weights_device = copy_device(&context, &stream, &pool, &weights)?;
    let mut output = copy_device(&context, &stream, &pool, &vec![0.0_f32; LENGTH * HEADS * DIM])?;
    let spec = CublasGemmSpec::new(
        (LENGTH, DIM, LENGTH, HEADS),
        CublasGemmOperand {
            leading: LENGTH,
            stride: LENGTH * LENGTH,
            transposed: false,
        },
        CublasGemmOperand {
            leading: WIDTH,
            stride: DIM,
            transposed: false,
        },
        CublasGemmOperand {
            leading: HEADS * DIM,
            stride: DIM,
            transposed: false,
        },
    )?;
    let offsets = CublasGemmOffsets {
        left: 0,
        right: 2 * HEADS * DIM,
        output: 0,
    };
    CublasGemmPlan::<f32, f32>::new(&context, &stream, spec)?.execute(
        &stream,
        (&weights_device, &qkv_device, &mut output),
        offsets,
        (1.0, 0.0),
    )?;
    let actual = read_device(&context, &stream, &output)?;
    for head in 0..HEADS {
        for query in 0..LENGTH {
            for d in 0..DIM {
                let expected: f64 = (0..LENGTH)
                    .map(|key| {
                        f64::from(weights[(head * LENGTH + query) * LENGTH + key])
                            * f64::from(qkv[key * WIDTH + (2 * HEADS + head) * DIM + d])
                    })
                    .sum();
                let got = actual[query * HEADS * DIM + head * DIM + d];
                let difference = (f64::from(got) - expected).abs();
                assert!(difference < 1e-5, "{head},{query},{d}: difference {difference}");
            }
        }
    }
    Ok(())
}

#[test]
fn strided_batched_rejects_an_operand_past_its_buffer() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let qkv = copy_device(&context, &stream, &pool, &vec![0.0_f32; LENGTH * WIDTH])?;
    let mut scores =
        copy_device(&context, &stream, &pool, &vec![0.0_f32; HEADS * LENGTH * LENGTH])?;
    let offsets = CublasGemmOffsets {
        left: 0,
        right: 3 * HEADS * DIM,
        output: 0,
    };
    let result = CublasGemmPlan::<f32, f32>::new(&context, &stream, scores_spec()?)?.execute(
        &stream,
        (&qkv, &qkv, &mut scores),
        offsets,
        (1.0, 0.0),
    );
    assert!(result.is_err());
    Ok(())
}
