use mircuda::{PaddedAttentionElement, PaddedAttentionPlan, bf16};

use super::{copy_device, environment, read_device};

const LENGTH: usize = 40;
const HEADS: usize = 3;
const LENGTHS: [u32; 2] = [40, 23];
/// Out-of-range lengths the kernel clamps to `[1, LENGTH]`.
const CLAMPED: [u32; 3] = [0, 41, u32::MAX];

fn pattern(count: usize) -> mircuda::Result<Vec<f32>> {
    (0..count)
        .map(|index| {
            let step = u32::try_from(index)?.wrapping_mul(2_654_435_761);
            Ok(f32::from(u16::try_from(step % 10_007)?) / 5_003.5 - 1.0)
        })
        .collect()
}

/// Reference attention of every padded query over its sequence's valid keys,
/// with each length clamped to `[1, LENGTH]` as the kernel advertises.
fn reference(qkv: &[f64], scale: f64, dim: usize, lengths: &[u32]) -> Vec<f64> {
    let hidden = HEADS * dim;
    let mut output = vec![0.0; lengths.len() * LENGTH * hidden];
    for (sequence, &valid) in lengths.iter().enumerate() {
        let row = |token: usize, part: usize, head: usize| {
            ((sequence * LENGTH + token) * 3 + part) * hidden + head * dim
        };
        let keys = usize::try_from(valid).unwrap_or(LENGTH).clamp(1, LENGTH);
        for head in 0..HEADS {
            for query in 0..LENGTH {
                let scores: Vec<f64> = (0..keys)
                    .map(|key| {
                        (0..dim)
                            .map(|d| qkv[row(query, 0, head) + d] * qkv[row(key, 1, head) + d])
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> =
                    scores.iter().map(|score| (score - maximum).exp()).collect();
                let total: f64 = weights.iter().sum();
                for d in 0..dim {
                    let mixed: f64 = weights
                        .iter()
                        .enumerate()
                        .map(|(key, weight)| weight * qkv[row(key, 2, head) + d])
                        .sum();
                    output[(sequence * LENGTH + query) * hidden + head * dim + d] = mixed / total;
                }
            }
        }
    }
    output
}

fn check<T: PaddedAttentionElement>(
    (dim, lengths): (usize, &[u32]),
    round: impl Fn(f32) -> (T, f64),
    widen: impl Fn(T) -> f64,
    tolerance: f64,
) -> mircuda::Result<()> {
    let hidden = HEADS * dim;
    let (context, stream, pool) = environment()?;
    let (input, rounded): (Vec<T>, Vec<f64>) =
        pattern(lengths.len() * LENGTH * 3 * hidden)?.into_iter().map(round).unzip();
    let qkv = copy_device(&context, &stream, &pool, &input)?;
    let lengths_device = copy_device(&context, &stream, &pool, lengths)?;
    let mut output =
        copy_device(&context, &stream, &pool, &vec![input[0]; lengths.len() * LENGTH * hidden])?;
    let scale = 0.125;
    PaddedAttentionPlan::<T>::new(&context, &stream, HEADS, dim)?.execute(
        &stream,
        (&qkv, &lengths_device),
        LENGTH,
        scale,
        &mut output,
    )?;
    let actual = read_device(&context, &stream, &output)?;
    let expected = reference(&rounded, 0.125, dim, lengths);
    for (index, (got, want)) in actual.into_iter().zip(expected).enumerate() {
        let difference = (widen(got) - want).abs();
        assert!(difference < tolerance, "dim {dim}, {index}: difference {difference}");
    }
    Ok(())
}

fn check_f32(case: (usize, &[u32])) -> mircuda::Result<()> {
    check(case, |value| (value, f64::from(value)), f64::from, 1e-5)
}

fn check_bf16(case: (usize, &[u32])) -> mircuda::Result<()> {
    check(
        case,
        |value| (bf16::from_f32(value), bf16::from_f32(value).to_f64()),
        bf16::to_f64,
        1e-2,
    )
}

#[test]
fn padded_attention_matches_a_reference_in_f32() -> mircuda::Result<()> {
    check_f32((64, &LENGTHS))?;
    check_f32((128, &LENGTHS))
}

#[test]
fn padded_attention_matches_a_reference_in_bf16() -> mircuda::Result<()> {
    check_bf16((64, &LENGTHS))?;
    check_bf16((128, &LENGTHS))
}

#[test]
fn padded_attention_clamps_out_of_range_lengths() -> mircuda::Result<()> {
    for dim in [64, 128] {
        check_f32((dim, &CLAMPED))?;
        check_bf16((dim, &CLAMPED))?;
    }
    Ok(())
}
