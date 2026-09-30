use mircuda::{PaddedAttentionElement, PaddedAttentionPlan, bf16};

use super::{copy_device, environment, read_device};

const LENGTH: usize = 40;
const HEADS: usize = 3;
const DIM: usize = 64;
const HIDDEN: usize = HEADS * DIM;
const LENGTHS: [u32; 2] = [40, 23];

fn pattern(count: usize) -> mircuda::Result<Vec<f32>> {
    (0..count)
        .map(|index| {
            let step = u32::try_from(index)?.wrapping_mul(2_654_435_761);
            Ok(f32::from(u16::try_from(step % 10_007)?) / 5_003.5 - 1.0)
        })
        .collect()
}

/// Reference attention of every padded query over its sequence's valid keys.
fn reference(qkv: &[f64], scale: f64) -> Vec<f64> {
    let mut output = vec![0.0; LENGTHS.len() * LENGTH * HIDDEN];
    for (sequence, &valid) in LENGTHS.iter().enumerate() {
        let row = |token: usize, part: usize, head: usize| {
            ((sequence * LENGTH + token) * 3 + part) * HIDDEN + head * DIM
        };
        let keys = usize::try_from(valid).unwrap_or(LENGTH);
        for head in 0..HEADS {
            for query in 0..LENGTH {
                let scores: Vec<f64> = (0..keys)
                    .map(|key| {
                        (0..DIM)
                            .map(|d| qkv[row(query, 0, head) + d] * qkv[row(key, 1, head) + d])
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> =
                    scores.iter().map(|score| (score - maximum).exp()).collect();
                let total: f64 = weights.iter().sum();
                for d in 0..DIM {
                    let mixed: f64 = weights
                        .iter()
                        .enumerate()
                        .map(|(key, weight)| weight * qkv[row(key, 2, head) + d])
                        .sum();
                    output[(sequence * LENGTH + query) * HIDDEN + head * DIM + d] = mixed / total;
                }
            }
        }
    }
    output
}

fn check<T: PaddedAttentionElement>(
    round: impl Fn(f32) -> (T, f64),
    widen: impl Fn(T) -> f64,
    tolerance: f64,
) -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let (input, rounded): (Vec<T>, Vec<f64>) =
        pattern(LENGTHS.len() * LENGTH * 3 * HIDDEN)?.into_iter().map(round).unzip();
    let qkv = copy_device(&context, &stream, &pool, &input)?;
    let lengths = copy_device(&context, &stream, &pool, &LENGTHS)?;
    let mut output =
        copy_device(&context, &stream, &pool, &vec![input[0]; LENGTHS.len() * LENGTH * HIDDEN])?;
    let scale = 0.125;
    PaddedAttentionPlan::<T>::new(&context, &stream, HEADS, DIM)?.execute(
        &stream,
        (&qkv, &lengths),
        LENGTH,
        scale,
        &mut output,
    )?;
    let actual = read_device(&context, &stream, &output)?;
    for (index, (got, want)) in actual.into_iter().zip(reference(&rounded, 0.125)).enumerate() {
        let difference = (widen(got) - want).abs();
        assert!(difference < tolerance, "{index}: difference {difference}");
    }
    Ok(())
}

#[test]
fn padded_attention_matches_a_reference_in_f32() -> mircuda::Result<()> {
    check(|value| (value, f64::from(value)), f64::from, 1e-5)
}

#[test]
fn padded_attention_matches_a_reference_in_bf16() -> mircuda::Result<()> {
    check(
        |value| (bf16::from_f32(value), bf16::from_f32(value).to_f64()),
        bf16::to_f64,
        1e-2,
    )
}
