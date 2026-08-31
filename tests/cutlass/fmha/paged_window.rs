use mircuda::{FmhaBf16Plan, FmhaBf16Spec, FmhaCausalWindow, bf16};

use super::{
    HEAD_DIM, KV_HEADS, QUERY_HEADS, bytes, copy_device, environment, read_device, values,
};

const PAGE_SIZE: usize = 16;
const TOKENS: usize = 17;
const WINDOW: usize = 4;

#[test]
fn paged_varlen_fmha_honors_sliding_causal_window() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let query = values(TOKENS, QUERY_HEADS, HEAD_DIM, 17);
    let keys = values(TOKENS, KV_HEADS, HEAD_DIM, 23);
    let values = values(TOKENS, KV_HEADS, HEAD_DIM, 31);
    let tokens = u32::try_from(TOKENS)?;
    let query_starts = copy_device(&context, &stream, &pool, &[0_u32, tokens])?;
    let token_counts = copy_device(&context, &stream, &pool, &[tokens])?;
    let context_starts = copy_device(&context, &stream, &pool, &[0_u32, tokens])?;
    let block_table = copy_device(&context, &stream, &pool, &[1_u32, 0])?;
    let query_device = copy_device(&context, &stream, &pool, &query)?;
    let key_pages = copy_device(&context, &stream, &pool, &paged(&keys))?;
    let value_pages = copy_device(&context, &stream, &pool, &paged(&values))?;
    let mut actual = pool.allocate_zeroed::<bf16>(&stream, query.len())?;
    let mut softmax_lse = pool.allocate_zeroed::<f32>(&stream, TOKENS * QUERY_HEADS)?;
    let plan = FmhaBf16Plan::new(
        &context,
        &stream,
        FmhaBf16Spec::new(QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_DIM)?,
    )?;
    let scale = 128.0_f32.sqrt().recip();
    plan.execute_paged_varlen_windowed(
        &stream,
        &query_device,
        &key_pages,
        &value_pages,
        &mut actual,
        &query_starts,
        &token_counts,
        &context_starts,
        &block_table,
        &mut softmax_lse,
        1,
        TOKENS,
        TOKENS,
        TOKENS,
        2,
        PAGE_SIZE,
        FmhaCausalWindow::sliding(WINDOW)?,
        scale,
    )?;
    let actual = read_device(&context, &stream, &actual)?;
    let expected = reference(&query, &keys, &values, scale);
    let error = expected
        .iter()
        .zip(actual)
        .map(|(left, right)| (left.to_f32() - right.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(error <= 0.031_25, "maximum sliding BF16 difference: {error}");
    Ok(())
}

fn reference(query: &[bf16], keys: &[bf16], values: &[bf16], scale: f32) -> Vec<bf16> {
    let mut output = vec![bf16::from_f32(0.0); query.len()];
    for token in 0..TOKENS {
        let first = (token + 1).saturating_sub(WINDOW);
        for query_head in 0..QUERY_HEADS {
            let kv_head = query_head / (QUERY_HEADS / KV_HEADS);
            let scores = (first..=token)
                .map(|key_token| dot(query, keys, token, query_head, key_token, kv_head) * scale)
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores.iter().map(|score| (score - maximum).exp()).collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f32>();
            for dim in 0..HEAD_DIM {
                let value = weights
                    .iter()
                    .zip(first..=token)
                    .map(|(weight, key_token)| {
                        weight * values[(key_token * KV_HEADS + kv_head) * HEAD_DIM + dim].to_f32()
                    })
                    .sum::<f32>();
                output[(token * QUERY_HEADS + query_head) * HEAD_DIM + dim] =
                    bf16::from_f32(value / denominator);
            }
        }
    }
    output
}

fn dot(
    query: &[bf16],
    keys: &[bf16],
    query_token: usize,
    query_head: usize,
    key_token: usize,
    kv_head: usize,
) -> f32 {
    (0..HEAD_DIM)
        .map(|dim| {
            query[(query_token * QUERY_HEADS + query_head) * HEAD_DIM + dim].to_f32()
                * keys[(key_token * KV_HEADS + kv_head) * HEAD_DIM + dim].to_f32()
        })
        .sum()
}

fn paged(packed: &[bf16]) -> Vec<u8> {
    let row = KV_HEADS * HEAD_DIM;
    let mut pages = vec![bf16::from_f32(0.0); 2 * PAGE_SIZE * row];
    for token in 0..TOKENS {
        let block = [1_usize, 0][token / PAGE_SIZE];
        let destination = (block * PAGE_SIZE + token % PAGE_SIZE) * row;
        let source = token * row;
        pages[destination..destination + row].copy_from_slice(&packed[source..source + row]);
    }
    bytes(&pages)
}
