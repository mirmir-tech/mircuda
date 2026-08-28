use mircuda::{FmhaBf16Plan, FmhaBf16Spec, bf16};

use super::{copy_device, environment, paged::maximum_error, read_device, values};

const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const PAGE_SIZE: usize = 16;

#[test]
fn paged_head256_decode_matches_reference() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let lengths = [17_usize, 20];
    let query = values(2, QUERY_HEADS, HEAD_DIM, 17);
    let keys = values(lengths.iter().sum(), KV_HEADS, HEAD_DIM, 23);
    let value = values(lengths.iter().sum(), KV_HEADS, HEAD_DIM, 31);
    let physical = [2_usize, 0, 3, 1];
    let query_starts = copy_device(&context, &stream, &pool, &[0_u32, 1, 2])?;
    let context_starts = copy_device(&context, &stream, &pool, &[0_u32, 17, 37])?;
    let token_counts = copy_device(&context, &stream, &pool, &[17_u32, 20])?;
    let block_table = copy_device(&context, &stream, &pool, &[2_u32, 0, 3, 1])?;
    let query_device = copy_device(&context, &stream, &pool, &query)?;
    let key_pages = copy_device(&context, &stream, &pool, &pages(&keys, lengths, physical))?;
    let value_pages = copy_device(&context, &stream, &pool, &pages(&value, lengths, physical))?;
    let mut actual = pool.allocate_zeroed::<bf16>(&stream, query.len())?;
    let mut lse = pool.allocate_zeroed::<f32>(&stream, 2 * QUERY_HEADS)?;
    let plan = FmhaBf16Plan::new(
        &context,
        &stream,
        FmhaBf16Spec::new(QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_DIM)?,
    )?;
    let scale = 0.0625_f32;
    plan.execute_paged_varlen(
        &stream, &query_device, &key_pages, &value_pages, &mut actual, &query_starts,
        &token_counts, &context_starts, &block_table, &mut lse, 2, 2, 1, 20, 2, PAGE_SIZE, scale,
    )?;
    let expected = reference(&query, &keys, &value, lengths, scale);
    let actual = read_device(&context, &stream, &actual)?;
    let error = maximum_error(&expected, &actual);
    assert!(error <= 0.031_25, "maximum head-dim 256 BF16 difference: {error}");
    for splits in [2, 4, 8, 16, 32] {
        let mut split_output = pool.allocate_zeroed::<bf16>(&stream, query.len())?;
        let mut split_lse = pool.allocate_zeroed::<f32>(&stream, 2 * QUERY_HEADS)?;
        let mut output_accum = pool.allocate_zeroed::<f32>(&stream, splits * query.len())?;
        let mut lse_accum = pool.allocate_zeroed::<f32>(&stream, splits * 2 * QUERY_HEADS)?;
        plan.execute_paged_varlen_split(
            &stream,
            &query_device,
            &key_pages,
            &value_pages,
            &mut split_output,
            &query_starts,
            &token_counts,
            &context_starts,
            &block_table,
            &mut split_lse,
            &mut output_accum,
            &mut lse_accum,
            splits,
            2,
            2,
            1,
            20,
            2,
            PAGE_SIZE,
            scale,
        )?;
        let split_output = read_device(&context, &stream, &split_output)?;
        let error = maximum_error(&expected, &split_output);
        assert!(error <= 0.031_25, "maximum {splits}-split BF16 difference: {error}");
    }
    Ok(())
}

fn reference(
    query: &[bf16],
    keys: &[bf16],
    values: &[bf16],
    lengths: [usize; 2],
    scale: f32,
) -> Vec<bf16> {
    let mut output = Vec::with_capacity(query.len());
    let mut context_start = 0;
    for (batch, length) in lengths.into_iter().enumerate() {
        for query_head in 0..QUERY_HEADS {
            let kv_head = query_head / (QUERY_HEADS / KV_HEADS);
            let query_start = (batch * QUERY_HEADS + query_head) * HEAD_DIM;
            let mut scores = Vec::with_capacity(length);
            for token in 0..length {
                let key_start = ((context_start + token) * KV_HEADS + kv_head) * HEAD_DIM;
                let score = (0..HEAD_DIM)
                    .map(|item| {
                        query[query_start + item].to_f32() * keys[key_start + item].to_f32()
                    })
                    .sum::<f32>()
                    * scale;
                scores.push(score);
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores.iter().map(|score| (score - maximum).exp()).sum::<f32>();
            for item in 0..HEAD_DIM {
                let result = scores
                    .iter()
                    .enumerate()
                    .map(|(token, score)| {
                        let value_start = ((context_start + token) * KV_HEADS + kv_head) * HEAD_DIM;
                        (score - maximum).exp() / denominator * values[value_start + item].to_f32()
                    })
                    .sum::<f32>();
                output.push(bf16::from_f32(result));
            }
        }
        context_start += length;
    }
    output
}

fn pages(packed: &[bf16], lengths: [usize; 2], physical: [usize; 4]) -> Vec<u8> {
    let row = KV_HEADS * HEAD_DIM;
    let mut pages = vec![bf16::from_f32(0.0); physical.len() * PAGE_SIZE * row];
    let mut packed_token = 0;
    let mut table = 0;
    for length in lengths {
        for token in 0..length {
            let block = physical[table + token / PAGE_SIZE];
            let destination = (block * PAGE_SIZE + token % PAGE_SIZE) * row;
            let source = (packed_token + token) * row;
            pages[destination..destination + row].copy_from_slice(&packed[source..source + row]);
        }
        packed_token += length;
        table += length.div_ceil(PAGE_SIZE);
    }
    pages.iter().flat_map(|value| value.to_bits().to_ne_bytes()).collect()
}
