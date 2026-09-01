use mircuda::{FmhaBf16Plan, FmhaBf16Spec, bf16};

use super::{
    HEAD_DIM, KV_HEADS, QUERY_HEADS, bytes, copy_device, environment, read_device, values,
};

const PAGE_SIZE: usize = 16;

#[test]
#[allow(clippy::too_many_lines)]
fn paged_varlen_fmha_matches_contiguous_varlen() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let query_lengths = [2_usize, 3];
    let context_lengths = [17_usize, 20];
    let total_query = query_lengths.iter().sum();
    let total_context = context_lengths.iter().sum();
    let query = values(total_query, QUERY_HEADS, HEAD_DIM, 17);
    let keys = values(total_context, KV_HEADS, HEAD_DIM, 23);
    let values = values(total_context, KV_HEADS, HEAD_DIM, 31);
    let query_starts = copy_device(&context, &stream, &pool, &[0_u32, 2, 5])?;
    let key_starts = copy_device(&context, &stream, &pool, &[0_u32, 17, 37])?;
    let token_counts = copy_device(&context, &stream, &pool, &[17_u32, 20])?;
    let block_table = copy_device(&context, &stream, &pool, &[2_u32, 0, 3, 1])?;
    let contiguous_table = copy_device(&context, &stream, &pool, &[0_u32, 1, 2, 3])?;
    let query_device = copy_device(&context, &stream, &pool, &query)?;
    let key_device = copy_device(&context, &stream, &pool, &keys)?;
    let value_device = copy_device(&context, &stream, &pool, &values)?;
    let key_pages = copy_device(
        &context,
        &stream,
        &pool,
        &paged_bytes(&keys, context_lengths, [2_usize, 0, 3, 1], HEAD_DIM),
    )?;
    let value_pages = copy_device(
        &context,
        &stream,
        &pool,
        &paged_bytes(&values, context_lengths, [2_usize, 0, 3, 1], HEAD_DIM),
    )?;
    let contiguous_key_pages = copy_device(
        &context,
        &stream,
        &pool,
        &paged_bytes(&keys, context_lengths, [0_usize, 1, 2, 3], HEAD_DIM),
    )?;
    let contiguous_value_pages = copy_device(
        &context,
        &stream,
        &pool,
        &paged_bytes(&values, context_lengths, [0_usize, 1, 2, 3], HEAD_DIM),
    )?;
    let mut expected = pool.allocate_zeroed::<bf16>(&stream, query.len())?;
    let mut actual = pool.allocate_zeroed::<bf16>(&stream, query.len())?;
    let mut contiguous_actual = pool.allocate_zeroed::<bf16>(&stream, query.len())?;
    let mut softmax_lse = pool.allocate_zeroed::<f32>(&stream, total_query * QUERY_HEADS)?;
    let plan = FmhaBf16Plan::new(
        &context,
        &stream,
        FmhaBf16Spec::new(QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_DIM)?,
    )?;
    let scale = 128.0_f32.sqrt().recip();
    plan.execute_varlen(
        &stream, &query_device, &key_device, &value_device, &mut expected, &query_starts,
        &key_starts, 2, total_query, total_context, 3, 20, scale,
    )?;
    plan.execute_paged_varlen(
        &stream, &query_device, &key_pages, &value_pages, &mut actual, &query_starts,
        &token_counts, &key_starts, &block_table, &mut softmax_lse, 2, total_query, 3, 20, 2,
        PAGE_SIZE, scale,
    )?;
    plan.execute_paged_varlen(
        &stream,
        &query_device,
        &contiguous_key_pages,
        &contiguous_value_pages,
        &mut contiguous_actual,
        &query_starts,
        &token_counts,
        &key_starts,
        &contiguous_table,
        &mut softmax_lse,
        2,
        total_query,
        3,
        20,
        2,
        PAGE_SIZE,
        scale,
    )?;
    let first_query_elements = query_lengths[0] * QUERY_HEADS * HEAD_DIM;
    let split_query = copy_device(&context, &stream, &pool, &query[..first_query_elements])?;
    let mut split_actual = pool.allocate_zeroed::<bf16>(&stream, first_query_elements)?;
    let mut split_lse = pool.allocate_zeroed::<f32>(&stream, query_lengths[0] * QUERY_HEADS)?;
    let mut output_accum = pool.allocate_zeroed::<f32>(&stream, first_query_elements * 2)?;
    let mut split_lse_accum =
        pool.allocate_zeroed::<f32>(&stream, query_lengths[0] * QUERY_HEADS * 2)?;
    plan.execute_paged_varlen_split(
        &stream,
        &split_query,
        &key_pages,
        &value_pages,
        &mut split_actual,
        &query_starts,
        &token_counts,
        &key_starts,
        &block_table,
        &mut split_lse,
        &mut output_accum,
        &mut split_lse_accum,
        2,
        1,
        query_lengths[0],
        query_lengths[0],
        context_lengths[0],
        2,
        PAGE_SIZE,
        scale,
    )?;
    let expected = read_device(&context, &stream, &expected)?;
    let actual = read_device(&context, &stream, &actual)?;
    let split_actual = read_device(&context, &stream, &split_actual)?;
    let contiguous_actual = read_device(&context, &stream, &contiguous_actual)?;
    let shuffled_error = maximum_error(&expected, &actual);
    let split_error = maximum_error(&expected[..first_query_elements], &split_actual);
    let contiguous_error = maximum_error(&expected, &contiguous_actual);
    assert!(shuffled_error <= 0.031_25, "maximum shuffled BF16 difference: {shuffled_error}");
    assert!(split_error <= 0.031_25, "maximum split BF16 difference: {split_error}");
    assert!(
        contiguous_error <= 0.031_25,
        "maximum contiguous BF16 difference: {contiguous_error}"
    );
    Ok(())
}

pub(super) fn maximum_error(expected: &[bf16], actual: &[bf16]) -> f32 {
    expected
        .iter()
        .zip(actual)
        .map(|(left, right)| (left.to_f32() - right.to_f32()).abs())
        .fold(0.0_f32, f32::max)
}

pub(super) fn paged_bytes(
    packed: &[bf16],
    lengths: [usize; 2],
    physical: [usize; 4],
    head_dim: usize,
) -> Vec<u8> {
    let row = KV_HEADS * head_dim;
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
    bytes(&pages)
}
