use mircuda::{FmhaBf16Plan, FmhaBf16Spec, bf16};

use super::{
    KV_HEADS, QUERY_HEADS, copy_device, environment,
    paged::{maximum_error, paged_bytes},
    read_device, values,
};

const HEAD_DIM: usize = 64;
const PAGE_SIZE: usize = 16;

#[test]
#[ignore = "single-token CUTLASS GQA compaction currently collapses KV head groups"]
fn single_token_supports_qwen2_seven_to_one_gqa() -> mircuda::Result<()> {
    const QUERY_HEADS: usize = 14;
    const KV_HEADS: usize = 2;
    let (context, stream, pool) = environment()?;
    let query = values(1, QUERY_HEADS, HEAD_DIM, 17);
    let keys = values(1, KV_HEADS, HEAD_DIM, 23);
    let values = values(1, KV_HEADS, HEAD_DIM, 31);
    let query = copy_device(&context, &stream, &pool, &query)?;
    let keys =
        copy_device(&context, &stream, &pool, &paged_bytes(&keys, [1, 0], [0; 4], HEAD_DIM))?;
    let value_bytes = paged_bytes(&values, [1, 0], [0; 4], HEAD_DIM);
    let value_device = copy_device(&context, &stream, &pool, &value_bytes)?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, QUERY_HEADS * HEAD_DIM)?;
    let plan = FmhaBf16Plan::new(
        &context,
        &stream,
        FmhaBf16Spec::new(QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_DIM)?,
    )?;
    plan.execute(
        &stream,
        &query,
        &keys,
        &value_device,
        &mut output,
        1,
        1,
        0,
        64.0_f32.sqrt().recip(),
    )?;
    let actual = read_device(&context, &stream, &output)?;
    for head in 0..QUERY_HEADS {
        let expected = &values[(head / 7) * HEAD_DIM..(head / 7 + 1) * HEAD_DIM];
        assert_eq!(&actual[head * HEAD_DIM..(head + 1) * HEAD_DIM], expected);
    }
    Ok(())
}

#[test]
fn paged_varlen_head64_matches_contiguous_varlen() -> mircuda::Result<()> {
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
    let mut expected = pool.allocate_zeroed::<bf16>(&stream, query.len())?;
    let mut actual = pool.allocate_zeroed::<bf16>(&stream, query.len())?;
    let mut softmax_lse = pool.allocate_zeroed::<f32>(&stream, total_query * QUERY_HEADS)?;
    let plan = FmhaBf16Plan::new(
        &context,
        &stream,
        FmhaBf16Spec::new(QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_DIM)?,
    )?;
    let scale = 64.0_f32.sqrt().recip();
    plan.execute_varlen(
        &stream, &query_device, &key_device, &value_device, &mut expected, &query_starts,
        &key_starts, 2, total_query, total_context, 3, 20, scale,
    )?;
    plan.execute_paged_varlen(
        &stream, &query_device, &key_pages, &value_pages, &mut actual, &query_starts,
        &token_counts, &key_starts, &block_table, &mut softmax_lse, 2, total_query, 3, 20, 2,
        PAGE_SIZE, scale,
    )?;
    let expected = read_device(&context, &stream, &expected)?;
    let actual = read_device(&context, &stream, &actual)?;
    let error = maximum_error(&expected, &actual);
    assert!(error <= 0.031_25, "maximum head-dim 64 BF16 difference: {error}");
    Ok(())
}
