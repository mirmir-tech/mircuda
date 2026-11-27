use mircuda::{FmhaBf16Plan, FmhaBf16Spec, FmhaCausalWindow, bf16};

use super::{
    super::{
        HEAD_DIM, KV_HEADS, QUERY_HEADS, copy_device, environment, read_device,
        reference::{ReferenceShape, attention},
        values,
    },
    PAGE_SIZE, maximum_error, paged_bytes,
};

/// Split-KV merges each sequence's partial rows by sequence index, so a
/// prefill batch whose sequences hold equal query counts places the second
/// sequence after all rows of the first, not one row after its start.
#[test]
fn split_prefill_places_every_sequence_of_a_dense_batch() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let query_lengths = [3_usize, 3];
    let context_lengths = [17_usize, 20];
    let total_query = query_lengths.iter().sum();
    let query = values(total_query, QUERY_HEADS, HEAD_DIM, 17);
    let keys = values(context_lengths.iter().sum(), KV_HEADS, HEAD_DIM, 23);
    let value = values(context_lengths.iter().sum(), KV_HEADS, HEAD_DIM, 31);
    let physical = [2_usize, 0, 3, 1];
    let query_starts = copy_device(&context, &stream, &pool, &[0_u32, 3, 6])?;
    let key_starts = copy_device(&context, &stream, &pool, &[0_u32, 17, 37])?;
    let token_counts = copy_device(&context, &stream, &pool, &[17_u32, 20])?;
    let block_table = copy_device(&context, &stream, &pool, &[2_u32, 0, 3, 1])?;
    let query_device = copy_device(&context, &stream, &pool, &query)?;
    let key_pages = copy_device(
        &context,
        &stream,
        &pool,
        &paged_bytes(&keys, context_lengths, physical, HEAD_DIM),
    )?;
    let value_pages = copy_device(
        &context,
        &stream,
        &pool,
        &paged_bytes(&value, context_lengths, physical, HEAD_DIM),
    )?;
    let plan = FmhaBf16Plan::new(
        &context,
        &stream,
        FmhaBf16Spec::new(QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_DIM)?,
    )?;
    let scale = 128.0_f32.sqrt().recip();
    let expected = attention(
        &query,
        &keys,
        &value,
        ReferenceShape {
            query_lengths,
            context_lengths,
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            scale,
        },
    );
    for splits in [2, 3] {
        let mut output = pool.allocate_zeroed::<bf16>(&stream, query.len())?;
        let mut lse = pool.allocate_zeroed::<f32>(&stream, total_query * QUERY_HEADS)?;
        let mut output_accum = pool.allocate_zeroed::<f32>(&stream, splits * query.len())?;
        let mut lse_accum =
            pool.allocate_zeroed::<f32>(&stream, splits * total_query * QUERY_HEADS)?;
        plan.execute_paged_varlen_split(
            &stream,
            &query_device,
            &key_pages,
            &value_pages,
            &mut output,
            &query_starts,
            &token_counts,
            &key_starts,
            &block_table,
            &mut lse,
            &mut output_accum,
            &mut lse_accum,
            splits,
            2,
            total_query,
            3,
            20,
            2,
            PAGE_SIZE,
            FmhaCausalWindow::Full,
            scale,
        )?;
        let error = maximum_error(&expected, &read_device(&context, &stream, &output)?);
        assert!(error <= 0.031_25, "{splits}-split dense prefill differs by {error}");
    }
    Ok(())
}
