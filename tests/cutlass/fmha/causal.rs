use mircuda::{FmhaBf16Plan, FmhaBf16Spec, bf16};

use super::{
    copy_device, environment,
    paged::{maximum_error, paged_bytes},
    read_device,
    reference::{ReferenceShape, attention},
    values,
};

#[test]
fn causal_paged_prefill_includes_its_diagonal_for_all_head_sizes() -> mircuda::Result<()> {
    const QUERY_HEADS: usize = 12;
    const KV_HEADS: usize = 2;
    let (context, stream, pool) = environment()?;
    for head_dim in [64, 128, 256] {
        for query_lengths in [[2_usize, 3], [17, 20]] {
            let context_lengths = [17_usize, 20];
            let total_query = query_lengths.iter().sum();
            let query = values(total_query, QUERY_HEADS, head_dim, 17);
            let keys = values(37, KV_HEADS, head_dim, 23);
            let value = values(37, KV_HEADS, head_dim, 31);
            let starts = [0_u32, u32::try_from(query_lengths[0])?, u32::try_from(total_query)?];
            let query_starts = copy_device(&context, &stream, &pool, &starts)?;
            let context_starts = copy_device(&context, &stream, &pool, &[0_u32, 17, 37])?;
            let token_counts = copy_device(&context, &stream, &pool, &[17_u32, 20])?;
            let block_table = copy_device(&context, &stream, &pool, &[2_u32, 0, 3, 1])?;
            let q = copy_device(&context, &stream, &pool, &query)?;
            let k = copy_device(
                &context,
                &stream,
                &pool,
                &paged_bytes(&keys, context_lengths, [2, 0, 3, 1], head_dim),
            )?;
            let v = copy_device(
                &context,
                &stream,
                &pool,
                &paged_bytes(&value, context_lengths, [2, 0, 3, 1], head_dim),
            )?;
            let mut output = pool.allocate_zeroed::<bf16>(&stream, query.len())?;
            let mut lse = pool.allocate_zeroed::<f32>(&stream, total_query * QUERY_HEADS)?;
            let plan = FmhaBf16Plan::new(
                &context,
                &stream,
                FmhaBf16Spec::new(QUERY_HEADS, KV_HEADS, head_dim, head_dim)?,
            )?;
            let scale = f32::from(u16::try_from(head_dim)?).sqrt().recip();
            plan.execute_paged_varlen(
                &stream, &q, &k, &v, &mut output, &query_starts, &token_counts, &context_starts,
                &block_table, &mut lse, 2, total_query, query_lengths[1], 20, 2, 16, scale,
            )?;
            let expected = attention(
                &query,
                &keys,
                &value,
                ReferenceShape {
                    query_lengths,
                    context_lengths,
                    query_heads: QUERY_HEADS,
                    kv_heads: KV_HEADS,
                    head_dim,
                    scale,
                },
            );
            let actual = read_device(&context, &stream, &output)?;
            let error = maximum_error(&expected, &actual);
            assert!(
                error <= 0.001_953_125,
                "head={head_dim} queries={query_lengths:?} error={error}"
            );
        }
    }
    Ok(())
}
