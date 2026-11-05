use mircuda::{FmhaBf16Plan, FmhaBf16Spec, bf16};

use super::{
    bytes, copy_device, environment,
    paged::maximum_error,
    read_device,
    reference::{ReferenceShape, attention},
    values,
};

const QUERY_HEADS: usize = 8;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 512;

#[test]
fn head512_accumulates_causal_attention_across_key_tiles() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let query_lengths = [17_usize, 70];
    let context_lengths = [80_usize, 150];
    let query = values(query_lengths.iter().sum(), QUERY_HEADS, HEAD_DIM, 17);
    let keys = values(context_lengths.iter().sum(), KV_HEADS, HEAD_DIM, 23);
    let value = values(context_lengths.iter().sum(), KV_HEADS, HEAD_DIM, 31);
    let spec = FmhaBf16Spec::new(QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_DIM)?;
    let plan = FmhaBf16Plan::new(&context, &stream, spec)?;
    let scale = 512.0_f32.sqrt().recip();
    let query_row = QUERY_HEADS * HEAD_DIM;
    let key_row = KV_HEADS * HEAD_DIM;
    let mut actual = Vec::with_capacity(query.len());
    let (mut query_start, mut key_start) = (0, 0);
    for (queries, tokens) in query_lengths.into_iter().zip(context_lengths) {
        let q = copy_device(
            &context,
            &stream,
            &pool,
            &query[query_start * query_row..(query_start + queries) * query_row],
        )?;
        let k = copy_device(
            &context,
            &stream,
            &pool,
            &bytes(&keys[key_start * key_row..(key_start + tokens) * key_row]),
        )?;
        let v = copy_device(
            &context,
            &stream,
            &pool,
            &bytes(&value[key_start * key_row..(key_start + tokens) * key_row]),
        )?;
        let mut output = pool.allocate_zeroed::<bf16>(&stream, queries * query_row)?;
        assert!(
            plan.execute(&stream, &q, &k, &v, &mut output, queries, tokens, 0, scale)
                .is_err()
        );
        let mut accumulator =
            pool.allocate_zeroed::<f32>(&stream, spec.accumulator_elements(queries))?;
        plan.execute_accumulated(
            &stream, &q, &k, &v, &mut output, &mut accumulator, queries, tokens, 0, scale,
        )?;
        actual.extend(read_device(&context, &stream, &output)?);
        query_start += queries;
        key_start += tokens;
    }
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
    let error = maximum_error(&expected, &actual);
    assert!(error <= 0.001_953_125, "head-dim 512 error={error}");
    Ok(())
}
