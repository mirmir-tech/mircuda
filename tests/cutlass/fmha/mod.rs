#![cfg(all(target_os = "linux", feature = "cutlass"))]

use mircuda::{
    Context, DeviceBuffer, DeviceElement, Driver, FmhaBf16Plan, FmhaBf16Spec, MemoryPool, Stream,
    bf16,
};

mod head64;
mod paged;

const QUERY_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;

#[test]
fn varlen_fmha_matches_independent_rows() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let query_lengths = [2_usize, 3];
    let context_lengths = [3_usize, 5];
    let query = values(query_lengths.iter().sum(), QUERY_HEADS, HEAD_DIM, 17);
    let keys = values(context_lengths.iter().sum(), KV_HEADS, HEAD_DIM, 23);
    let values = values(context_lengths.iter().sum(), KV_HEADS, HEAD_DIM, 31);
    let query_starts = copy_device(&context, &stream, &pool, &[0_u32, 2, 5])?;
    let key_starts = copy_device(&context, &stream, &pool, &[0_u32, 3, 8])?;
    let query_device = copy_device(&context, &stream, &pool, &query)?;
    let key_device = copy_device(&context, &stream, &pool, &keys)?;
    let value_device = copy_device(&context, &stream, &pool, &values)?;
    let mut actual = pool.allocate_zeroed::<bf16>(&stream, query.len())?;
    let plan = FmhaBf16Plan::new(
        &context,
        &stream,
        FmhaBf16Spec::new(QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_DIM)?,
    )?;
    plan.execute_varlen(
        &stream,
        &query_device,
        &key_device,
        &value_device,
        &mut actual,
        &query_starts,
        &key_starts,
        2,
        5,
        8,
        3,
        5,
        128.0_f32.sqrt().recip(),
    )?;

    let mut expected = Vec::new();
    let mut query_offset = 0;
    let mut context_offset = 0;
    for (query_tokens, context_tokens) in query_lengths.into_iter().zip(context_lengths) {
        let query_elements = query_tokens * QUERY_HEADS * HEAD_DIM;
        let context_elements = context_tokens * KV_HEADS * HEAD_DIM;
        let row_query = copy_device(
            &context,
            &stream,
            &pool,
            &query[query_offset..query_offset + query_elements],
        )?;
        let row_keys = copy_device(
            &context,
            &stream,
            &pool,
            &bytes(&keys[context_offset..context_offset + context_elements]),
        )?;
        let row_values = copy_device(
            &context,
            &stream,
            &pool,
            &bytes(&values[context_offset..context_offset + context_elements]),
        )?;
        let mut row_output = pool.allocate_zeroed::<bf16>(&stream, query_elements)?;
        plan.execute(
            &stream,
            &row_query,
            &row_keys,
            &row_values,
            &mut row_output,
            query_tokens,
            context_tokens,
            0,
            128.0_f32.sqrt().recip(),
        )?;
        expected.extend(read_device(&context, &stream, &row_output)?);
        query_offset += query_elements;
        context_offset += context_elements;
    }
    let actual = read_device(&context, &stream, &actual)?;
    let error = expected
        .iter()
        .zip(actual)
        .map(|(left, right)| (left.to_f32() - right.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(error <= 0.015_625, "maximum varlen BF16 difference: {error}");
    Ok(())
}

fn values(tokens: usize, heads: usize, head_dim: usize, modulus: u8) -> Vec<bf16> {
    (0..tokens * heads * head_dim)
        .map(|index| {
            let value = u8::try_from(index % usize::from(modulus)).unwrap_or(0);
            bf16::from_f32(f32::from(value) / f32::from(modulus) - 0.5)
        })
        .collect()
}

fn bytes(values: &[bf16]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_bits().to_ne_bytes()).collect()
}

fn environment() -> mircuda::Result<(Context, Stream, MemoryPool)> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    Ok((context, stream, pool))
}

fn copy_device<T: DeviceElement>(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    values: &[T],
) -> mircuda::Result<DeviceBuffer<T>> {
    let mut host = context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = pool.allocate::<T>(stream, values.len())?;
    stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read_device<T: DeviceElement>(
    context: &Context,
    stream: &Stream,
    device: &DeviceBuffer<T>,
) -> mircuda::Result<Vec<T>> {
    let mut host = context.allocate_pinned::<T>(device.len())?;
    stream.copy_to_host(device, &mut host)?;
    stream.synchronize()?;
    host.to_vec()
}
