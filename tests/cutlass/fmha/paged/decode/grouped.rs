use std::num::NonZeroUsize;

use mircuda::{Context, FmhaBf16Plan, FmhaBf16Spec, FmhaCausalWindow, MemoryPool, Stream, bf16};

use super::{
    super::{
        super::{bytes, copy_device, environment, read_device, values},
        maximum_error,
    },
    PAGE_SIZE,
};

const QUERY_HEADS: usize = 8;
const LENGTHS: [usize; 2] = [300, 517];

/// Windowed GQA decode packs each group's query heads into one K/V pass.
/// Replicating K/V per query head runs the same attention unpacked, so both
/// must agree for windows that cut through K/V blocks and splits.
#[test]
fn grouped_windowed_decode_matches_ungrouped_heads() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    for (kv_heads, head_dim) in [(2, 256), (4, 128), (1, 64)] {
        let shape = Shape { kv_heads, head_dim };
        for window in [128, 100, 1] {
            let window = FmhaCausalWindow::Sliding(
                NonZeroUsize::new(window).ok_or(mircuda::Error::InvalidLaunch)?,
            );
            for splits in [1, 3, 8] {
                let grouped =
                    attend(&context, &stream, &pool, shape, shape.kv_heads, window, splits)?;
                let ungrouped =
                    attend(&context, &stream, &pool, shape, QUERY_HEADS, window, splits)?;
                let error = maximum_error(&ungrouped, &grouped);
                assert!(
                    error <= 0.007_812_5,
                    "{kv_heads} K/V heads, head dim {head_dim}, {window:?}, {splits} splits: \
                     grouped decode differs by {error}"
                );
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Shape {
    kv_heads: usize,
    head_dim: usize,
}

/// Decodes one query token per sequence with `stored_heads` K/V heads, where
/// a stored head `h` holds the K/V of logical head `h * kv_heads / stored_heads`.
fn attend(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    shape: Shape,
    stored_heads: usize,
    window: FmhaCausalWindow,
    splits: usize,
) -> mircuda::Result<Vec<bf16>> {
    let tokens = LENGTHS.iter().sum();
    let query = values(LENGTHS.len(), QUERY_HEADS, shape.head_dim, 19);
    let keys = replicate(&values(tokens, shape.kv_heads, shape.head_dim, 23), shape, stored_heads);
    let value = replicate(&values(tokens, shape.kv_heads, shape.head_dim, 29), shape, stored_heads);
    let blocks = LENGTHS.map(|length| length.div_ceil(PAGE_SIZE));
    let max_blocks = blocks.iter().copied().max().unwrap_or(0);
    let physical = (0..blocks.iter().sum::<usize>()).rev().collect::<Vec<_>>();
    let mut table = vec![0_u32; LENGTHS.len() * max_blocks];
    let mut next = physical.iter();
    for (sequence, count) in blocks.into_iter().enumerate() {
        for slot in &mut table[sequence * max_blocks..sequence * max_blocks + count] {
            *slot = u32::try_from(*next.next().ok_or(mircuda::Error::InvalidLaunch)?)?;
        }
    }
    let row = stored_heads * shape.head_dim;
    let page = |packed: &[bf16]| {
        let mut pages = vec![bf16::ZERO; physical.len() * PAGE_SIZE * row];
        let mut first = 0;
        for (sequence, length) in LENGTHS.into_iter().enumerate() {
            for token in 0..length {
                let block = usize::try_from(table[sequence * max_blocks + token / PAGE_SIZE])
                    .unwrap_or(usize::MAX);
                let destination = (block * PAGE_SIZE + token % PAGE_SIZE) * row;
                let source = (first + token) * row;
                pages[destination..destination + row]
                    .copy_from_slice(&packed[source..source + row]);
            }
            first += length;
        }
        bytes(&pages)
    };
    let [first, second] = LENGTHS.map(|length| u32::try_from(length).unwrap_or(u32::MAX));
    let query_starts = copy_device(context, stream, pool, &[0_u32, 1, 2])?;
    let context_starts = copy_device(context, stream, pool, &[0_u32, first, first + second])?;
    let token_counts = copy_device(context, stream, pool, &[first, second])?;
    let block_table = copy_device(context, stream, pool, &table)?;
    let query_device = copy_device(context, stream, pool, &query)?;
    let key_pages = copy_device(context, stream, pool, &page(&keys))?;
    let value_pages = copy_device(context, stream, pool, &page(&value))?;
    let mut output = pool.allocate_zeroed::<bf16>(stream, query.len())?;
    let rows = LENGTHS.len() * QUERY_HEADS;
    let mut lse = pool.allocate_zeroed::<f32>(stream, rows)?;
    let mut output_accum = pool.allocate_zeroed::<f32>(stream, splits * query.len())?;
    let mut lse_accum = pool.allocate_zeroed::<f32>(stream, splits * rows)?;
    let plan = FmhaBf16Plan::new(
        context,
        stream,
        FmhaBf16Spec::new(QUERY_HEADS, stored_heads, shape.head_dim, shape.head_dim)?,
    )?;
    let scale = f32::from(u16::try_from(shape.head_dim)?).sqrt().recip();
    let (rows, sequences) = (LENGTHS.len(), LENGTHS.len());
    if splits == 1 {
        plan.execute_paged_varlen_windowed(
            stream, &query_device, &key_pages, &value_pages, &mut output, &query_starts,
            &token_counts, &context_starts, &block_table, &mut lse, sequences, rows, 1, LENGTHS[1],
            max_blocks, PAGE_SIZE, window, scale,
        )?;
    } else {
        plan.execute_paged_varlen_split(
            stream,
            &query_device,
            &key_pages,
            &value_pages,
            &mut output,
            &query_starts,
            &token_counts,
            &context_starts,
            &block_table,
            &mut lse,
            &mut output_accum,
            &mut lse_accum,
            splits,
            sequences,
            rows,
            1,
            LENGTHS[1],
            max_blocks,
            PAGE_SIZE,
            window,
            scale,
        )?;
    }
    read_device(context, stream, &output)
}

fn replicate(packed: &[bf16], shape: Shape, stored_heads: usize) -> Vec<bf16> {
    let groups = stored_heads / shape.kv_heads;
    packed
        .chunks_exact(shape.head_dim)
        .flat_map(|head| std::iter::repeat_n(head, groups).flatten().copied())
        .collect()
}
