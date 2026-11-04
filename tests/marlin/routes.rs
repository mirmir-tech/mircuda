use mircuda::{Driver, MarlinNvFp4MoeSpec, MarlinNvFp4ThreadConfig, bf16};

use super::{copy_device, read_device};

/// Rows of one expert must not depend on scheduling: Marlin's results for a
/// row depend on its place in a tile, so each expert's rows are ordered by
/// assignment.
#[test]
fn routes_order_each_experts_rows_by_assignment() -> mircuda::Result<()> {
    const TOKENS: usize = 2_400;
    const TOP_K: usize = 8;
    const EXPERTS: usize = 16;
    const BLOCK: usize = 8;
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let assignments = TOKENS * TOP_K;
    let choices = (0..assignments)
        .map(|index| u32::try_from((index * 7 + index / 5) % EXPERTS).unwrap_or_default())
        .collect::<Vec<_>>();
    let selected = copy_device(&context, &stream, &pool, &choices)?;
    let routing = copy_device(&context, &stream, &pool, &vec![bf16::from_f32(1.0); assignments])?;
    let capacity = assignments + EXPERTS * (BLOCK - 1);
    let mut sorted = pool.allocate(&stream, capacity)?;
    let mut expert_ids = pool.allocate(&stream, capacity.div_ceil(BLOCK))?;
    let mut padded = pool.allocate(&stream, 1)?;
    let mut offsets = pool.allocate(&stream, EXPERTS)?;
    let mut routing_f32 = pool.allocate(&stream, assignments)?;
    let spec = MarlinNvFp4MoeSpec::new(
        EXPERTS,
        TOKENS,
        TOP_K,
        128,
        128,
        MarlinNvFp4ThreadConfig::N128K128,
    )?;
    context.marlin_prepare_moe_routes(
        &stream, spec, &selected, &routing, &mut sorted, &mut expert_ids, &mut padded,
        &mut offsets, &mut routing_f32,
    )?;
    let sorted: Vec<i32> = read_device(&context, &stream, &sorted)?;
    let offsets: Vec<i32> = read_device(&context, &stream, &offsets)?;
    for (expert, offset) in offsets.iter().enumerate() {
        let expected = (0..assignments)
            .filter(|&index| choices[index] as usize == expert)
            .map(|index| i32::try_from(index).unwrap_or_default())
            .collect::<Vec<_>>();
        let start = usize::try_from(*offset).unwrap_or_default();
        assert_eq!(sorted[start..start + expected.len()], expected[..], "expert {expert}");
    }
    Ok(())
}
