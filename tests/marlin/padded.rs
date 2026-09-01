use mircuda::{
    Driver, MarlinMxFp4MoeOperands, MarlinMxFp4RepackSpec, MarlinNvFp4MoeSpec,
    MarlinNvFp4ThreadConfig, bf16,
};

use super::{EXPERTS, K, N, canonical_weights, copy_device, read_device, scale_destination};

const PADDED_N: usize = 128;
const PADDED_K: usize = 128;

#[test]
fn mxfp4_interleaved_repack_zero_pads_physical_gate_and_up_halves() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let spec = MarlinMxFp4RepackSpec::new(EXPERTS, PADDED_N, PADDED_K, N, K)?;

    let raw = canonical_weights();
    let input = copy_device(&context, &stream, &pool, &raw)?;
    let mut output = pool.allocate_zeroed::<u8>(&stream, EXPERTS * PADDED_N * PADDED_K / 2)?;
    context.marlin_repack_mxfp4(&stream, spec, &input, &mut output, true)?;
    assert_eq!(read_device(&context, &stream, &output)?, reference_weights(&raw));

    let scales = (0..EXPERTS * N * K / 32)
        .map(|index| u8::try_from(index % 251).unwrap_or_default())
        .collect::<Vec<_>>();
    let input = copy_device(&context, &stream, &pool, &scales)?;
    let mut output = pool.allocate_zeroed::<u8>(&stream, EXPERTS * PADDED_N * PADDED_K / 32)?;
    context.marlin_prepare_mxfp4_scales(&stream, spec, &input, &mut output, true)?;
    assert_eq!(read_device(&context, &stream, &output)?, reference_scales(&scales));
    Ok(())
}

#[test]
fn mxfp4_sequential_padded_projections_reset_workspace() -> mircuda::Result<()> {
    const TOKENS: usize = 2;
    const TOP_K: usize = 2;
    const ROUTED: usize = TOKENS * TOP_K;
    const ROUTED_EXPERTS: usize = 4;
    const PHYSICAL_N: usize = 256;
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let selected = copy_device(&context, &stream, &pool, &[0_u32, 1, 2, 3])?;
    let routing = copy_device(&context, &stream, &pool, &[bf16::from_f32(1.0); ROUTED])?;
    let capacity = ROUTED + ROUTED_EXPERTS * 7;
    let mut sorted = pool.allocate(&stream, capacity)?;
    let mut expert_ids = pool.allocate(&stream, capacity.div_ceil(8))?;
    let mut padded = pool.allocate(&stream, 1)?;
    let mut offsets = pool.allocate(&stream, ROUTED_EXPERTS)?;
    let mut routing_f32 = pool.allocate(&stream, ROUTED)?;
    let mut temporary = pool.allocate(&stream, capacity * PHYSICAL_N)?;
    let mut locks = pool.allocate_zeroed(&stream, 1_024)?;

    for (projection, physical_k) in [256, 128].into_iter().enumerate() {
        let repack = MarlinMxFp4RepackSpec::new(ROUTED_EXPERTS, PHYSICAL_N, physical_k, N, K)?;
        let raw =
            copy_device(&context, &stream, &pool, &vec![0x22_u8; ROUTED_EXPERTS * N * K / 2])?;
        let mut weight =
            pool.allocate_zeroed(&stream, ROUTED_EXPERTS * PHYSICAL_N * physical_k / 2)?;
        context.marlin_repack_mxfp4(&stream, repack, &raw, &mut weight, false)?;
        let scales = copy_device(&context, &stream, &pool, &[127_u8; ROUTED_EXPERTS * N * K / 32])?;
        let mut prepared =
            pool.allocate_zeroed(&stream, ROUTED_EXPERTS * PHYSICAL_N * physical_k / 32)?;
        context.marlin_prepare_mxfp4_scales(&stream, repack, &scales, &mut prepared, false)?;
        let rows = if projection == 0 {
            TOKENS
        } else {
            ROUTED
        };
        let mut input_values = vec![bf16::from_f32(0.0); rows * physical_k];
        for row in input_values.chunks_exact_mut(physical_k) {
            row[..K].fill(bf16::from_f32(1.0));
        }
        let input = copy_device(&context, &stream, &pool, &input_values)?;
        let mut output = pool.allocate_zeroed(&stream, ROUTED * PHYSICAL_N)?;
        let spec = MarlinNvFp4MoeSpec::new(
            ROUTED_EXPERTS,
            rows,
            if projection == 0 {
                TOP_K
            } else {
                1
            },
            PHYSICAL_N,
            physical_k,
            MarlinNvFp4ThreadConfig::N128K128,
        )?;
        if projection == 0 {
            context.marlin_prepare_moe_routes(
                &stream, spec, &selected, &routing, &mut sorted, &mut expert_ids, &mut padded,
                &mut offsets, &mut routing_f32,
            )?;
        }
        context.marlin_mxfp4_moe(
            &stream,
            spec,
            &MarlinMxFp4MoeOperands {
                input: &input,
                weight: &weight,
                scales: &prepared,
                routing: &routing_f32,
                sorted: &sorted,
                expert_ids: &expert_ids,
                padded: &padded,
                temporary: &mut temporary,
                locks: &mut locks,
                output: &mut output,
                multiply_routing: false,
                atomic_reduce: false,
            },
        )?;
        let actual = read_device(&context, &stream, &output)?;
        for row in actual.as_chunks::<PHYSICAL_N>().0 {
            assert!(row[..N].iter().all(|value| (value.to_f32() - 64.0).abs() <= 0.5));
            assert!(row[N..].iter().all(|value| value.to_f32().abs() <= 0.5));
        }
    }
    Ok(())
}

fn reference_weights(raw: &[u8]) -> Vec<u8> {
    let tiles_n = PADDED_N / 64;
    let tiles_k = PADDED_K / 16;
    let mut output = vec![0_u32; EXPERTS * tiles_n * tiles_k * 128];
    for expert in 0..EXPERTS {
        for tile_k in 0..tiles_k {
            for tile_n in 0..tiles_n {
                for lane in 0..32 {
                    for warp in 0..4 {
                        let n = tile_n * 64 + warp * 16 + lane / 4;
                        let k = tile_k * 16 + (lane % 4) * 2;
                        let values = [
                            value(raw, expert, n, k),
                            value(raw, expert, n, k + 1),
                            value(raw, expert, n, k + 8),
                            value(raw, expert, n, k + 9),
                            value(raw, expert, n + 8, k),
                            value(raw, expert, n + 8, k + 1),
                            value(raw, expert, n + 8, k + 8),
                            value(raw, expert, n + 8, k + 9),
                        ];
                        let order = [0, 2, 4, 6, 1, 3, 5, 7];
                        let packed =
                            order.iter().enumerate().fold(0_u32, |word, (shift, index)| {
                                word | values[*index] << (shift * 4)
                            });
                        let tile = (expert * tiles_k + tile_k) * tiles_n + tile_n;
                        output[tile * 128 + lane * 4 + warp] = packed;
                    }
                }
            }
        }
    }
    output.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn reference_scales(raw: &[u8]) -> Vec<u8> {
    let groups = PADDED_K / 32;
    let source_groups = K / 32;
    let per_expert = PADDED_N * groups;
    let mut output = vec![127; EXPERTS * per_expert];
    for expert in 0..EXPERTS {
        for group in 0..source_groups {
            for n in 0..PADDED_N {
                let Some(source_n) = source_row(n) else {
                    continue;
                };
                let logical = group * PADDED_N + n;
                let source = (expert * N + source_n) * source_groups + group;
                output[expert * per_expert + scale_destination(logical)] = raw[source];
            }
        }
    }
    output
}

fn value(raw: &[u8], expert: usize, n: usize, k: usize) -> u32 {
    let Some(source_n) = source_row(n) else {
        return 0;
    };
    if k >= K {
        return 0;
    }
    let packed = raw[(expert * N + source_n) * (K / 2) + k / 2];
    u32::from((packed >> ((k & 1) * 4)) & 15)
}

fn source_row(n: usize) -> Option<usize> {
    let physical = PADDED_N / 2;
    let logical = N / 2;
    let (unit, up) = if n < physical {
        (n, 0)
    } else {
        (n - physical, 1)
    };
    (unit < logical).then_some(unit * 2 + up)
}
