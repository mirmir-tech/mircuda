#![cfg(all(feature = "marlin", target_os = "linux"))]

use mircuda::{
    Context, DeviceBuffer, DeviceElement, Driver, MarlinMxFp4MoeOperands, MarlinMxFp4RepackSpec,
    MarlinNvFp4MoeSpec, MarlinNvFp4RepackSpec, MarlinNvFp4ThreadConfig, MemoryPool, Stream, bf16,
};

const EXPERTS: usize = 2;
const N: usize = 64;
const K: usize = 64;

#[path = "marlin/padded.rs"]
mod padded;

#[test]
fn nvfp4_repack_matches_marlin_tile_layout() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let raw = canonical_weights();
    let input = copy_device(&context, &stream, &pool, &raw)?;
    let mut output = pool.allocate_zeroed::<u8>(&stream, raw.len())?;
    let spec = MarlinNvFp4RepackSpec::new(EXPERTS, N, K)?;
    context.marlin_repack_nvfp4(&stream, spec, &input, &mut output)?;
    let actual = read_device(&context, &stream, &output)?;
    assert_eq!(actual, reference_repack(&raw));
    Ok(())
}

#[test]
fn mxfp4_interleaved_weights_and_scales_match_marlin_layout() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let spec = MarlinMxFp4RepackSpec::new(EXPERTS, N, K, N, K)?;

    let raw = canonical_weights();
    let input = copy_device(&context, &stream, &pool, &raw)?;
    let mut output = pool.allocate_zeroed::<u8>(&stream, raw.len())?;
    context.marlin_repack_mxfp4(&stream, spec, &input, &mut output, true)?;
    assert_eq!(
        read_device(&context, &stream, &output)?,
        reference_repack(&concatenated_gate_up(&raw)),
    );

    let scales = (0..EXPERTS * N * K / 32)
        .map(|index| u8::try_from(index % 251).unwrap_or_default())
        .collect::<Vec<_>>();
    let input = copy_device(&context, &stream, &pool, &scales)?;
    let mut output = pool.allocate_zeroed::<u8>(&stream, scales.len())?;
    context.marlin_prepare_mxfp4_scales(&stream, spec, &input, &mut output, true)?;
    assert_eq!(read_device(&context, &stream, &output)?, reference_scales(&scales));
    Ok(())
}

#[test]
fn mxfp4_moe_executes_unit_projection() -> mircuda::Result<()> {
    const TOKENS: usize = 8;
    const OUTPUTS: usize = 64;
    const PADDED_K: usize = 128;
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let repack = MarlinMxFp4RepackSpec::new(1, OUTPUTS, PADDED_K, OUTPUTS, K)?;
    let weights = copy_device(&context, &stream, &pool, &vec![0x22_u8; OUTPUTS * K / 2])?;
    let mut packed = pool.allocate_zeroed(&stream, OUTPUTS * PADDED_K / 2)?;
    context.marlin_repack_mxfp4(&stream, repack, &weights, &mut packed, false)?;
    let scales = copy_device(&context, &stream, &pool, &[127_u8; OUTPUTS * K / 32])?;
    let mut prepared_scales = pool.allocate_zeroed(&stream, OUTPUTS * PADDED_K / 32)?;
    context.marlin_prepare_mxfp4_scales(&stream, repack, &scales, &mut prepared_scales, false)?;

    let mut input_values = vec![bf16::from_f32(0.0); TOKENS * PADDED_K];
    for row in input_values.as_chunks_mut::<PADDED_K>().0 {
        row[..K].fill(bf16::from_f32(1.0));
    }
    let input = copy_device(&context, &stream, &pool, &input_values)?;
    let selected = copy_device(&context, &stream, &pool, &[0_u32; TOKENS])?;
    let routing = copy_device(&context, &stream, &pool, &[bf16::from_f32(1.0); TOKENS])?;
    let capacity = TOKENS + 7;
    let mut sorted = pool.allocate(&stream, capacity)?;
    let mut expert_ids = pool.allocate(&stream, capacity.div_ceil(8))?;
    let mut padded = pool.allocate(&stream, 1)?;
    let mut offsets = pool.allocate(&stream, 1)?;
    let mut routing_f32 = pool.allocate(&stream, TOKENS)?;
    let mut temporary = pool.allocate(&stream, capacity * OUTPUTS)?;
    let mut locks = pool.allocate_zeroed(&stream, 1024)?;
    let mut output = pool.allocate_zeroed(&stream, TOKENS * OUTPUTS)?;
    let spec =
        MarlinNvFp4MoeSpec::new(1, TOKENS, 1, OUTPUTS, PADDED_K, MarlinNvFp4ThreadConfig::N64K128)?;
    context.marlin_prepare_moe_routes(
        &stream, spec, &selected, &routing, &mut sorted, &mut expert_ids, &mut padded,
        &mut offsets, &mut routing_f32,
    )?;
    context.marlin_mxfp4_moe(
        &stream,
        spec,
        &MarlinMxFp4MoeOperands {
            input: &input,
            weight: &packed,
            scales: &prepared_scales,
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
    assert!(actual.iter().all(|value| (value.to_f32() - 64.0).abs() <= 0.5));
    Ok(())
}

fn canonical_weights() -> Vec<u8> {
    let mut result = vec![0_u8; EXPERTS * N * K / 2];
    for expert in 0..EXPERTS {
        for n in 0..N {
            for k in (0..K).step_by(2) {
                let low = (expert * 7 + n * 3 + k) & 15;
                let high = (expert * 7 + n * 3 + k + 1) & 15;
                result[expert * N * K / 2 + n * K / 2 + k / 2] =
                    u8::try_from(low | high << 4).unwrap_or_default();
            }
        }
    }
    result
}

fn reference_repack(raw: &[u8]) -> Vec<u8> {
    let mut output = vec![0_u32; EXPERTS * N * K / 8];
    for expert in 0..EXPERTS {
        for kt in 0..K / 16 {
            for lane in 0..32 {
                for warp in 0..4 {
                    let n = warp * 16 + lane / 4;
                    let k = kt * 16 + (lane % 4) * 2;
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
                    let packed = order
                        .iter()
                        .enumerate()
                        .fold(0_u32, |word, (shift, index)| word | values[*index] << (shift * 4));
                    let tile = expert * K / 16 + kt;
                    output[tile * 128 + lane * 4 + warp] = packed;
                }
            }
        }
    }
    output.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn concatenated_gate_up(raw: &[u8]) -> Vec<u8> {
    let row_bytes = K / 2;
    let mut output = vec![0; raw.len()];
    for expert in 0..EXPERTS {
        for row in 0..N {
            let source_row = if row < N / 2 {
                row * 2
            } else {
                (row - N / 2) * 2 + 1
            };
            let source = (expert * N + source_row) * row_bytes;
            let destination = (expert * N + row) * row_bytes;
            output[destination..destination + row_bytes]
                .copy_from_slice(&raw[source..source + row_bytes]);
        }
    }
    output
}

fn reference_scales(raw: &[u8]) -> Vec<u8> {
    let groups = K / 32;
    let per_expert = N * groups;
    let mut output = vec![0; raw.len()];
    for expert in 0..EXPERTS {
        for group in 0..groups {
            for row in 0..N {
                let source_row = if row < N / 2 {
                    row * 2
                } else {
                    (row - N / 2) * 2 + 1
                };
                let logical = group * N + row;
                let source = (expert * N + source_row) * groups + group;
                output[expert * per_expert + scale_destination(logical)] = raw[source];
            }
        }
    }
    output
}

const fn scale_destination(linear: usize) -> usize {
    let chunk = linear / 64;
    let local = linear % 64;
    let permuted = chunk * 64 + 8 * (local % 8) + local / 8;
    let reorder = [0, 2, 1, 3];
    permuted / 4 * 4 + reorder[permuted % 4]
}

fn value(raw: &[u8], expert: usize, n: usize, k: usize) -> u32 {
    let packed = raw[expert * N * K / 2 + n * K / 2 + k / 2];
    u32::from((packed >> ((k & 1) * 4)) & 15)
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
