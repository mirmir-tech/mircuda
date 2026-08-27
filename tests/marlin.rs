#![cfg(all(feature = "marlin", target_os = "linux"))]

use mircuda::{
    Context, DeviceBuffer, DeviceElement, Driver, MarlinNvFp4RepackSpec, MemoryPool, Stream,
};

const EXPERTS: usize = 2;
const N: usize = 64;
const K: usize = 32;

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
