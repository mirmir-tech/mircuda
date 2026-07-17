#![cfg(all(target_os = "linux", feature = "cutlass"))]

use mircuda::{
    BlockScaledFp4Plan, BlockScaledFp4Spec, Context, DeviceBuffer, DeviceElement, Driver,
    MemoryPool, Stream, bf16,
};

const M: usize = 1;
const N: usize = 704;
const K: usize = 2_816;

#[test]
fn block_scaled_fp4_produces_bf16() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let a = copy_device(&context, &stream, &pool, &vec![0x22_u8; M * K / 2])?;
    let a_scales = copy_device(&context, &stream, &pool, &vec![0x38_u8; scales(M, K)])?;
    let b = copy_device(&context, &stream, &pool, &vec![0x22_u8; N * K / 2])?;
    let b_scales = copy_device(&context, &stream, &pool, &vec![0x38_u8; scales(N, K)])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, M * N)?;
    let mut plan = BlockScaledFp4Plan::new(&context, &stream, BlockScaledFp4Spec::new(M, N, K)?)?;
    plan.execute(&stream, &a, &a_scales, &b, &b_scales, &mut output, 1.0)?;
    drop(plan);
    let actual = read_device(&context, &stream, &output)?;
    let expected = bf16::from_f32(f32::from(u16::try_from(K)?));
    assert!(actual.iter().all(|value| *value == expected));
    Ok(())
}

const fn scales(rows: usize, columns: usize) -> usize {
    rows.div_ceil(128) * columns / 64 * 512
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
    stream.synchronize()?;
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
