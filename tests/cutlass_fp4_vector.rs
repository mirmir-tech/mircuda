#![cfg(all(target_os = "linux", feature = "cutlass"))]

use mircuda::{
    BlockScaledFp4VectorPlan, BlockScaledFp4VectorSpec, Context, DeviceBuffer, DeviceElement,
    Driver, MemoryPool, Stream, bf16,
};

const N: usize = 704;
const K: usize = 2_816;

#[test]
fn block_scaled_fp4_vector_produces_bf16() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let input = copy_device(&context, &stream, &pool, &vec![0x22_u8; K / 2])?;
    let input_scales = copy_device(&context, &stream, &pool, &vec![0x38_u8; scales(1, K)])?;
    let weight = copy_device(&context, &stream, &pool, &vec![0x22_u8; N * K / 2])?;
    let weight_scales = copy_device(&context, &stream, &pool, &vec![0x38_u8; scales(N, K)])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, N)?;
    let mut plan =
        BlockScaledFp4VectorPlan::new(&context, &stream, BlockScaledFp4VectorSpec::new(N, K)?)?;
    plan.execute(&stream, &input, &input_scales, &weight, &weight_scales, &mut output, 1.0)?;
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
