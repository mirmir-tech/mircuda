#![cfg(all(target_os = "linux", feature = "cutlass"))]

use mircuda::{
    BlockwiseFp8VectorPlan, BlockwiseFp8VectorSpec, Context, DeviceBuffer, DeviceElement, Driver,
    MemoryPool, Stream, bf16,
};

const N: usize = 128;
const K: usize = 128;
const E4M3_ONE: u8 = 0x38;

#[test]
fn blockwise_fp8_vector_produces_bf16() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let input = copy_device(&context, &stream, &pool, &[E4M3_ONE; K])?;
    let input_scales = copy_device(&context, &stream, &pool, &[1.0_f32])?;
    let weight = copy_device(&context, &stream, &pool, &[E4M3_ONE; N * K])?;
    let weight_scales = copy_device(&context, &stream, &pool, &[1.0_f32])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, N)?;
    let spec = BlockwiseFp8VectorSpec::new(N, K)?;
    let mut plan = BlockwiseFp8VectorPlan::new(&context, &stream, spec)?;
    plan.execute(&stream, &input, &input_scales, &weight, &weight_scales, &mut output)?;
    assert_eq!(plan.spec(), spec);
    drop(plan);
    let actual = read_device(&context, &stream, &output)?;
    let expected = bf16::from_f32(128.0);
    assert!(actual.iter().all(|value| *value == expected));
    Ok(())
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
