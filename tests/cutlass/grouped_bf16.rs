#![cfg(all(target_os = "linux", feature = "cutlass"))]

use mircuda::{
    Context, DeviceBuffer, DeviceElement, Driver, MemoryPool, Stream, VariableGroupedBf16Plan,
    VariableGroupedBf16Spec, bf16,
};

const N: usize = 128;
const K: usize = 128;

#[test]
fn variable_grouped_bf16_uses_compact_device_routing() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let capacity = 6;
    let input = copy_device(&context, &stream, &pool, &vec![bf16::ONE; capacity * K])?;
    let matrix_elements = K * N;
    let mut weight_values = vec![bf16::ONE; 4 * matrix_elements];
    weight_values[matrix_elements..2 * matrix_elements].fill(bf16::from_f32(2.0));
    weight_values[2 * matrix_elements..3 * matrix_elements].fill(bf16::from_f32(3.0));
    weight_values[3 * matrix_elements..].fill(bf16::from_f32(4.0));
    let weights = copy_device(&context, &stream, &pool, &weight_values)?;
    let rows = copy_device(&context, &stream, &pool, &[3_u32, 0, 2, 1])?;
    let offsets = copy_device(&context, &stream, &pool, &[0_u32, 3, 3, 5])?;
    let mut output = copy_device(&context, &stream, &pool, &vec![bf16::NAN; capacity * N])?;
    let spec = VariableGroupedBf16Spec::new(4, 4, N, K, capacity)?;
    let mut plan = VariableGroupedBf16Plan::new(&context, &stream, spec)?;
    plan.execute(&stream, &input, &weights, &rows, &offsets, &mut output, 0.0)?;
    drop(plan);
    let actual = read_device(&context, &stream, &output)?;
    assert_rows(&actual, &[128.0, 128.0, 128.0, 384.0, 384.0, 512.0]);
    Ok(())
}

fn assert_rows(actual: &[bf16], expected_rows: &[f32]) {
    let (rows, remainder) = actual.as_chunks::<N>();
    assert!(remainder.is_empty());
    for (row, values) in rows.iter().enumerate() {
        let expected = bf16::from_f32(expected_rows[row]);
        assert!(values.iter().all(|value| *value == expected), "row {row}");
    }
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
