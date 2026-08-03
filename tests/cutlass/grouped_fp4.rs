#![cfg(all(target_os = "linux", feature = "cutlass"))]

use mircuda::{
    Context, DeviceBuffer, DeviceElement, Driver, IndexedGroupedFp4Plan, IndexedGroupedFp4Spec,
    MemoryPool, PairedVariableGroupedFp4Launch, PairedVariableGroupedFp4Plan, Stream,
    VariableGroupedFp4Metadata, VariableGroupedFp4Operands, VariableGroupedFp4Plan,
    VariableGroupedFp4Spec, bf16,
};

const GROUPS: usize = 64;
const MATRICES: usize = 33;
const N: usize = 128;
const K: usize = 128;

#[test]
fn indexed_grouped_fp4_selects_device_resident_matrices() -> mircuda::Result<()> {
    run_case(GROUPS, 128, false)
}

#[test]
fn indexed_grouped_fp4_supports_decode_rows() -> mircuda::Result<()> {
    run_case(GROUPS, 1, false)
}

#[test]
fn indexed_grouped_fp4_supports_multiple_metadata_blocks() -> mircuda::Result<()> {
    run_case(1_025, 1, false)
}

#[test]
fn indexed_grouped_fp4_broadcasts_one_input() -> mircuda::Result<()> {
    run_case(2, 2, true)
}

#[test]
fn variable_grouped_fp4_uses_device_rows_and_compact_offsets() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let rows_values = [3_u32, 0, 2, 1];
    let offset_values = [0_u32, 3, 3, 5];
    let index_values = [1_u32, 0, 2, 1];
    let capacity_rows = 6;
    let a = copy_device(&context, &stream, &pool, &vec![0x22_u8; capacity_rows * K / 2])?;
    let scale_bytes = K / 64 * 512;
    let a_scales = copy_device(&context, &stream, &pool, &vec![0x38_u8; 4 * scale_bytes])?;
    let matrix_bytes = N * K / 2;
    let mut weights = vec![0x22_u8; 3 * matrix_bytes];
    weights[..matrix_bytes].fill(0);
    let b = copy_device(&context, &stream, &pool, &weights)?;
    let b_scales = copy_device(&context, &stream, &pool, &vec![0x38_u8; 3 * scale_bytes])?;
    let alphas = copy_device(&context, &stream, &pool, &[1.0_f32, 0.5, 0.25])?;
    let indices = copy_device(&context, &stream, &pool, &index_values)?;
    let rows = copy_device(&context, &stream, &pool, &rows_values)?;
    let offsets = copy_device(&context, &stream, &pool, &offset_values)?;
    let scale_offsets = copy_device(&context, &stream, &pool, &[0_u32, 128, 128, 256])?;
    let mut output = copy_device(&context, &stream, &pool, &vec![bf16::ONE; capacity_rows * N])?;
    let spec = VariableGroupedFp4Spec::new(4, 3, 4, N, K, capacity_rows)?;
    let mut plan = VariableGroupedFp4Plan::new(&context, &stream, spec)?;
    plan.execute(
        &stream, &a, &a_scales, &b, &b_scales, &alphas, &indices, &rows, &offsets, &scale_offsets,
        &mut output,
    )?;
    let actual = read_device(&context, &stream, &output)?;
    assert_rows(&actual, &[64.0, 64.0, 64.0, 32.0, 32.0, 64.0]);

    let rows = copy_device(&context, &stream, &pool, &[0_u32, 1, 1, 4])?;
    let offsets = copy_device(&context, &stream, &pool, &[0_u32, 0, 1, 2])?;
    let scale_offsets = copy_device(&context, &stream, &pool, &[0_u32, 0, 128, 256])?;
    plan.execute(
        &stream, &a, &a_scales, &b, &b_scales, &alphas, &indices, &rows, &offsets, &scale_offsets,
        &mut output,
    )?;
    drop(plan);
    let actual = read_device(&context, &stream, &output)?;
    assert_rows(&actual, &[0.0, 32.0, 64.0, 64.0, 64.0, 64.0]);
    Ok(())
}

#[test]
#[allow(clippy::similar_names)]
fn paired_variable_grouped_fp4_shares_only_device_metadata() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let capacity = 6;
    let activation = vec![0x22_u8; capacity * K / 2];
    let scales = vec![0x38_u8; 4 * K / 64 * 512];
    let left_a = copy_device(&context, &stream, &pool, &activation)?;
    let right_a = copy_device(&context, &stream, &pool, &activation)?;
    let left_a_scales = copy_device(&context, &stream, &pool, &scales)?;
    let right_a_scales = copy_device(&context, &stream, &pool, &scales)?;
    let mut weights = vec![0x22_u8; 3 * N * K / 2];
    weights[..N * K / 2].fill(0);
    let left_b = copy_device(&context, &stream, &pool, &weights)?;
    let right_b = copy_device(&context, &stream, &pool, &weights)?;
    let bank_scales = vec![0x38_u8; 3 * N.div_ceil(128) * K / 64 * 512];
    let left_b_scales = copy_device(&context, &stream, &pool, &bank_scales)?;
    let right_b_scales = copy_device(&context, &stream, &pool, &bank_scales)?;
    let left_alphas = copy_device(&context, &stream, &pool, &[1.0_f32, 0.5, 0.25])?;
    let right_alphas = copy_device(&context, &stream, &pool, &[2.0_f32, 1.0, 0.5])?;
    let indices = copy_device(&context, &stream, &pool, &[1_u32, 0, 2, 1])?;
    let rows = copy_device(&context, &stream, &pool, &[3_u32, 0, 2, 1])?;
    let offsets = copy_device(&context, &stream, &pool, &[0_u32, 3, 3, 5])?;
    let scale_offsets = copy_device(&context, &stream, &pool, &[0_u32, 128, 128, 256])?;
    let mut left = copy_device(&context, &stream, &pool, &vec![bf16::ONE; capacity * N])?;
    let mut right = copy_device(&context, &stream, &pool, &vec![bf16::ONE; capacity * N])?;
    let spec = VariableGroupedFp4Spec::new(4, 3, 4, N, K, capacity)?;
    let mut plan = PairedVariableGroupedFp4Plan::new(&context, &stream, spec)?;
    plan.execute(
        &stream,
        &mut PairedVariableGroupedFp4Launch {
            left: VariableGroupedFp4Operands {
                a: &left_a,
                a_scales: &left_a_scales,
                b: &left_b,
                b_scales: &left_b_scales,
                alphas: &left_alphas,
                output: &mut left,
            },
            right: VariableGroupedFp4Operands {
                a: &right_a,
                a_scales: &right_a_scales,
                b: &right_b,
                b_scales: &right_b_scales,
                alphas: &right_alphas,
                output: &mut right,
            },
            metadata: VariableGroupedFp4Metadata {
                indices: &indices,
                rows: &rows,
                offsets: &offsets,
                scale_offsets: &scale_offsets,
            },
        },
    )?;
    assert_rows(&read_device(&context, &stream, &left)?, &[64.0, 64.0, 64.0, 32.0, 32.0, 64.0]);
    assert_rows(
        &read_device(&context, &stream, &right)?,
        &[128.0, 128.0, 128.0, 64.0, 64.0, 128.0],
    );
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

fn run_case(groups: usize, m: usize, broadcast: bool) -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let inputs = if broadcast {
        1
    } else {
        groups
    };
    let a = copy_device(&context, &stream, &pool, &vec![0x22_u8; inputs * m * K / 2])?;
    let a_scale_stride = m.div_ceil(128) * K / 64 * 512;
    let a_scales = copy_device(&context, &stream, &pool, &vec![0x38_u8; inputs * a_scale_stride])?;
    let matrix_bytes = N * K / 2;
    let mut weights = vec![0x22_u8; MATRICES * matrix_bytes];
    weights[..matrix_bytes].fill(0);
    let b = copy_device(&context, &stream, &pool, &weights)?;
    let b_scale_stride = N.div_ceil(128) * K / 64 * 512;
    let b_scales =
        copy_device(&context, &stream, &pool, &vec![0x38_u8; MATRICES * b_scale_stride])?;
    let alpha_values = (0..MATRICES)
        .map(|matrix| {
            if matrix.is_multiple_of(2) {
                0.5_f32
            } else {
                1.0_f32
            }
        })
        .collect::<Vec<_>>();
    let alphas = copy_device(&context, &stream, &pool, &alpha_values)?;
    let index_values = (0..groups)
        .map(|group| u32::try_from(group % MATRICES))
        .collect::<Result<Vec<_>, _>>()?;
    let indices = copy_device(&context, &stream, &pool, &index_values)?;
    let mut output = copy_device(&context, &stream, &pool, &vec![bf16::ONE; groups * m * N])?;
    let mut spec = IndexedGroupedFp4Spec::new(groups, MATRICES, m, N, K)?;
    if broadcast {
        spec = spec.with_broadcast_input();
    }
    let mut plan = IndexedGroupedFp4Plan::new(&context, &stream, spec)?;
    plan.execute(&stream, &a, &a_scales, &b, &b_scales, &alphas, &indices, &mut output)?;
    drop(plan);
    let actual = read_device(&context, &stream, &output)?;
    for (group, values) in actual.chunks_exact(m * N).enumerate() {
        let expert = group % MATRICES;
        let expected = if expert == 0 {
            bf16::ZERO
        } else {
            bf16::from_f32(f32::from(u16::try_from(K)?) * alpha_values[expert])
        };
        let mismatch = values.iter().enumerate().find(|(_, value)| **value != expected);
        assert!(
            mismatch.is_none(),
            "group={group} expert={expert} mismatch={:?} expected={}",
            mismatch.map(|(column, actual)| (column, actual.to_f32())),
            expected.to_f32(),
        );
    }
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
