use std::num::NonZeroU16;

use mircuda::{
    CaptureMode, DenseVectorKernel, DenseVectorPlan, DenseVectorSpec, DeviceBuffer, Error, Stream,
    bf16, f16,
};

use super::super::{copy_device, environment, read_device};

const fn split(splits: u16) -> DenseVectorKernel {
    match NonZeroU16::new(splits) {
        Some(splits) => DenseVectorKernel::SplitK { splits },
        None => DenseVectorKernel::BlockPerRow,
    }
}

/// Untuned plans keep one block per row for router-sized outputs only.
#[test]
fn default_kernel_reserves_block_per_row_for_small_outputs() -> mircuda::Result<()> {
    let kernel = |n, k| DenseVectorSpec::new(n, k).map(DenseVectorSpec::kernel);
    assert_eq!(kernel(512, 2_048)?, DenseVectorKernel::BlockPerRow);
    assert_eq!(kernel(513, 2_048)?, DenseVectorKernel::RowGroups);
    assert_eq!(kernel(262_144, 2_816)?, DenseVectorKernel::RowGroups);
    assert_eq!(kernel(32, 2_050)?, DenseVectorKernel::RowGroups);
    Ok(())
}

/// Split-K needs 8-element vectors and at least one vector per slice.
#[test]
fn split_k_rejects_inadmissible_widths() -> mircuda::Result<()> {
    let shape = DenseVectorSpec::new(32, 64)?;
    assert_eq!(shape.with_kernel(split(8))?.kernel(), split(8));
    for (shape, kernel) in [(shape, split(9)), (DenseVectorSpec::new(32, 2_050)?, split(2))] {
        let error = shape.with_kernel(kernel).err();
        assert!(
            matches!(error, Some(Error::VectorKernelShape { kernel: rejected, n: 32, .. })
                if rejected == kernel),
            "{kernel:?} was not rejected: {error:?}"
        );
    }
    Ok(())
}

/// Candidates fill two waves of one block per SM, then double; outputs
/// that fill two waves on their own measure only two slices.
#[test]
fn split_k_candidates_follow_the_multiprocessor_count() -> mircuda::Result<()> {
    let candidates = |n, k| DenseVectorSpec::new(n, k).map(|spec| spec.split_k_candidates(48));
    assert_eq!(candidates(32, 2_048)?, [split(3), split(6)]);
    assert_eq!(candidates(64, 7_168)?, [split(2), split(4)]);
    assert_eq!(candidates(32, 32)?, [split(3)]);
    assert_eq!(DenseVectorSpec::new(32, 2_048)?.split_k_candidates(96), [split(6), split(12)]);
    for (n, k) in [(96, 2_048), (128, 7_168), (256, 4_096), (262_144, 2_816)] {
        assert_eq!(candidates(n, k)?, [split(2)], "n={n} k={k}");
    }
    assert!(candidates(4_096, 8)?.is_empty());
    assert!(candidates(32, 2_050)?.is_empty());
    Ok(())
}

/// The last-arriving slice applies alpha and beta exactly once.
#[test]
fn split_k_supports_alpha_and_beta() -> mircuda::Result<()> {
    const N: usize = 96;
    const K: usize = 1_024;
    let (context, stream, pool) = environment()?;
    let input = copy_device(&context, &stream, &pool, &[f16::ONE; K])?;
    let weight = copy_device(&context, &stream, &pool, &vec![f16::ONE; N * K])?;
    let mut output = copy_device(&context, &stream, &pool, &[f16::ONE; N])?;
    let spec = DenseVectorSpec::new(N, K)?.with_kernel(split(5))?;
    let mut plan = DenseVectorPlan::new(&context, &stream, spec)?;
    assert_eq!(plan.workspace_bytes(), N * (size_of::<u32>() + 5 * size_of::<f32>()));
    plan.execute(&stream, &input, &weight, &mut output, 0.5, 1.0)?;
    let actual = read_device(&context, &stream, &output)?;
    assert!(actual.iter().all(|value| *value == f16::from_f32(513.0)), "{actual:?}");
    Ok(())
}

struct Replay<'a> {
    stream: &'a Stream,
    plan: DenseVectorPlan<bf16>,
    input: &'a DeviceBuffer<bf16>,
    weight: &'a DeviceBuffer<bf16>,
    output: &'a mut DeviceBuffer<bf16>,
}

/// Graph replays reuse the plan workspace and reproduce direct execution
/// bit for bit, so the fixed-order reduction is deterministic.
#[test]
fn split_k_replays_deterministically_under_capture() -> mircuda::Result<()> {
    const N: usize = 128;
    const K: usize = 4_096;
    let (context, stream, pool) = environment()?;
    let input = (0..K)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 29)?) / 16.0 - 0.9)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    let weight = (0..N * K)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 13)?) / 8.0 - 0.7)))
        .collect::<mircuda::Result<Vec<_>>>()?;
    let input = copy_device(&context, &stream, &pool, &input)?;
    let weight = copy_device(&context, &stream, &pool, &weight)?;
    let spec = DenseVectorSpec::new(N, K)?.with_kernel(split(10))?;
    let mut direct = copy_device(&context, &stream, &pool, &[bf16::NAN; N])?;
    DenseVectorPlan::new(&context, &stream, spec)?
        .execute(&stream, &input, &weight, &mut direct, 1.0, 0.0)?;
    let expected = read_device(&context, &stream, &direct)?;
    let mut replayed = copy_device(&context, &stream, &pool, &[bf16::NAN; N])?;
    let replay = Replay {
        stream: &stream,
        plan: DenseVectorPlan::new(&context, &stream, spec)?,
        input: &input,
        weight: &weight,
        output: &mut replayed,
    };
    let mut graph = stream.capture_with(CaptureMode::ThreadLocal, replay, |replay| {
        replay
            .plan
            .execute(replay.stream, replay.input, replay.weight, replay.output, 1.0, 0.0)
    })?;
    for _ in 0..3 {
        graph.launch(&stream)?;
        stream.synchronize()?;
        let actual = read_device(&context, &stream, &*graph.resources().output)?;
        let bits = |values: &[bf16]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&actual), bits(&expected));
    }
    Ok(())
}
