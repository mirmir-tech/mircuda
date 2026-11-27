//! Operand views whose addresses break the 16-byte vector loads.

use std::num::NonZeroU16;

use mircuda::{DenseVectorKernel, DenseVectorPlan, DenseVectorSpec, DeviceBuffer, Error, bf16};

use super::super::{copy_device, environment, read_device};

const OUTPUTS: usize = 33;
/// A multiple of 8, so aligned operands take the 16-byte loads.
const FEATURES: usize = 2_048;

fn pattern(len: usize, period: usize, scale: f32) -> mircuda::Result<Vec<bf16>> {
    (0..len)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % period)?) / scale - 0.25)))
        .collect()
}

/// Each operand alone viewed 1, 2 and 8 elements into its allocation (2, 4
/// and 16 bytes): row groups load at the width both addresses allow and
/// match the host reference once the stream is synchronized; block-per-row
/// and split-K refuse a view off 16 bytes with a typed error.
#[test]
fn offset_views_load_at_their_alignment_or_are_refused() -> mircuda::Result<()> {
    let (context, stream, pool) = environment()?;
    let input = pattern(FEATURES, 31, 32.0)?;
    let weight = pattern(OUTPUTS * FEATURES, 17, 64.0)?;
    let expected = weight
        .chunks(FEATURES)
        .map(|row| row.iter().zip(&input).map(|(w, x)| w.to_f32() * x.to_f32()).sum::<f32>())
        .collect::<Vec<_>>();
    let view = |values: &[bf16], offset: usize| -> mircuda::Result<DeviceBuffer<bf16>> {
        let mut padded = vec![bf16::NAN; offset];
        padded.extend_from_slice(values);
        copy_device(&context, &stream, &pool, &padded)?.slice(offset..offset + values.len())
    };
    let splits = NonZeroU16::new(4).ok_or(Error::InvalidMatmulShape)?;
    let kernels = [
        DenseVectorKernel::RowGroups,
        DenseVectorKernel::BlockPerRow,
        DenseVectorKernel::SplitK { splits },
    ];
    for offset in [1, 2, 8] {
        for operand in ["input", "weight"] {
            let shifted = |name| {
                if name == operand {
                    offset
                } else {
                    0
                }
            };
            let (input_view, weight_view) =
                (view(&input, shifted("input"))?, view(&weight, shifted("weight"))?);
            for kernel in kernels {
                let spec = DenseVectorSpec::new(OUTPUTS, FEATURES)?.with_kernel(kernel)?;
                let mut output = copy_device(&context, &stream, &pool, &[bf16::NAN; OUTPUTS])?;
                let result = DenseVectorPlan::new(&context, &stream, spec)?
                    .execute(&stream, &input_view, &weight_view, &mut output, 1.0, 0.0);
                if kernel != DenseVectorKernel::RowGroups && offset % 8 != 0 {
                    assert!(
                        matches!(result, Err(Error::MisalignedOperand {
                            operand: rejected, alignment: 16, kernel: refused
                        }) if rejected == operand && refused == kernel),
                        "{kernel:?} accepted {operand} {offset} elements in: {result:?}"
                    );
                    continue;
                }
                result?;
                stream.synchronize()?;
                let actual = read_device(&context, &stream, &output)?;
                for (row, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    let error = (actual.to_f32() - expected).abs();
                    assert!(
                        error <= 0.125 + expected.abs() / 128.0,
                        "{kernel:?}, {operand} {offset} elements in, row {row}: \
                         {actual} != {expected}"
                    );
                }
            }
        }
    }
    Ok(())
}
