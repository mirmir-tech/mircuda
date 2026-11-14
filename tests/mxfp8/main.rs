//! MXFP8 geometry, matmuls, gathered banks, embeddings and quantization.

#[cfg(target_os = "linux")]
use mircuda::{Context, DeviceBuffer, DeviceElement, MemoryPool, Stream};
use mircuda::{MxFp8EmbeddingSpec, MxFp8GatheredSpec, MxFp8Spec};

#[cfg(target_os = "linux")]
mod gathered;
#[cfg(target_os = "linux")]
mod matmul;
#[cfg(target_os = "linux")]
mod quantize;

#[test]
fn validates_mxfp8_geometry_without_a_device() -> mircuda::Result<()> {
    let spec = MxFp8Spec::new(2, 32, 3)?;
    assert_eq!(spec.tokens(), 2);
    assert_eq!(spec.input_features(), 32);
    assert_eq!(spec.output_features(), 3);
    assert_eq!(spec.weight_words()?, 24);
    assert_eq!(spec.scale_elements()?, 3);
    assert!(MxFp8Spec::new(1, 31, 1).is_err());
    assert_eq!(MxFp8EmbeddingSpec::new(3, 32, 1.0)?.weight_words()?, 24);
    assert!(MxFp8GatheredSpec::new_routed(1, 2, 2, 32, 3).is_ok());
    Ok(())
}

#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
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
