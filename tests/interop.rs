#[test]
#[cfg(target_os = "linux")]
fn foreign_handles_borrow_the_native_context_stream_and_allocation() -> mircuda::Result<()> {
    let driver = mircuda::Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let other = context.create_stream()?;
    let foreign = context.foreign_context()?;
    assert_eq!(foreign.ordinal, context.device_info()?.ordinal);
    assert!(foreign.device >= 0);
    assert_ne!(stream.foreign_stream()?, other.foreign_stream()?);
    let pool = context.default_memory_pool()?;
    let values = pool.allocate_zeroed::<f32>(&stream, 1_024)?;
    let native = values.foreign_pointer(&stream)?;
    assert_eq!(native.bytes, 4_096);
    assert_ne!(native.address, 0);
    assert_eq!(values.foreign_pointer(&other)?, native);
    // Cross-stream use makes the drop synchronize before freeing.
    drop(values);
    stream.synchronize()?;
    other.synchronize()?;
    Ok(())
}
