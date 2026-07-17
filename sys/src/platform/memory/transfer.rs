use cudarc::driver::result;

use super::{DeviceBuffer, PinnedBuffer, ensure_stream, ensure_transfer_size};
use crate::{Error, Result, platform::driver::Stream};

impl Stream {
    pub fn copy_to_device(&self, source: &PinnedBuffer, target: &DeviceBuffer) -> Result<()> {
        ensure_stream(target, self)?;
        ensure_transfer_size(source.bytes, target.bytes)?;
        // SAFETY: both ranges have equal size and stay alive through the recorded event.
        unsafe {
            result::memcpy_htod_async(
                target.pointer,
                std::slice::from_raw_parts(source.pointer.as_ptr(), source.bytes),
                self.inner.cu_stream(),
            )?;
        }
        source.record(self)
    }

    pub fn copy_to_host(&self, source: &DeviceBuffer, target: &PinnedBuffer) -> Result<()> {
        ensure_stream(source, self)?;
        ensure_transfer_size(source.bytes, target.bytes)?;
        // SAFETY: both ranges have equal size and stay alive through the recorded event.
        unsafe {
            result::memcpy_dtoh_async(
                std::slice::from_raw_parts_mut(target.pointer.as_ptr(), target.bytes),
                source.pointer,
                self.inner.cu_stream(),
            )?;
        }
        target.record(self)
    }

    pub fn copy_device_range(
        &self,
        source: &DeviceBuffer,
        source_offset: usize,
        target: &DeviceBuffer,
        target_offset: usize,
        bytes: usize,
    ) -> Result<()> {
        ensure_stream(source, self)?;
        ensure_stream(target, self)?;
        validate_range(source_offset, bytes, source.bytes)?;
        validate_range(target_offset, bytes, target.bytes)?;
        let source_pointer = source
            .pointer
            .checked_add(u64::try_from(source_offset)?)
            .ok_or(Error::InvalidTransferRange)?;
        let target_pointer = target
            .pointer
            .checked_add(u64::try_from(target_offset)?)
            .ok_or(Error::InvalidTransferRange)?;
        // SAFETY: validated ranges belong to live allocations on this stream.
        Ok(unsafe {
            result::memcpy_dtod_async(target_pointer, source_pointer, bytes, self.inner.cu_stream())
        }?)
    }
}

fn validate_range(offset: usize, bytes: usize, allocation: usize) -> Result<()> {
    if bytes > 0 && offset.checked_add(bytes).is_some_and(|end| end <= allocation) {
        Ok(())
    } else {
        Err(Error::InvalidTransferRange)
    }
}
