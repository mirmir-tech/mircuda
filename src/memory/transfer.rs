use std::ops::Range;

use super::{DeviceBuffer, DeviceElement, PinnedBuffer, validate_lengths};
use crate::{Error, Result, Stream};

impl Stream {
    /// Enqueues a pinned-host to device transfer.
    pub fn copy_to_device<T: DeviceElement>(
        &self,
        source: &mut PinnedBuffer<T>,
        target: &mut DeviceBuffer<T>,
    ) -> Result<()> {
        validate_lengths(source.len, target.len)?;
        Ok(self.native.copy_to_device(&source.native, &target.native)?)
    }

    /// Enqueues a device to pinned-host transfer.
    pub fn copy_to_host<T: DeviceElement>(
        &self,
        source: &DeviceBuffer<T>,
        target: &mut PinnedBuffer<T>,
    ) -> Result<()> {
        validate_lengths(source.len, target.len)?;
        Ok(self.native.copy_to_host(&source.native, &target.native)?)
    }

    /// Enqueues a typed device-to-device range copy on this stream.
    pub fn copy_device_range<T: DeviceElement>(
        &self,
        source: &DeviceBuffer<T>,
        range: Range<usize>,
        target: &mut DeviceBuffer<T>,
        target_offset: usize,
    ) -> Result<()> {
        let elements = range
            .end
            .checked_sub(range.start)
            .filter(|elements| *elements > 0)
            .ok_or(Error::InvalidTransferRange)?;
        let target_end = target_offset.checked_add(elements).ok_or(Error::InvalidTransferRange)?;
        if range.end > source.len || target_end > target.len {
            return Err(Error::InvalidTransferRange);
        }
        let bytes = size_of::<T>();
        Ok(self.native.copy_device_range(
            &source.native,
            range.start * bytes,
            &target.native,
            target_offset * bytes,
            elements * bytes,
        )?)
    }
}
