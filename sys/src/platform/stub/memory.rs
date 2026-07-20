use super::{DeviceBuffer, PinnedBuffer, Stream, unsupported};
use crate::Result;

impl Stream {
    pub const fn copy_to_device(
        &self,
        _source: &PinnedBuffer,
        _target: &DeviceBuffer,
    ) -> Result<()> {
        Err(unsupported())
    }

    pub const fn copy_to_host(&self, _source: &DeviceBuffer, _target: &PinnedBuffer) -> Result<()> {
        Err(unsupported())
    }

    pub const fn copy_device_range(
        &self,
        _source: &DeviceBuffer,
        _source_offset: usize,
        _target: &DeviceBuffer,
        _target_offset: usize,
        _bytes: usize,
    ) -> Result<()> {
        Err(unsupported())
    }
}
