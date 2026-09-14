use super::{DeviceBuffer, KernelArgument, PinnedBuffer, Stream, unsupported};
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

impl DeviceBuffer {
    pub const fn slice(self: &std::sync::Arc<Self>, _offset: usize, _bytes: usize) -> Result<Self> {
        Err(unsupported())
    }

    #[must_use]
    pub const fn is_aligned(&self, _alignment: usize) -> bool {
        false
    }

    #[must_use]
    pub const fn bytes(&self) -> usize {
        0
    }

    #[must_use]
    pub const fn argument(&self) -> KernelArgument {
        KernelArgument::Pointer { value: 0, stream: 0 }
    }
}
