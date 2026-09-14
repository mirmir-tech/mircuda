use std::sync::{Arc, atomic::Ordering};

use cudarc::driver::{result, sys};

use super::DeviceBuffer;

impl DeviceBuffer {
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn slice(self: &Arc<Self>, offset: usize, bytes: usize) -> crate::Result<Self> {
        if bytes == 0 || offset.checked_add(bytes).is_none_or(|end| end > self.bytes) {
            return Err(crate::Error::InvalidTransferRange);
        }
        let pointer = self
            .pointer
            .checked_add(u64::try_from(offset)?)
            .ok_or(crate::Error::InvalidTransferRange)?;
        Ok(Self {
            pointer,
            bytes,
            stream: self.stream.clone(),
            cross_stream: self.cross_stream.clone(),
            owner: Some(self.owner.as_ref().unwrap_or(self).clone()),
        })
    }

    #[must_use]
    pub fn is_aligned(&self, alignment: usize) -> bool {
        u64::try_from(alignment)
            .is_ok_and(|alignment| alignment > 0 && self.pointer.is_multiple_of(alignment))
    }

    #[must_use]
    pub fn argument(&self) -> super::super::compiler::KernelArgument {
        super::super::compiler::KernelArgument::Pointer {
            value: self.pointer,
            stream: self.stream.cu_stream(),
            context: Arc::as_ptr(self.stream.context()),
            cross_stream: Arc::as_ptr(&self.cross_stream),
        }
    }

    #[must_use]
    pub(in crate::platform) const fn pointer(&self) -> sys::CUdeviceptr {
        self.pointer
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.owner.is_some() {
            return;
        }
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        if self.cross_stream.load(Ordering::Acquire) {
            self.stream.context().record_err(self.stream.context().synchronize());
        }
        // SAFETY: this is the sole owner and free is ordered on the allocation stream.
        self.stream
            .context()
            .record_err(unsafe { result::free_async(self.pointer, self.stream.cu_stream()) });
    }
}
