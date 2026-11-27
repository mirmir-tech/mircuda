use std::sync::Arc;

use cudarc::driver::{CudaStream, result, sys};

use super::{
    super::{capture_gate, driver::bind_current},
    DeviceBuffer,
    foreign::order_before,
};

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
            foreign: self.foreign.clone(),
            capture: self.capture,
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
            foreign: Arc::as_ptr(&self.foreign),
        }
    }

    #[cfg(any(feature = "cublaslt", feature = "cutlass", feature = "marlin"))]
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
        let foreign = self.foreign.take();
        // The gate stays held from finding no capture through the free, so
        // no capture can begin and take the free, and the work queued on the
        // allocation stream precedes it.
        if foreign.is_empty()
            && capture_gate::release_when_idle(|| free(self.pointer, &self.stream))
        {
            return;
        }
        let (pointer, stream) = (self.pointer, self.stream.clone());
        // A free on a stream another capture records into would become a
        // node of that graph for memory it did not allocate, and the foreign
        // ordering records an event on each foreign stream and waits on it
        // from the allocation stream, which joins or invalidates a capture on
        // either. The release then waits for those captures to end.
        let streams = std::iter::once(stream.clone()).chain(foreign.iter().cloned()).collect();
        capture_gate::release_ordered(streams, self.capture, move || {
            release(pointer, stream, &foreign)
        });
    }
}

/// Frees `pointer` on `stream` after the work queued on `foreign`, or returns
/// the device-synchronizing free to run when that ordering fails.
fn release(
    pointer: sys::CUdeviceptr,
    stream: Arc<CudaStream>,
    foreign: &[Arc<CudaStream>],
) -> Result<(), capture_gate::Teardown> {
    let ordered = bind_current(stream.context())
        .and_then(|current| order_before(foreign, &stream).map(|()| current));
    if let Err(error) = ordered {
        // Without the stream ordering the free could overtake foreign work,
        // so it waits for the whole context instead.
        stream.context().record_err::<()>(Err(error));
        return Err(synchronized_free(pointer, stream));
    }
    free(pointer, &stream);
    Ok(())
}

/// Frees `pointer` once the context has finished all work, or leaks it when
/// that cannot be established: freeing then could overtake foreign work.
fn synchronized_free(pointer: sys::CUdeviceptr, stream: Arc<CudaStream>) -> capture_gate::Teardown {
    Box::new(move || {
        let context = stream.context();
        match bind_current(context).and_then(|current| result::ctx::synchronize().map(|()| current))
        {
            Ok(_current) => free(pointer, &stream),
            Err(error) => context.record_err::<()>(Err(error)),
        }
    })
}

/// Frees `pointer` on its allocation `stream` once the work queued there has
/// finished, recording a failure on the context. The caller solely owns the
/// allocation and has ordered every other use of it before that work.
fn free(pointer: sys::CUdeviceptr, stream: &CudaStream) {
    let context = stream.context();
    // SAFETY: the caller owns the allocation and ordered its uses, and the
    // context is current for the free.
    let freed = bind_current(context)
        .and_then(|_current| unsafe { result::free_async(pointer, stream.cu_stream()) });
    context.record_err(freed);
}

#[cfg(test)]
mod tests;
