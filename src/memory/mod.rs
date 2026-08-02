use std::{marker::PhantomData, sync::Arc};

use crate::{Context, Error, Result, Stream};

mod transfer;

mod sealed {
    pub trait Sealed {}
}

/// Plain scalar that can be stored in CUDA device memory.
pub trait DeviceElement: sealed::Sealed + Copy + Send + Sync + 'static {}

macro_rules! device_elements {
    ($($element:ty),+ $(,)?) => {$ (
        impl sealed::Sealed for $element {}
        impl DeviceElement for $element {}
    )+ };
}

device_elements!(
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    usize,
    isize,
    f32,
    f64,
    half::f16,
    half::bf16,
);

/// Stream-ordered allocation from an explicit CUDA memory pool.
#[derive(Clone, Debug)]
pub struct MemoryPool {
    native: mircuda_sys::MemoryPool,
}

/// Current device memory retained and used by a CUDA pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryPoolStats {
    /// Bytes currently reserved from the device.
    pub reserved: u64,
    /// Bytes currently backing live allocations.
    pub used: u64,
}

/// Shared handle to one device allocation bound to its allocation stream.
///
/// Cloning this handle retains the same CUDA allocation. It does not allocate,
/// transfer, or copy device memory.
#[derive(Clone, Debug)]
pub struct DeviceBuffer<T: DeviceElement> {
    pub(crate) native: Arc<mircuda_sys::DeviceBuffer>,
    len: usize,
    marker: PhantomData<T>,
}

/// Page-locked host allocation for asynchronous CUDA transfers.
#[derive(Debug)]
pub struct PinnedBuffer<T: DeviceElement> {
    pub(crate) native: mircuda_sys::PinnedBuffer,
    len: usize,
    marker: PhantomData<T>,
}

impl Context {
    /// Returns the CUDA-owned default pool with explicit policy controls.
    pub fn default_memory_pool(&self) -> Result<MemoryPool> {
        Ok(MemoryPool {
            native: self.native.default_memory_pool()?,
        })
    }

    /// Allocates zero-initialized page-locked host memory.
    pub fn allocate_pinned<T: DeviceElement>(&self, len: usize) -> Result<PinnedBuffer<T>> {
        let bytes = allocation_bytes::<T>(len)?;
        Ok(PinnedBuffer {
            native: self.native.allocate_pinned(bytes)?,
            len,
            marker: PhantomData,
        })
    }
}

impl MemoryPool {
    /// Retains up to `bytes` of unused memory before returning it to CUDA.
    pub fn set_release_threshold(&self, bytes: u64) -> Result<()> {
        Ok(self.native.set_release_threshold(bytes)?)
    }

    /// Returns current pool accounting without synchronizing execution streams.
    pub fn stats(&self) -> Result<MemoryPoolStats> {
        let stats = self.native.stats()?;
        Ok(MemoryPoolStats {
            reserved: stats.reserved,
            used: stats.used,
        })
    }

    /// Releases unused pool storage while retaining at least `bytes`.
    pub fn trim_to(&self, bytes: usize) -> Result<()> {
        Ok(self.native.trim_to(bytes)?)
    }

    /// Enqueues an uninitialized allocation on `stream`.
    pub fn allocate<T: DeviceElement>(
        &self,
        stream: &Stream,
        len: usize,
    ) -> Result<DeviceBuffer<T>> {
        self.allocate_with(stream, len, false)
    }

    /// Enqueues a zero-initialized allocation on `stream`.
    pub fn allocate_zeroed<T: DeviceElement>(
        &self,
        stream: &Stream,
        len: usize,
    ) -> Result<DeviceBuffer<T>> {
        self.allocate_with(stream, len, true)
    }

    fn allocate_with<T: DeviceElement>(
        &self,
        stream: &Stream,
        len: usize,
        zeroed: bool,
    ) -> Result<DeviceBuffer<T>> {
        let bytes = allocation_bytes::<T>(len)?;
        let native = match self.native.allocate(&stream.native, bytes, zeroed) {
            Ok(native) => native,
            Err(source) => return Err(Error::DeviceAllocation { bytes, source }),
        };
        Ok(DeviceBuffer {
            native: Arc::new(native),
            len,
            marker: PhantomData,
        })
    }
}

impl<T: DeviceElement> DeviceBuffer<T> {
    /// Number of typed elements in the allocation.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether this allocation contains no elements. Allocations reject this state.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of allocated bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.native.bytes()
    }

    /// Returns a differently typed view over the same device allocation.
    ///
    /// The view retains the allocation and performs no transfer or conversion.
    pub fn reinterpret<U: DeviceElement>(&self) -> Result<DeviceBuffer<U>> {
        let element_bytes = std::mem::size_of::<U>();
        let len = self.bytes().checked_div(element_bytes).ok_or(Error::InvalidDeviceView)?;
        if len == 0 || len.checked_mul(element_bytes) != Some(self.bytes()) {
            return Err(Error::InvalidDeviceView);
        }
        Ok(DeviceBuffer {
            native: self.native.clone(),
            len,
            marker: PhantomData,
        })
    }
}

impl<T: DeviceElement> PinnedBuffer<T> {
    /// Number of typed elements in the allocation.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether this allocation contains no elements. Allocations reject this state.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies ordinary host data into this pinned allocation after prior transfers finish.
    pub fn copy_from_slice(&mut self, values: &[T]) -> Result<()> {
        validate_lengths(values.len(), self.len)?;
        Ok(self.native.with_mut_slice(self.len, |target| target.copy_from_slice(values))?)
    }

    /// Copies a typed prefix while retaining the remaining pinned allocation.
    pub fn write_prefix(&mut self, values: &[T]) -> Result<()> {
        if values.len() > self.len {
            return Err(Error::LengthMismatch {
                source_len: values.len(),
                target_len: self.len,
            });
        }
        Ok(self.native.with_mut_slice(self.len, |target| {
            target[..values.len()].copy_from_slice(values);
        })?)
    }

    /// Provides mutable byte access after waiting for prior transfers.
    pub fn with_bytes_mut<R>(&mut self, write: impl FnOnce(&mut [u8]) -> R) -> Result<R> {
        Ok(self.native.with_mut_slice(self.native.bytes(), write)?)
    }

    /// Provides immutable byte access after waiting for prior transfers.
    pub fn with_bytes<R>(&self, read: impl FnOnce(&[u8]) -> R) -> Result<R> {
        Ok(self.native.with_slice(self.native.bytes(), read)?)
    }

    /// Performs an explicit host read after waiting for prior transfers.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        Ok(self.native.with_slice(self.len, <[T]>::to_vec)?)
    }
}

fn allocation_bytes<T>(len: usize) -> Result<usize> {
    if len == 0 {
        return Err(Error::EmptyAllocation);
    }
    let element_bytes = std::mem::size_of::<T>();
    len.checked_mul(element_bytes)
        .ok_or(Error::AllocationOverflow { elements: len, element_bytes })
}

const fn validate_lengths(source: usize, target: usize) -> Result<()> {
    if source == target {
        Ok(())
    } else {
        Err(Error::LengthMismatch { source_len: source, target_len: target })
    }
}
