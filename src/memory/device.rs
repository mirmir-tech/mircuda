use std::{marker::PhantomData, sync::Arc};

use super::{DeviceBuffer, DeviceElement};
use crate::{Error, Result};

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

    /// Retains a nonempty element range without copying or synchronizing.
    /// Views share ownership and stream-use tracking with their allocation.
    pub fn slice(&self, range: std::ops::Range<usize>) -> Result<Self> {
        if range.is_empty() || range.end > self.len {
            return Err(Error::InvalidDeviceView);
        }
        let element_bytes = std::mem::size_of::<T>();
        let start = range.start.checked_mul(element_bytes).ok_or(Error::InvalidDeviceView)?;
        let len = range.end - range.start;
        let bytes = len.checked_mul(element_bytes).ok_or(Error::InvalidDeviceView)?;
        Ok(Self {
            native: Arc::new(self.native.slice(start, bytes)?),
            len,
            marker: PhantomData,
        })
    }

    /// Returns a differently typed view over the same device allocation.
    ///
    /// The view retains the allocation and performs no transfer or conversion.
    pub fn reinterpret<U: DeviceElement>(&self) -> Result<DeviceBuffer<U>> {
        let element_bytes = std::mem::size_of::<U>();
        let len = self.bytes().checked_div(element_bytes).ok_or(Error::InvalidDeviceView)?;
        if len == 0
            || len.checked_mul(element_bytes) != Some(self.bytes())
            || !self.native.is_aligned(std::mem::align_of::<U>())
        {
            return Err(Error::InvalidDeviceView);
        }
        Ok(DeviceBuffer {
            native: self.native.clone(),
            len,
            marker: PhantomData,
        })
    }
}
