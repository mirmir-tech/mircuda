use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, sys::CUdevice_attribute};

use crate::Result;

#[derive(Debug)]
pub struct Driver;

impl Driver {
    pub fn initialize() -> Result<Self> {
        let _ = CudaContext::device_count()?;
        Ok(Self)
    }

    pub fn device_count(&self) -> Result<usize> {
        Ok(usize::try_from(CudaContext::device_count()?)?)
    }

    pub fn create_context(&self, ordinal: usize) -> Result<Context> {
        Ok(Context { inner: CudaContext::new(ordinal)? })
    }
}

#[derive(Clone, Debug)]
pub struct Context {
    pub(super) inner: Arc<CudaContext>,
}

impl Context {
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        Ok(self.inner.mem_get_info()?)
    }

    pub fn device_info(&self) -> Result<DeviceInfo> {
        Ok(DeviceInfo {
            ordinal: self.inner.ordinal(),
            name: self.inner.name()?,
            compute_capability: self.inner.compute_capability()?,
            multiprocessor_count: self
                .inner
                .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            total_memory: self.inner.total_mem()?,
            memory_pools: self.inner.has_async_alloc(),
            integrated: self.inner.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)?
                != 0,
        })
    }

    pub fn create_stream(&self) -> Result<Stream> {
        Ok(Stream { inner: self.inner.new_stream()? })
    }
}

#[derive(Clone, Debug)]
pub struct Stream {
    pub(super) inner: Arc<CudaStream>,
}

impl Stream {
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.inner.synchronize()?)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub multiprocessor_count: i32,
    pub total_memory: usize,
    pub memory_pools: bool,
    pub integrated: bool,
}
