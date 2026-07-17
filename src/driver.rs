use crate::Result;

/// Initialized CUDA Driver API entry point.
#[derive(Debug)]
pub struct Driver {
    native: mircuda_sys::Driver,
}

impl Driver {
    /// Initializes the CUDA driver on the current host.
    pub fn initialize() -> Result<Self> {
        Ok(Self {
            native: mircuda_sys::Driver::initialize()?,
        })
    }

    /// Enumerates CUDA devices without creating execution contexts.
    pub fn devices(&self) -> Result<Vec<Device>> {
        let count = self.native.device_count()?;
        Ok((0..count).map(|ordinal| Device { ordinal }).collect())
    }

    /// Retains the primary context for a selected device.
    pub fn create_context(&self, device: Device) -> Result<Context> {
        Ok(Context {
            native: self.native.create_context(device.ordinal)?,
        })
    }
}

/// Stable ordinal identifying a CUDA device.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Device {
    ordinal: usize,
}

impl Device {
    /// Returns the CUDA device ordinal.
    #[must_use]
    pub const fn ordinal(self) -> usize {
        self.ordinal
    }
}

/// Hardware properties used for kernel and execution-plan selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    /// CUDA device ordinal.
    pub ordinal: usize,
    /// Driver-reported device name.
    pub name: String,
    /// Compute capability `(major, minor)`.
    pub compute_capability: (i32, i32),
    /// Number of streaming multiprocessors available to execution plans.
    pub multiprocessor_count: u32,
    /// Total device memory in bytes.
    pub total_memory: usize,
    /// Whether stream-ordered allocation is supported.
    pub memory_pools: bool,
    /// Whether the CUDA device shares its physical memory subsystem with the host.
    pub integrated: bool,
}

/// Retained CUDA primary context.
#[derive(Clone, Debug)]
pub struct Context {
    pub(crate) native: mircuda_sys::Context,
}

impl Context {
    /// Returns free and total device memory in bytes.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        Ok(self.native.memory_info()?)
    }

    /// Queries immutable properties of this context's device.
    pub fn device_info(&self) -> Result<DeviceInfo> {
        let info = self.native.device_info()?;
        Ok(DeviceInfo {
            ordinal: info.ordinal,
            name: info.name,
            compute_capability: info.compute_capability,
            multiprocessor_count: u32::try_from(info.multiprocessor_count)?,
            total_memory: info.total_memory,
            memory_pools: info.memory_pools,
            integrated: info.integrated,
        })
    }

    /// Creates an independent non-blocking stream.
    pub fn create_stream(&self) -> Result<Stream> {
        Ok(Stream { native: self.native.create_stream()? })
    }
}

/// Explicit non-blocking CUDA execution stream.
#[derive(Clone, Debug)]
pub struct Stream {
    pub(crate) native: mircuda_sys::Stream,
}

impl Stream {
    /// Blocks the host until all queued stream work has completed.
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.native.synchronize()?)
    }
}
