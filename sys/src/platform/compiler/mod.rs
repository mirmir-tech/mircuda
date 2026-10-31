use std::{ffi::CString, sync::Arc};

use cudarc::{
    driver::{CudaContext, result, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts, sys as nvrtc_sys},
};

use super::driver::{Context, Stream};
use crate::Result;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CompileSpec {
    pub name: String,
    pub architecture: String,
    pub fast_math: bool,
    pub max_registers: Option<usize>,
    pub extra_options: Vec<String>,
}

pub fn compiler_version() -> Result<(i32, i32)> {
    let mut major = 0;
    let mut minor = 0;
    // SAFETY: NVRTC writes two initialized integers and retains no pointers.
    unsafe { nvrtc_sys::nvrtcVersion(&raw mut major, &raw mut minor) }.result()?;
    Ok((major, minor))
}

impl Context {
    pub fn compile_ptx(source: &str, spec: CompileSpec) -> Result<CompiledPtx> {
        let mut options = spec.extra_options;
        options.push(format!("--gpu-architecture={}", spec.architecture));
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                options,
                use_fast_math: Some(spec.fast_math),
                maxrregcount: spec.max_registers,
                name: Some(spec.name),
                ..CompileOptions::default()
            },
        )?;
        let image = ptx.as_bytes().ok_or(crate::Error::MissingPtx)?;
        CompiledPtx::from_bytes(image.to_vec())
    }

    pub fn load_ptx(&self, image: &CompiledPtx) -> Result<Module> {
        self.inner.bind_to_thread()?;
        // SAFETY: CompiledPtx owns a validated NUL-terminated image for this call.
        let handle = unsafe { result::module::load_data(image.as_bytes().as_ptr().cast()) }?;
        Ok(Module {
            inner: Arc::new(ModuleInner { handle, context: self.inner.clone() }),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledPtx(Vec<u8>);

impl CompiledPtx {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        if bytes.last().copied() != Some(0) {
            return Err(crate::Error::InvalidPtx);
        }
        Ok(Self(bytes))
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct Module {
    inner: Arc<ModuleInner>,
}

#[derive(Debug)]
struct ModuleInner {
    handle: sys::CUmodule,
    context: Arc<CudaContext>,
}

// SAFETY: the retained CudaContext binds itself before every driver operation.
unsafe impl Send for ModuleInner {}
// SAFETY: module lifetime is immutable and every operation binds the retained context.
unsafe impl Sync for ModuleInner {}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        self.context.record_err(self.context.bind_to_thread());
        // SAFETY: this is the sole owner of the loaded module handle.
        self.context.record_err(unsafe { result::module::unload(self.handle) });
    }
}

impl Module {
    pub fn kernel(&self, name: &str) -> Result<Kernel> {
        self.inner.context.bind_to_thread()?;
        let name = CString::new(name)?;
        // SAFETY: the module is live and the CString is NUL-terminated.
        let handle = unsafe { result::module::get_function(self.inner.handle, name) }?;
        Ok(Kernel { handle, module: self.clone() })
    }
}

#[derive(Clone, Debug)]
pub struct Kernel {
    handle: sys::CUfunction,
    module: Module,
}

// SAFETY: Kernel retains its immutable module and context for the complete lifetime.
unsafe impl Send for Kernel {}
// SAFETY: launches bind the module context and CUDA function handles are immutable.
unsafe impl Sync for Kernel {}

impl Kernel {
    pub(super) const fn handle(&self) -> sys::CUfunction {
        self.handle
    }

    pub fn set_max_dynamic_shared_memory_bytes(&self, bytes: u32) -> Result<()> {
        self.module.inner.context.bind_to_thread()?;
        let bytes = i32::try_from(bytes)?;
        // SAFETY: this kernel retains the live module owning the CUDA function.
        Ok(unsafe {
            result::function::set_function_attribute(
                self.handle,
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes,
            )
        }?)
    }

    pub(super) fn argument_pointers(
        &self,
        stream: &Stream,
        arguments: &mut [KernelArgument],
    ) -> Result<Vec<*mut std::ffi::c_void>> {
        if !Arc::ptr_eq(&self.module.inner.context, stream.inner.context()) {
            return Err(crate::Error::ContextMismatch);
        }
        let stream_handle = stream.inner.cu_stream();
        let context = Arc::as_ptr(stream.inner.context());
        for argument in arguments.iter() {
            if let KernelArgument::Pointer {
                stream: allocation_stream,
                context: allocation_context,
                cross_stream,
                ..
            } = argument
            {
                if *allocation_context != context {
                    return Err(crate::Error::ContextMismatch);
                }
                if *allocation_stream != stream_handle {
                    // SAFETY: generated KernelArguments retain the DeviceBuffer borrow.
                    unsafe { &**cross_stream }.store(true, std::sync::atomic::Ordering::Release);
                }
            }
        }
        self.module.inner.context.bind_to_thread()?;
        Ok(arguments.iter_mut().map(KernelArgument::as_mut_pointer).collect())
    }

    pub fn launch(
        &self,
        stream: &Stream,
        config: LaunchConfig,
        arguments: &mut [KernelArgument],
    ) -> Result<()> {
        let stream_handle = stream.inner.cu_stream();
        let mut pointers = self.argument_pointers(stream, arguments)?;
        // SAFETY: typed signatures own argument ABI; context and stream identity were checked.
        Ok(unsafe {
            result::launch_kernel(
                self.handle,
                config.grid,
                config.block,
                config.shared_memory_bytes,
                stream_handle,
                &mut pointers,
            )
        }?)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_memory_bytes: u32,
}

#[derive(Debug)]
pub enum KernelArgument {
    Pointer {
        value: sys::CUdeviceptr,
        stream: sys::CUstream,
        context: *const CudaContext,
        cross_stream: *const std::sync::atomic::AtomicBool,
    },
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    Usize(usize),
    Isize(isize),
    F32(f32),
    F64(f64),
}

impl KernelArgument {
    #[allow(clippy::match_same_arms)]
    const fn as_mut_pointer(&mut self) -> *mut std::ffi::c_void {
        match self {
            Self::Pointer { value, .. } => std::ptr::from_mut(value).cast(),
            Self::U8(value) => std::ptr::from_mut(value).cast(),
            Self::I8(value) => std::ptr::from_mut(value).cast(),
            Self::U16(value) => std::ptr::from_mut(value).cast(),
            Self::I16(value) => std::ptr::from_mut(value).cast(),
            Self::U32(value) => std::ptr::from_mut(value).cast(),
            Self::I32(value) => std::ptr::from_mut(value).cast(),
            Self::U64(value) => std::ptr::from_mut(value).cast(),
            Self::I64(value) => std::ptr::from_mut(value).cast(),
            Self::Usize(value) => std::ptr::from_mut(value).cast(),
            Self::Isize(value) => std::ptr::from_mut(value).cast(),
            Self::F32(value) => std::ptr::from_mut(value).cast(),
            Self::F64(value) => std::ptr::from_mut(value).cast(),
        }
    }
}
