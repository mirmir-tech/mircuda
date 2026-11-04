use std::ffi::{CStr, CString};

use cudarc::nvrtc::{CompileError, result as nvrtc, sys as nvrtc_sys};

use super::{CompileSpec, Module};
use crate::{Error, Result, platform::driver::Context};

/// ELF image NVRTC compiled for the device's real architecture.
///
/// Loading it needs no driver JIT, so modules compiled by a CUDA 13 toolkit
/// newer than the installed driver still load; PTX of a newer ISA would not.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledCubin(Vec<u8>);

impl CompiledCubin {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        if !bytes.starts_with(b"\x7fELF") {
            return Err(Error::InvalidCubin);
        }
        Ok(Self(bytes))
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl Context {
    pub fn compile_cubin(source: &str, spec: CompileSpec) -> Result<CompiledCubin> {
        let program = Program::create(source, &spec.name)?;
        program.compile(spec.options())?;
        program.cubin()
    }

    pub fn load_cubin(&self, image: &CompiledCubin) -> Result<Module> {
        self.load_image(image.as_bytes())
    }
}

impl CompileSpec {
    /// NVRTC options in the order cudarc's PTX helper passed them, so the
    /// compiler sees the same command line as before.
    fn options(self) -> Vec<String> {
        let mut options = Vec::new();
        if self.fast_math {
            options.push("--fmad=true".into());
        }
        if let Some(count) = self.max_registers {
            options.push(format!("--maxrregcount={count}"));
        }
        options.extend(self.extra_options);
        options.push(format!("--gpu-architecture={}", self.architecture));
        options
    }
}

struct Program {
    handle: nvrtc_sys::nvrtcProgram,
    // NVRTC borrows both strings until the program is destroyed.
    _source: CString,
    _name: CString,
}

impl Program {
    fn create(source: &str, name: &str) -> Result<Self> {
        let source = CString::new(source)?;
        let name = CString::new(name)?;
        let handle = nvrtc::create_program(&source, Some(&name))?;
        Ok(Self { handle, _source: source, _name: name })
    }

    fn compile(&self, options: Vec<String>) -> Result<()> {
        // SAFETY: the program is live until `Drop`.
        let Err(nvrtc) = (unsafe { nvrtc::compile_program(self.handle, &options) }) else {
            return Ok(());
        };
        // SAFETY: the program is live.
        let log = unsafe { nvrtc::get_program_log(self.handle) }?;
        // SAFETY: NVRTC NUL-terminates the log, which outlives this borrow.
        let log = unsafe { CStr::from_ptr(log.as_ptr()) }.to_owned();
        Err(CompileError::CompileError { nvrtc, options, log }.into())
    }

    fn cubin(&self) -> Result<CompiledCubin> {
        let mut size = 0;
        // SAFETY: the program compiled successfully and writes one size.
        unsafe { nvrtc_sys::nvrtcGetCUBINSize(self.handle, &raw mut size) }.result()?;
        let mut bytes = vec![0_u8; size];
        // SAFETY: `bytes` holds exactly the size NVRTC reported.
        unsafe { nvrtc_sys::nvrtcGetCUBIN(self.handle, bytes.as_mut_ptr().cast()) }.result()?;
        CompiledCubin::from_bytes(bytes)
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: this is the sole owner of the program handle. A destructor
        // cannot report a failed release.
        let _ = unsafe { nvrtc::destroy_program(self.handle) };
    }
}
