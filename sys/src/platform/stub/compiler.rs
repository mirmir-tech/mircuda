use super::{Context, Module, unsupported};
use crate::{Error, Result};

pub const fn compiler_version() -> Result<(i32, i32)> {
    Err(unsupported())
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CompileSpec {
    pub name: String,
    pub architecture: String,
    pub fast_math: bool,
    pub max_registers: Option<usize>,
    pub extra_options: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledPtx(Vec<u8>);

impl CompiledPtx {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        if bytes.last().copied() != Some(0) {
            return Err(Error::InvalidPtx);
        }
        Ok(Self(bytes))
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl Context {
    pub fn compile_ptx(_source: &str, _spec: CompileSpec) -> Result<CompiledPtx> {
        Err(unsupported())
    }

    pub const fn load_ptx(&self, _image: &CompiledPtx) -> Result<Module> {
        Err(unsupported())
    }
}
