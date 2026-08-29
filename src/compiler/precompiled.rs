use super::{CompileKey, Compiler, Module};
use crate::{Error, PtxSource, Result};

impl Compiler {
    /// Loads or retrieves an embedded, ahead-of-time compiled PTX module.
    ///
    /// The module is accepted only on the exact compute capability declared by
    /// its source. This keeps device-specialized kernels out of accidental
    /// fallback paths.
    pub fn load_ptx(&self, source: PtxSource) -> Result<Module> {
        let target = source.compute_capability();
        if target != self.capability {
            return Err(Error::PtxArchitectureMismatch { target, device: self.capability });
        }
        let key = CompileKey::precompiled(source.name(), source.body(), target);
        let mut cache = self.cache.lock();
        if let Some(module) = cache.get(&key) {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(module.clone());
        }
        if source.body().as_bytes().contains(&0) {
            return Err(Error::InvalidEmbeddedPtx);
        }
        let mut bytes = source.body().as_bytes().to_vec();
        bytes.push(0);
        let image = mircuda_sys::CompiledPtx::from_bytes(bytes)?;
        let module = Module {
            native: self.context.native.load_ptx(&image)?,
        };
        cache.insert(key, module.clone());
        drop(cache);
        self.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(module)
    }
}
