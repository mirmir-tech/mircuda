mod architecture;
mod cache;
mod key;

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use architecture::target_architecture;
use cache::ArtifactCache;
use key::CompileKey;
use parking_lot::Mutex;

use crate::{Context, KernelSignature, KernelSource, Result, TypedKernel};

/// NVRTC policy used to build a CUDA module for the current device.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct CompileOptions {
    /// Enables NVRTC fast-math transformations.
    pub fast_math: bool,
    /// Optional maximum register count per thread.
    pub max_registers: Option<usize>,
    /// Additional explicit NVRTC options.
    pub extra_options: Vec<String>,
}

/// Context-bound compiler construction policy.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompilerConfig {
    /// CUDA toolkit or consumer header roots passed to NVRTC.
    pub include_paths: Vec<PathBuf>,
    /// Optional persistent PTX artifact directory.
    pub cache_directory: Option<PathBuf>,
}

/// In-memory and persistent module-cache counters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompileCacheStats {
    /// Number of lookups served without invoking NVRTC.
    pub hits: u64,
    /// Number of hits loaded from persistent PTX artifacts.
    pub persistent_hits: u64,
    /// Number of NVRTC compilations.
    pub misses: u64,
    /// Number of best-effort persistent cache I/O or load failures.
    pub persistent_errors: u64,
    /// Number of modules retained by this compiler.
    pub entries: usize,
}

/// Context-bound NVRTC compiler and loaded-module cache.
#[derive(Debug)]
pub struct Compiler {
    context: Context,
    architecture: String,
    nvrtc_version: (i32, i32),
    config: CompilerConfig,
    persistent: Option<ArtifactCache>,
    cache: Mutex<HashMap<CompileKey, Module>>,
    hits: AtomicU64,
    persistent_hits: AtomicU64,
    misses: AtomicU64,
    persistent_errors: AtomicU64,
}

impl Compiler {
    /// Creates a compiler targeting the context device's virtual architecture.
    pub fn new(context: Context) -> Result<Self> {
        Self::with_config(context, CompilerConfig::default())
    }

    /// Creates a compiler with explicit CUDA toolkit or consumer header paths.
    pub fn with_include_paths(
        context: Context,
        include_paths: impl IntoIterator<Item = PathBuf>,
    ) -> Result<Self> {
        Self::with_config(
            context,
            CompilerConfig {
                include_paths: include_paths.into_iter().collect(),
                cache_directory: None,
            },
        )
    }

    /// Creates a compiler with explicit include and persistent-cache policy.
    pub fn with_config(context: Context, config: CompilerConfig) -> Result<Self> {
        let capability = context.device_info()?.compute_capability;
        let nvrtc_version = mircuda_sys::compiler_version()?;
        let persistent = config.cache_directory.clone().map(ArtifactCache::new);
        Ok(Self {
            context,
            architecture: target_architecture(capability),
            nvrtc_version,
            config,
            persistent,
            cache: Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            persistent_hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            persistent_errors: AtomicU64::new(0),
        })
    }

    /// Compiles or retrieves a context-bound module for `source` and `options`.
    pub fn compile(&self, source: KernelSource, options: &CompileOptions) -> Result<Module> {
        let source_text = source.source();
        let key = CompileKey::new(
            source.name(),
            &source_text,
            options,
            &self.architecture,
            self.nvrtc_version,
            &self.config.include_paths,
        );
        let mut cache = self.cache.lock();
        if let Some(module) = cache.get(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(module.clone());
        }
        if let Some(module) = self.load_persistent(key) {
            cache.insert(key, module.clone());
            self.hits.fetch_add(1, Ordering::Relaxed);
            self.persistent_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(module);
        }
        let image = mircuda_sys::Context::compile_ptx(&source_text, self.spec(source, options))?;
        let native = self.context.native.load_ptx(&image)?;
        self.store_persistent(key, image.as_bytes());
        let module = Module { native };
        cache.insert(key, module.clone());
        drop(cache);
        self.misses.fetch_add(1, Ordering::Relaxed);
        Ok(module)
    }

    /// Returns cache counters without invoking CUDA.
    pub fn cache_stats(&self) -> CompileCacheStats {
        CompileCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            persistent_hits: self.persistent_hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            persistent_errors: self.persistent_errors.load(Ordering::Relaxed),
            entries: self.cache.lock().len(),
        }
    }

    fn spec(&self, source: KernelSource, options: &CompileOptions) -> mircuda_sys::CompileSpec {
        let mut extra_options = options.extra_options.clone();
        extra_options.extend(
            self.config
                .include_paths
                .iter()
                .map(|path| format!("--include-path={}", path.display())),
        );
        mircuda_sys::CompileSpec {
            name: source.name().into(),
            architecture: self.architecture.clone(),
            fast_math: options.fast_math,
            max_registers: options.max_registers,
            extra_options,
        }
    }

    fn load_persistent(&self, key: CompileKey) -> Option<Module> {
        let persistent = self.persistent.as_ref()?;
        let bytes = match persistent.read(key) {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return None,
            Err(_) => {
                self.persistent_errors.fetch_add(1, Ordering::Relaxed);
                return None;
            },
        };
        let Ok(image) = mircuda_sys::CompiledPtx::from_bytes(bytes) else {
            return self.reject_persistent(persistent, key);
        };
        if let Ok(native) = self.context.native.load_ptx(&image) {
            return Some(Module { native });
        }
        self.reject_persistent(persistent, key)
    }

    fn reject_persistent(&self, persistent: &ArtifactCache, key: CompileKey) -> Option<Module> {
        self.persistent_errors.fetch_add(1, Ordering::Relaxed);
        if persistent.remove(key).is_err() {
            self.persistent_errors.fetch_add(1, Ordering::Relaxed);
        }
        None
    }

    fn store_persistent(&self, key: CompileKey, bytes: &[u8]) {
        if self.persistent.as_ref().is_some_and(|cache| cache.write(key, bytes).is_err()) {
            self.persistent_errors.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Loaded CUDA module retained by its originating context.
#[derive(Clone, Debug)]
pub struct Module {
    native: mircuda_sys::Module,
}

impl Module {
    /// Resolves the symbol declared by a checked [`KernelSignature`].
    pub fn kernel<S: KernelSignature>(&self) -> Result<TypedKernel<S>> {
        Ok(TypedKernel::new(self.native.kernel(S::NAME)?))
    }
}
