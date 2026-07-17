use std::path::PathBuf;

use super::CompileOptions;

const CACHE_SCHEMA: &[u8] = b"mircuda-ptx-v1";

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct CompileKey(pub(super) [u8; 32]);

impl CompileKey {
    pub(super) fn new(
        source_name: &str,
        source: &str,
        options: &CompileOptions,
        architecture: &str,
        compiler_version: (i32, i32),
        include_paths: &[PathBuf],
    ) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(CACHE_SCHEMA);
        hasher.update(source_name.as_bytes());
        hasher.update(&[0]);
        hasher.update(source.as_bytes());
        hasher.update(&[u8::from(options.fast_math)]);
        hasher.update(&options.max_registers.unwrap_or_default().to_le_bytes());
        hasher.update(architecture.as_bytes());
        hasher.update(&compiler_version.0.to_le_bytes());
        hasher.update(&compiler_version.1.to_le_bytes());
        for path in include_paths {
            hasher.update(&[0]);
            hasher.update(path.to_string_lossy().as_bytes());
        }
        for option in &options.extra_options {
            hasher.update(&[0]);
            hasher.update(option.as_bytes());
        }
        Self(*hasher.finalize().as_bytes())
    }

    pub(super) fn file_name(self) -> String {
        format!("{}.ptx", blake3::Hash::from_bytes(self.0).to_hex())
    }
}
