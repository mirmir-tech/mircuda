use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use super::CompileKey;

static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub(super) struct ArtifactCache {
    directory: PathBuf,
}

impl ArtifactCache {
    pub(super) const fn new(directory: PathBuf) -> Self {
        Self { directory }
    }

    pub(super) fn read(&self, key: CompileKey) -> io::Result<Option<Vec<u8>>> {
        match fs::read(self.path(key)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    pub(super) fn write(&self, key: CompileKey, bytes: &[u8]) -> io::Result<()> {
        fs::create_dir_all(&self.directory)?;
        let target = self.path(key);
        let temporary = temporary_path(&target);
        let mut file = fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        match fs::rename(&temporary, &target) {
            Ok(()) => Ok(()),
            Err(_error) if target.exists() => {
                fs::remove_file(temporary)?;
                Ok(())
            },
            Err(error) => Err(error),
        }
    }

    pub(super) fn remove(&self, key: CompileKey) -> io::Result<()> {
        match fs::remove_file(self.path(key)) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    fn path(&self, key: CompileKey) -> PathBuf {
        self.directory.join(key.file_name())
    }
}

fn temporary_path(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_owned();
    let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    name.push(format!(".{}.{sequence}.tmp", std::process::id()));
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_artifact_is_a_cache_miss() -> io::Result<()> {
        let directory = std::env::temp_dir().join(format!("mircuda-cache-{}", std::process::id()));
        let cache = ArtifactCache::new(directory.clone());
        let key = CompileKey([7; 32]);
        assert_eq!(cache.read(key)?, None);
        cache.write(key, b"ptx\0")?;
        assert_eq!(cache.read(key)?, Some(b"ptx\0".to_vec()));
        cache.remove(key)?;
        fs::remove_dir_all(directory)?;
        Ok(())
    }
}
