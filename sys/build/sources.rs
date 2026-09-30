use std::{
    env,
    error::Error,
    path::{Path, PathBuf},
    process::Command,
};

const CUTLASS_VERSION: &str = "v4.4.2";
const CUTLASS_REPOSITORY: &str = "https://github.com/NVIDIA/cutlass.git";
const FLASH_ATTN_REVISION: &str = "2c839c33742309ec41e620bf837495ec9926c56e";
const FLASH_ATTN_REPOSITORY: &str = "https://github.com/vllm-project/flash-attention.git";

/// CUDA toolkit root; the installer's default is not on `PATH`.
pub fn cuda_home() -> PathBuf {
    env::var_os("CUDA_HOME").map_or_else(|| PathBuf::from("/usr/local/cuda"), PathBuf::from)
}

/// `NVCC`, else the toolkit's own compiler, so builds do not depend on `PATH`.
pub fn nvcc(cuda: &Path) -> PathBuf {
    env::var_os("NVCC").map_or_else(|| cuda.join("bin/nvcc"), PathBuf::from)
}

/// Pinned header-only CUTLASS, fetched once into the user cache when absent.
pub fn cutlass_dir() -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = env::var_os("MIRCUDA_CUTLASS_DIR") {
        return Ok(PathBuf::from(path));
    }
    let directory = cache()?.join(format!("cutlass-{CUTLASS_VERSION}"));
    if !directory.join("include/cutlass/cutlass.h").is_file() {
        fetch(
            &directory,
            &["clone", "--depth", "1", "--branch", CUTLASS_VERSION, CUTLASS_REPOSITORY],
        )?;
    }
    Ok(directory)
}

/// Pinned `FlashAttention` sources plus their CUTLASS submodule.
pub fn flash_attn_dir() -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = env::var_os("MIRCUDA_FLASH_ATTN_DIR") {
        return Ok(PathBuf::from(path));
    }
    let directory = cache()?.join(format!("flash-attention-{FLASH_ATTN_REVISION}"));
    if !directory.join("csrc/cutlass/include/cutlass/cutlass.h").is_file() {
        if !directory.join(".git").is_dir() {
            fetch(&directory, &["clone", "--filter=blob:none", FLASH_ATTN_REPOSITORY])?;
        }
        git(&directory, &["checkout", FLASH_ATTN_REVISION])?;
        git(&directory, &["submodule", "update", "--init", "--depth", "1", "csrc/cutlass"])?;
    }
    Ok(directory)
}

fn cache() -> Result<PathBuf, Box<dyn Error>> {
    let home = env::var_os("HOME").ok_or("HOME is unavailable")?;
    Ok(PathBuf::from(home).join(".cache/mircuda"))
}

fn fetch(directory: &Path, clone: &[&str]) -> Result<(), Box<dyn Error>> {
    println!(
        "cargo::warning=fetching pinned CUDA kernel sources into {}; set MIRCUDA_CUTLASS_DIR \
         and MIRCUDA_FLASH_ATTN_DIR to use existing checkouts",
        directory.display()
    );
    let parent = directory.parent().ok_or("kernel source cache has no parent")?;
    std::fs::create_dir_all(parent)?;
    let mut arguments = clone.to_vec();
    let target = directory.to_str().ok_or("kernel source cache path is not UTF-8")?;
    arguments.push(target);
    git(parent, &arguments)
}

fn git(directory: &Path, arguments: &[&str]) -> Result<(), Box<dyn Error>> {
    let status = Command::new("git").args(arguments).current_dir(directory).status();
    match status {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("`git {}` failed with {status}", arguments.join(" ")).into()),
        Err(error) => Err(format!(
            "git is required to fetch pinned CUDA kernel sources ({error}); install it or set \
             MIRCUDA_CUTLASS_DIR and MIRCUDA_FLASH_ATTN_DIR"
        )
        .into()),
    }
}
