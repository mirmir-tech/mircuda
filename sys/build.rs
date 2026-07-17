use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo::rerun-if-env-changed=MIRCUDA_CUTLASS_DIR");
    println!("cargo::rerun-if-env-changed=MIRCUDA_CUDA_ARCH");
    println!("cargo::rerun-if-env-changed=CUDA_HOME");
    println!("cargo::rerun-if-changed=native/cutlass_probe.cu");
    println!("cargo::rerun-if-changed=native/dense_sm120.cu");
    println!("cargo::rerun-if-changed=native/dense_sm120.cuh");
    println!("cargo::rerun-if-changed=native/dense_vector_sm120.cu");
    println!("cargo::rerun-if-changed=native/dense_vector_sm120.cuh");
    println!("cargo::rerun-if-changed=native/fp4_sm120.cu");
    println!("cargo::rerun-if-changed=native/fp4_vector_sm120.cu");
    println!("cargo::rerun-if-changed=native/fp8_vector_sm120.cu");
    println!("cargo::rerun-if-changed=native/grouped_fp4_sm120.cu");
    println!("cargo::rerun-if-changed=native/grouped_fp4_sm120.cuh");
    println!("cargo::rerun-if-changed=native/variable_grouped_fp4_sm120.cu");
    println!("cargo::rerun-if-changed=native/variable_grouped_fp4_sm120.cuh");
    if env::var_os("CARGO_FEATURE_CUTLASS").is_none()
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux")
    {
        return Ok(());
    }
    let cutlass = cutlass_dir()?;
    let include = cutlass.join("include");
    let util = cutlass.join("tools/util/include");
    if !include.join("cutlass/cutlass.h").is_file() {
        return Err(format!("CUTLASS 4.4.2 headers not found in {}", cutlass.display()).into());
    }
    let cuda =
        env::var_os("CUDA_HOME").map_or_else(|| PathBuf::from("/usr/local/cuda"), PathBuf::from);
    let arch = cuda_arch()?;
    let gencode = format!("-gencode=arch=compute_{arch},code=sm_{arch}");
    let configure = |build: &mut cc::Build| {
        build
            .cuda(true)
            .debug(false)
            .opt_level(3)
            .warnings(false)
            .cudart("shared")
            .include(&include)
            .include(&util)
            .include(cuda.join("include"))
            .flag("-std=c++17")
            .flag("--expt-relaxed-constexpr")
            .flag("-diag-suppress=20013")
            .flag("-diag-suppress=20015")
            .flag("-diag-suppress=2908")
            .flag(&gencode);
    };
    let mut probe = cc::Build::new();
    configure(&mut probe);
    probe.file("native/cutlass_probe.cu").compile("mircuda_cutlass_probe");
    let mut dense = cc::Build::new();
    configure(&mut dense);
    dense.file("native/dense_sm120.cu").compile("mircuda_cutlass_dense");
    let mut dense_vector = cc::Build::new();
    configure(&mut dense_vector);
    dense_vector
        .file("native/dense_vector_sm120.cu")
        .compile("mircuda_cutlass_dense_vector");
    let mut fp4 = cc::Build::new();
    configure(&mut fp4);
    fp4.file("native/fp4_sm120.cu").compile("mircuda_cutlass_fp4");
    let mut fp4_vector = cc::Build::new();
    configure(&mut fp4_vector);
    fp4_vector
        .file("native/fp4_vector_sm120.cu")
        .compile("mircuda_cutlass_fp4_vector");
    let mut fp8_vector = cc::Build::new();
    configure(&mut fp8_vector);
    fp8_vector
        .file("native/fp8_vector_sm120.cu")
        .compile("mircuda_cutlass_fp8_vector");
    let mut grouped_fp4 = cc::Build::new();
    configure(&mut grouped_fp4);
    grouped_fp4
        .file("native/grouped_fp4_sm120.cu")
        .compile("mircuda_cutlass_grouped_fp4");
    let mut variable_grouped_fp4 = cc::Build::new();
    configure(&mut variable_grouped_fp4);
    variable_grouped_fp4
        .file("native/variable_grouped_fp4_sm120.cu")
        .compile("mircuda_cutlass_variable_grouped_fp4");
    Ok(())
}

fn cuda_arch() -> Result<String, Box<dyn Error>> {
    let arch = env::var("MIRCUDA_CUDA_ARCH").unwrap_or_else(|_| String::from("120f"));
    if arch.chars().all(|character| character.is_ascii_digit())
        || (arch.ends_with(['a', 'f'])
            && arch[..arch.len() - 1].chars().all(|character| character.is_ascii_digit()))
    {
        Ok(arch)
    } else {
        Err(format!("invalid MIRCUDA_CUDA_ARCH: {arch}").into())
    }
}

fn cutlass_dir() -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = env::var_os("MIRCUDA_CUTLASS_DIR") {
        return Ok(PathBuf::from(path));
    }
    let home = env::var_os("HOME").ok_or("HOME is unavailable")?;
    Ok(PathBuf::from(home).join(".cache/mircuda/cutlass-v4.4.2"))
}
