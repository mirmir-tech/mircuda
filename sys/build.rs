use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo::rerun-if-env-changed=MIRCUDA_CUTLASS_DIR");
    println!("cargo::rerun-if-env-changed=MIRCUDA_FLASH_ATTN_DIR");
    println!("cargo::rerun-if-env-changed=MIRCUDA_CUDA_ARCH");
    println!("cargo::rerun-if-env-changed=MIRCUDA_MARLIN_CUDA_ARCH");
    println!("cargo::rerun-if-env-changed=CUDA_HOME");
    rerun_sources();
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return Ok(());
    }
    let cuda =
        env::var_os("CUDA_HOME").map_or_else(|| PathBuf::from("/usr/local/cuda"), PathBuf::from);
    let arch = cuda_arch()?;
    let gencode = format!("-gencode=arch=compute_{arch},code=sm_{arch}");
    let marlin_arch = env::var("MIRCUDA_MARLIN_CUDA_ARCH").ok();
    let configure = |build: &mut cc::Build| {
        build
            .cuda(true)
            .debug(false)
            .opt_level(3)
            .warnings(false)
            .cudart("shared")
            .include(cuda.join("include"))
            .flag("-std=c++17")
            .flag("--expt-relaxed-constexpr")
            .flag("-diag-suppress=20013")
            .flag("-diag-suppress=20015")
            .flag("-diag-suppress=2908")
            .flag(&gencode);
    };
    if env::var_os("CARGO_FEATURE_CUBLASLT").is_some() {
        let mut vendor = cc::Build::new();
        configure(&mut vendor);
        vendor
            .file("native/cublas_dense.cu")
            .file("native/cublaslt_dense.cu")
            .file("native/cublaslt_fp8.cu")
            .compile("mircuda_vendor_dense");
        println!("cargo::rustc-link-lib=dylib=cublas");
        println!("cargo::rustc-link-lib=dylib=cublasLt");
    }
    if env::var_os("CARGO_FEATURE_MARLIN").is_some() {
        let mut marlin = cc::Build::new();
        configure(&mut marlin);
        if let Some(arch) = &marlin_arch {
            marlin.flag(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        }
        marlin
            .include("native/marlin/vendor")
            .file("native/marlin/repack.cu")
            .file("native/marlin/dense_kernel.cu")
            .file("native/marlin/moe_kernel.cu")
            .file("native/marlin/mxfp4_moe_kernel.cu")
            .compile("mircuda_marlin");
    }
    if env::var_os("CARGO_FEATURE_CUTLASS").is_none() {
        return Ok(());
    }
    compile_cutlass(&configure)?;
    Ok(())
}

fn rerun_sources() {
    for source in [
        "cutlass_probe.cu",
        "cublas_dense.cu",
        "cublaslt_dense.cu",
        "cublaslt_fp8.cu",
        "dense_sm120.cu",
        "dense_sm120.cuh",
        "dense_vector_sm120.cu",
        "dense_vector_sm120.cuh",
        "fp4_sm120.cu",
        "mxfp8_sm120.cu",
        "fp4_vector_sm120.cu",
        "fp8_vector_sm120.cu",
        "scaled_fp8_sm120.cu",
        "scaled_fp8_sm120.cuh",
        "fmha_sm80.cu",
        "flash_attn2_bf16.cu",
        "grouped_fp4_sm120.cu",
        "grouped_fp4_sm120.cuh",
        "variable_grouped_fp4_sm120.cu",
        "variable_grouped_fp4_sm120.cuh",
        "variable_grouped_fp4_pair_sm120.cuh",
        "variable_grouped_bf16_sm120.cu",
        "variable_grouped_bf16_sm120.cuh",
        "marlin/repack.cu",
        "marlin/dense_kernel.cu",
        "marlin/moe_kernel.cu",
        "marlin/mxfp4_moe_kernel.cu",
        "marlin/vendor/dequant.h",
        "marlin/vendor/marlin.cuh",
        "marlin/vendor/marlin_dtypes.cuh",
        "marlin/vendor/marlin_mma.h",
        "marlin/vendor/moe_template.h",
        "marlin/vendor/dense_template.h",
        "marlin/vendor/scalar_type.hpp",
    ] {
        println!("cargo::rerun-if-changed=native/{source}");
    }
}

fn compile_cutlass(configure: &impl Fn(&mut cc::Build)) -> Result<(), Box<dyn Error>> {
    let cutlass = cutlass_dir()?;
    let include = cutlass.join("include");
    let util = cutlass.join("tools/util/include");
    if !include.join("cutlass/cutlass.h").is_file() {
        return Err(format!("CUTLASS 4.4.2 headers not found in {}", cutlass.display()).into());
    }
    let configure_cutlass = |build: &mut cc::Build| {
        configure(build);
        build.include(&include).include(&util);
    };
    let mut probe = cc::Build::new();
    configure_cutlass(&mut probe);
    probe.file("native/cutlass_probe.cu").compile("mircuda_cutlass_probe");
    let mut dense = cc::Build::new();
    configure_cutlass(&mut dense);
    dense.file("native/dense_sm120.cu").compile("mircuda_cutlass_dense");
    let mut fmha = cc::Build::new();
    configure_cutlass(&mut fmha);
    fmha.include(cutlass.join("examples/41_fused_multi_head_attention"));
    fmha.file("native/fmha_sm80.cu").compile("mircuda_cutlass_fmha");
    compile_flash_attention(configure)?;
    let mut dense_vector = cc::Build::new();
    configure_cutlass(&mut dense_vector);
    dense_vector
        .file("native/dense_vector_sm120.cu")
        .compile("mircuda_cutlass_dense_vector");
    let mut fp4 = cc::Build::new();
    configure_cutlass(&mut fp4);
    fp4.file("native/fp4_sm120.cu").compile("mircuda_cutlass_fp4");
    let mut mxfp8 = cc::Build::new();
    configure_cutlass(&mut mxfp8);
    mxfp8.file("native/mxfp8_sm120.cu").compile("mircuda_cutlass_mxfp8");
    let mut fp4_vector = cc::Build::new();
    configure_cutlass(&mut fp4_vector);
    fp4_vector
        .file("native/fp4_vector_sm120.cu")
        .compile("mircuda_cutlass_fp4_vector");
    let mut fp8_vector = cc::Build::new();
    configure_cutlass(&mut fp8_vector);
    fp8_vector
        .file("native/fp8_vector_sm120.cu")
        .compile("mircuda_cutlass_fp8_vector");
    let mut scaled_fp8 = cc::Build::new();
    configure_cutlass(&mut scaled_fp8);
    scaled_fp8
        .file("native/scaled_fp8_sm120.cu")
        .compile("mircuda_cutlass_scaled_fp8");
    let mut grouped_fp4 = cc::Build::new();
    configure_cutlass(&mut grouped_fp4);
    grouped_fp4
        .file("native/grouped_fp4_sm120.cu")
        .compile("mircuda_cutlass_grouped_fp4");
    let mut variable_grouped_fp4 = cc::Build::new();
    configure_cutlass(&mut variable_grouped_fp4);
    variable_grouped_fp4
        .file("native/variable_grouped_fp4_sm120.cu")
        .compile("mircuda_cutlass_variable_grouped_fp4");
    let mut variable_grouped_bf16 = cc::Build::new();
    configure_cutlass(&mut variable_grouped_bf16);
    variable_grouped_bf16
        .file("native/variable_grouped_bf16_sm120.cu")
        .compile("mircuda_cutlass_variable_grouped_bf16");
    Ok(())
}

fn compile_flash_attention(configure: &impl Fn(&mut cc::Build)) -> Result<(), Box<dyn Error>> {
    let flash = flash_attn_dir()?;
    let source = flash.join("csrc/flash_attn/src");
    let cutlass = flash.join("csrc/cutlass/include");
    let specialization64 = source.join("flash_fwd_split_hdim64_bf16_causal_sm80.cu");
    let specialization128 = source.join("flash_fwd_split_hdim128_bf16_causal_sm80.cu");
    let specialization256 = source.join("flash_fwd_split_hdim256_bf16_causal_sm80.cu");
    let decode64 = source.join("flash_fwd_split_hdim64_bf16_sm80.cu");
    let decode128 = source.join("flash_fwd_split_hdim128_bf16_sm80.cu");
    let decode256 = source.join("flash_fwd_split_hdim256_bf16_sm80.cu");
    if !specialization64.is_file()
        || !specialization128.is_file()
        || !specialization256.is_file()
        || !decode64.is_file()
        || !decode128.is_file()
        || !decode256.is_file()
        || !cutlass.join("cutlass/cutlass.h").is_file()
    {
        return Err(
            format!("pinned FlashAttention sources not found in {}", flash.display()).into()
        );
    }
    let mut build = cc::Build::new();
    configure(&mut build);
    build
        .include("native/compat")
        .include(&source)
        .include(&cutlass)
        .flag("--expt-extended-lambda")
        .flag("--use_fast_math")
        .flag("-include")
        .flag("native/compat/flash_attn_prelude.h")
        .define("CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL", "1")
        .define("FLASHATTENTION_DISABLE_BACKWARD", None)
        .define("FLASHATTENTION_DISABLE_DROPOUT", None)
        .define("FLASHATTENTION_DISABLE_PYBIND", None)
        .file(specialization64)
        .file(specialization128)
        .file(specialization256)
        .file(decode64)
        .file(decode128)
        .file(decode256)
        .file("native/flash_attn2_bf16.cu")
        .compile("mircuda_flash_attn2");
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

fn flash_attn_dir() -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = env::var_os("MIRCUDA_FLASH_ATTN_DIR") {
        return Ok(PathBuf::from(path));
    }
    let home = env::var_os("HOME").ok_or("HOME is unavailable")?;
    Ok(PathBuf::from(home)
        .join(".cache/mircuda/flash-attention-2c839c33742309ec41e620bf837495ec9926c56e"))
}
