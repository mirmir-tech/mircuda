#[cfg(target_os = "linux")]
use mircuda::{CompileOptions, Compiler, CompilerConfig, Driver, LaunchConfig, cuda_kernel_file};
use mircuda::{DeviceBuffer, cuda_export, cuda_ptx_file};

cuda_export!(
    ProbeKernel = "mircuda_probe"(
        output: &mut DeviceBuffer<f32>,
        value: f32,
        length: u32,
    )
);

cuda_export!(PrecompiledProbe = "mircuda_precompiled_probe"());

#[test]
#[cfg(target_os = "linux")]
fn compiles_caches_and_launches_on_an_explicit_stream() -> mircuda::Result<()> {
    const LENGTH: usize = 4_096;
    const LENGTH_U32: u32 = 4_096;
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    pool.set_release_threshold(64 * 1_024 * 1_024)?;
    let compiler = Compiler::new(context.clone())?;
    let source = cuda_kernel_file!("../kernels/probe.cu");
    let module = compiler.compile(source, &CompileOptions::default())?;
    let _cached = compiler.compile(source, &CompileOptions::default())?;
    assert_eq!(compiler.cache_stats().hits, 1);
    let kernel = module.kernel::<ProbeKernel>()?;
    let mut device_values = pool.allocate_zeroed::<f32>(&stream, LENGTH)?;
    let mut copied_values = pool.allocate_zeroed::<f32>(&stream, LENGTH)?;
    let mut host_values = context.allocate_pinned::<f32>(LENGTH)?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(&stream)?;
    kernel.launch(
        &stream,
        LaunchConfig::for_elements(LENGTH, 256)?,
        (&mut device_values, 7.0, LENGTH_U32),
    )?;
    stream.copy_device_range(&device_values, 1_024..3_072, &mut copied_values, 512)?;
    completed.record(&stream)?;
    stream.copy_to_host(&device_values, &mut host_values)?;
    assert!(host_values.to_vec()?.iter().all(|value| value.to_bits() == 7.0_f32.to_bits()));
    let mut copied_host = context.allocate_pinned::<f32>(LENGTH)?;
    stream.copy_to_host(&copied_values, &mut copied_host)?;
    let copied_host = copied_host.to_vec()?;
    assert!(copied_host[..512].iter().all(|value| value.to_bits() == 0.0_f32.to_bits()));
    assert!(copied_host[512..2_560].iter().all(|value| value.to_bits() == 7.0_f32.to_bits()));
    assert!(copied_host[2_560..].iter().all(|value| value.to_bits() == 0.0_f32.to_bits()));
    assert!(started.elapsed_ms(&completed)? >= 0.0);
    assert!(pool.stats()?.reserved >= device_values.bytes() as u64);
    drop(device_values);
    stream.synchronize()?;
    pool.trim_to(0)?;
    Ok(())
}

#[test]
#[cfg(target_os = "linux")]
fn loads_caches_and_launches_precompiled_ptx() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    if context.device_info()?.compute_capability != (12, 1) {
        return Ok(());
    }
    let stream = context.create_stream()?;
    let compiler = Compiler::new(context)?;
    let source = cuda_ptx_file!((12, 1), "../kernels/probe.ptx");
    let module = compiler.load_ptx(source)?;
    let cached = compiler.load_ptx(source)?;
    assert_eq!(compiler.cache_stats().hits, 1);
    module.kernel::<PrecompiledProbe>()?.launch(
        &stream,
        LaunchConfig {
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_memory_bytes: 0,
        },
        (),
    )?;
    drop(cached);
    stream.synchronize()
}

#[test]
#[cfg(target_os = "linux")]
fn loads_persisted_ptx_without_reinvoking_nvrtc() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let directory = std::env::temp_dir().join(format!("mircuda-ptx-{}", std::process::id()));
    let config = CompilerConfig {
        cache_directory: Some(directory.clone()),
        ..CompilerConfig::default()
    };
    let source = cuda_kernel_file!("../kernels/probe.cu");

    let first = Compiler::with_config(context.clone(), config.clone())?;
    let first_module = first.compile(source, &CompileOptions::default())?;
    assert_eq!(first.cache_stats().misses, 1);
    drop(first_module);
    drop(first);

    let second = Compiler::with_config(context, config)?;
    let second_module = second.compile(source, &CompileOptions::default())?;
    let stats = second.cache_stats();
    assert_eq!(stats.persistent_hits, 1);
    assert_eq!(stats.misses, 0);
    drop(second_module);
    drop(second);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}
