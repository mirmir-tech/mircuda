use std::{
    io::{self, Write},
    time::{Duration, Instant},
};

use mircuda::{
    CompileCacheStats, CompileOptions, Compiler, DeviceBuffer, DeviceInfo, Driver, LaunchConfig,
    MemoryPoolStats, cuda_export, cuda_kernel_file,
};

cuda_export!(
    ProbeKernel = "mircuda_probe"(
        output: &mut DeviceBuffer<f32>,
        value: f32,
        length: u32,
    )
);

const ELEMENTS: usize = 1_048_576;
const ELEMENTS_U32: u32 = 1_048_576;
const ITERATIONS: u16 = 1_000;

struct Report {
    device: DeviceInfo,
    cold_compile: Duration,
    cached_compile: Duration,
    cache: CompileCacheStats,
    cold_allocation: Duration,
    warm_allocation: Duration,
    pinned_allocation: Duration,
    event_creation: Duration,
    launch_enqueue: Duration,
    kernel_ms: f32,
    copy_ms: f32,
    memory: MemoryPoolStats,
    valid: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = profile()?;
    write_report(&report)?;
    if !report.valid {
        return Err("CUDA probe returned invalid data".into());
    }
    Ok(())
}

fn profile() -> mircuda::Result<Report> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let info = context.device_info()?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    pool.set_release_threshold(512 * 1_024 * 1_024)?;

    let compiler = Compiler::new(context.clone())?;
    let source = cuda_kernel_file!("../kernels/probe.cu");
    let cold_started = Instant::now();
    let module = compiler.compile(source, &CompileOptions::default())?;
    let cold_compile = cold_started.elapsed();
    let cached_started = Instant::now();
    let _cached_module = compiler.compile(source, &CompileOptions::default())?;
    let cached_compile = cached_started.elapsed();
    let kernel = module.kernel::<ProbeKernel>()?;

    let cold_allocation_started = Instant::now();
    let cold_allocation = {
        let _cold_device_values = pool.allocate::<f32>(&stream, ELEMENTS)?;
        cold_allocation_started.elapsed()
    };
    stream.synchronize()?;
    let warm_allocation_started = Instant::now();
    let mut device_values = pool.allocate::<f32>(&stream, ELEMENTS)?;
    let warm_allocation = warm_allocation_started.elapsed();
    let pinned_allocation_started = Instant::now();
    let mut host_values = context.allocate_pinned::<f32>(ELEMENTS)?;
    let pinned_allocation = pinned_allocation_started.elapsed();
    let config = LaunchConfig::for_elements(ELEMENTS, 256)?;

    let event_creation_started = Instant::now();
    let kernel_started = context.create_event(true)?;
    let kernel_completed = context.create_event(true)?;
    let copy_started = context.create_event(true)?;
    let copy_completed = context.create_event(true)?;
    let event_creation = event_creation_started.elapsed();
    kernel_started.record(&stream)?;
    let launch_enqueue_started = Instant::now();
    for iteration in 0..ITERATIONS {
        kernel.launch(&stream, config, (&mut device_values, f32::from(iteration), ELEMENTS_U32))?;
    }
    let launch_enqueue = launch_enqueue_started.elapsed();
    kernel_completed.record(&stream)?;

    copy_started.record(&stream)?;
    stream.copy_to_host(&device_values, &mut host_values)?;
    copy_completed.record(&stream)?;
    let values = host_values.to_vec()?;
    let expected = f32::from(ITERATIONS - 1);
    Ok(Report {
        device: info,
        cold_compile,
        cached_compile,
        cache: compiler.cache_stats(),
        cold_allocation,
        warm_allocation,
        pinned_allocation,
        event_creation,
        launch_enqueue,
        kernel_ms: kernel_started.elapsed_ms(&kernel_completed)?,
        copy_ms: copy_started.elapsed_ms(&copy_completed)?,
        memory: pool.stats()?,
        valid: values.iter().all(|value| value.to_bits() == expected.to_bits()),
    })
}

fn write_report(report: &Report) -> io::Result<()> {
    let mut output = io::stdout().lock();
    writeln!(
        output,
        "device: {} (compute {}.{})",
        report.device.name, report.device.compute_capability.0, report.device.compute_capability.1
    )?;
    writeln!(output, "compile cold: {:.3} ms", report.cold_compile.as_secs_f64() * 1_000.0)?;
    writeln!(
        output,
        "compile cached: {:.3} us",
        report.cached_compile.as_secs_f64() * 1_000_000.0
    )?;
    writeln!(
        output,
        "cache: {} hit / {} miss / {} entry",
        report.cache.hits, report.cache.misses, report.cache.entries
    )?;
    writeln!(
        output,
        "device allocation cold: {:.3} us",
        report.cold_allocation.as_secs_f64() * 1_000_000.0
    )?;
    writeln!(
        output,
        "device allocation warm: {:.3} us",
        report.warm_allocation.as_secs_f64() * 1_000_000.0
    )?;
    writeln!(
        output,
        "pinned allocation: {:.3} us",
        report.pinned_allocation.as_secs_f64() * 1_000_000.0
    )?;
    writeln!(
        output,
        "four event creations: {:.3} us",
        report.event_creation.as_secs_f64() * 1_000_000.0
    )?;
    writeln!(
        output,
        "launch enqueue: {:.3} us/launch",
        report.launch_enqueue.as_secs_f64() * 1_000_000.0 / f64::from(ITERATIONS)
    )?;
    writeln!(
        output,
        "kernel: {:.3} ms total, {:.3} us/launch",
        report.kernel_ms,
        f64::from(report.kernel_ms) * 1_000.0 / f64::from(ITERATIONS)
    )?;
    writeln!(output, "device-to-host: {:.3} ms", report.copy_ms)?;
    writeln!(
        output,
        "pool: {} reserved / {} used bytes",
        report.memory.reserved, report.memory.used
    )?;
    writeln!(
        output,
        "result: {}",
        if report.valid {
            "valid"
        } else {
            "invalid"
        }
    )
}
