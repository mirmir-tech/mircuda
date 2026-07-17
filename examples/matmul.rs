use std::{
    io::{self, Write},
    time::{Duration, Instant},
};

use mircuda::{
    Context, DenseMatmulElement, DenseMatmulPlan, DenseMatmulSpec, DeviceElement, Driver,
    MemoryPool, Stream, bf16, f16,
};

struct Report {
    name: &'static str,
    spec: DenseMatmulSpec,
    iterations: u16,
    plan_creation: Duration,
    workspace_bytes: usize,
    enqueue: Duration,
    device_ms: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let info = context.device_info()?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    pool.set_release_threshold(256 * 1_024 * 1_024)?;
    let reports = [
        run_case::<bf16>(
            "gemma4-dense-pair",
            &context,
            &stream,
            &pool,
            DenseMatmulSpec::new(16, 4_224, 2_816)?,
            100,
        )?,
        run_case::<bf16>(
            "gemma4-dense-down",
            &context,
            &stream,
            &pool,
            DenseMatmulSpec::new(16, 2_816, 2_112)?,
            100,
        )?,
        run_case::<bf16>(
            "gemma4-query",
            &context,
            &stream,
            &pool,
            DenseMatmulSpec::new(16, 4_096, 2_816)?,
            100,
        )?,
        run_case::<bf16>(
            "gemma4-global-query",
            &context,
            &stream,
            &pool,
            DenseMatmulSpec::new(16, 8_192, 2_816)?,
            100,
        )?,
        run_case::<bf16>(
            "decode-bf16",
            &context,
            &stream,
            &pool,
            DenseMatmulSpec::new(1, 4_096, 4_096)?,
            1_000,
        )?,
        run_case::<bf16>(
            "prefill-bf16",
            &context,
            &stream,
            &pool,
            DenseMatmulSpec::new(512, 4_096, 4_096)?,
            20,
        )?,
        run_case::<f16>(
            "decode-f16",
            &context,
            &stream,
            &pool,
            DenseMatmulSpec::new(1, 4_096, 4_096)?,
            1_000,
        )?,
    ];
    let mut output = io::stdout().lock();
    writeln!(
        output,
        "device: {} (compute {}.{})",
        info.name, info.compute_capability.0, info.compute_capability.1
    )?;
    for report in &reports {
        write_report(&mut output, report)?;
    }
    Ok(())
}

fn run_case<T: DeviceElement + DenseMatmulElement>(
    name: &'static str,
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    spec: DenseMatmulSpec,
    iterations: u16,
) -> mircuda::Result<Report> {
    let mut a = pool.allocate_zeroed::<T>(stream, elements(spec.m(), spec.k())?)?;
    let mut b = pool.allocate_zeroed::<T>(stream, elements(spec.n(), spec.k())?)?;
    let mut c = pool.allocate_zeroed::<T>(stream, elements(spec.m(), spec.n())?)?;
    let plan_started = Instant::now();
    let mut plan = DenseMatmulPlan::new(context, stream, spec)?;
    let plan_creation = plan_started.elapsed();
    for _ in 0..10 {
        plan.execute(stream, &a, &b, &mut c, 1.0, 0.0)?;
    }
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    let enqueue_started = Instant::now();
    for _ in 0..iterations {
        plan.execute(stream, &a, &b, &mut c, 1.0, 0.0)?;
    }
    let enqueue = enqueue_started.elapsed();
    completed.record(stream)?;
    completed.synchronize()?;
    let device_ms = started.elapsed_ms(&completed)?;
    std::hint::black_box((&mut a, &mut b));
    Ok(Report {
        name,
        spec,
        iterations,
        plan_creation,
        workspace_bytes: plan.workspace_bytes(),
        enqueue,
        device_ms,
    })
}

fn write_report(output: &mut impl Write, report: &Report) -> io::Result<()> {
    let iterations = f64::from(report.iterations);
    let operations = 2.0
        * dimension(report.spec.m())
        * dimension(report.spec.n())
        * dimension(report.spec.k())
        * iterations;
    let tflops = operations / (f64::from(report.device_ms) / 1_000.0) / 1_000_000_000_000.0;
    writeln!(
        output,
        "{}: m={} n={} k={}",
        report.name,
        report.spec.m(),
        report.spec.n(),
        report.spec.k()
    )?;
    writeln!(
        output,
        "  plan: {:.3} ms, workspace: {} bytes",
        report.plan_creation.as_secs_f64() * 1_000.0,
        report.workspace_bytes
    )?;
    writeln!(
        output,
        "  enqueue: {:.3} us/op",
        report.enqueue.as_secs_f64() * 1_000_000.0 / iterations
    )?;
    writeln!(
        output,
        "  device: {:.3} us/op, {:.3} TFLOP/s",
        f64::from(report.device_ms) * 1_000.0 / iterations,
        tflops
    )
}

fn elements(rows: usize, columns: usize) -> mircuda::Result<usize> {
    rows.checked_mul(columns).ok_or(mircuda::Error::InvalidMatmulShape)
}

fn dimension(value: usize) -> f64 {
    u32::try_from(value).map_or(f64::NAN, f64::from)
}
