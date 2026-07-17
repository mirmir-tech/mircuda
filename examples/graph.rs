use std::{
    io::{self, Write},
    time::{Duration, Instant},
};

use mircuda::{DenseMatmulPlan, DenseMatmulSpec, DeviceBuffer, Driver, Graph, Stream, bf16};

const M: usize = 1;
const N: usize = 4_096;
const K: usize = 4_096;
const NODES: u16 = 32;
const REPLAYS: u16 = 200;

struct Report {
    name: &'static str,
    enqueue: Duration,
    device_ms: f32,
}

struct Sequence<'a> {
    stream: &'a Stream,
    plan: &'a mut DenseMatmulPlan<bf16>,
    a: &'a DeviceBuffer<bf16>,
    b: &'a DeviceBuffer<bf16>,
    c: &'a mut DeviceBuffer<bf16>,
}

fn capture_sequence(sequence: &mut Sequence<'_>) -> mircuda::Result<()> {
    for _ in 0..NODES {
        sequence
            .plan
            .execute(sequence.stream, sequence.a, sequence.b, sequence.c, 1.0, 0.0)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let info = context.device_info()?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let a = pool.allocate_zeroed::<bf16>(&stream, M * K)?;
    let b = pool.allocate_zeroed::<bf16>(&stream, N * K)?;
    let mut c = pool.allocate_zeroed::<bf16>(&stream, M * N)?;
    let mut plan = DenseMatmulPlan::new(&context, &stream, DenseMatmulSpec::new(M, N, K)?)?;
    let direct = measure_direct(&context, &stream, &mut plan, &a, &b, &mut c)?;
    let sequence = Sequence {
        stream: &stream,
        plan: &mut plan,
        a: &a,
        b: &b,
        c: &mut c,
    };
    let mut graph = stream.capture(sequence, capture_sequence)?;
    let replay = measure_graph(&context, &stream, &mut graph)?;
    let mut output = io::stdout().lock();
    writeln!(output, "device: {}", info.name)?;
    writeln!(output, "sequence: {NODES} BF16 GEMMs, m={M} n={N} k={K}")?;
    write_report(&mut output, &direct)?;
    write_report(&mut output, &replay)?;
    Ok(())
}

fn measure_direct(
    context: &mircuda::Context,
    stream: &Stream,
    plan: &mut DenseMatmulPlan<bf16>,
    a: &DeviceBuffer<bf16>,
    b: &DeviceBuffer<bf16>,
    c: &mut DeviceBuffer<bf16>,
) -> mircuda::Result<Report> {
    for _ in 0..NODES {
        plan.execute(stream, a, b, c, 1.0, 0.0)?;
    }
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    let host_started = Instant::now();
    for _ in 0..REPLAYS {
        for _ in 0..NODES {
            plan.execute(stream, a, b, c, 1.0, 0.0)?;
        }
    }
    let enqueue = host_started.elapsed();
    completed.record(stream)?;
    completed.synchronize()?;
    Ok(Report {
        name: "direct",
        enqueue,
        device_ms: started.elapsed_ms(&completed)?,
    })
}

fn measure_graph<Resources>(
    context: &mircuda::Context,
    stream: &Stream,
    graph: &mut Graph<Resources>,
) -> mircuda::Result<Report> {
    for _ in 0..10 {
        graph.launch(stream)?;
    }
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    let host_started = Instant::now();
    for _ in 0..REPLAYS {
        graph.launch(stream)?;
    }
    let enqueue = host_started.elapsed();
    completed.record(stream)?;
    completed.synchronize()?;
    Ok(Report {
        name: "graph",
        enqueue,
        device_ms: started.elapsed_ms(&completed)?,
    })
}

fn write_report(output: &mut impl Write, report: &Report) -> io::Result<()> {
    let replays = f64::from(REPLAYS);
    let nodes = f64::from(NODES);
    writeln!(output, "{}:", report.name)?;
    writeln!(
        output,
        "  enqueue: {:.3} us/replay, {:.3} ns/GEMM",
        report.enqueue.as_secs_f64() * 1_000_000.0 / replays,
        report.enqueue.as_secs_f64() * 1_000_000_000.0 / replays / nodes
    )?;
    writeln!(
        output,
        "  device: {:.3} us/replay, {:.3} us/GEMM",
        f64::from(report.device_ms) * 1_000.0 / replays,
        f64::from(report.device_ms) * 1_000.0 / replays / nodes
    )
}
