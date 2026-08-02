use std::io::{self, Write};

use mircuda::{
    Context, CublasBf16Plan, CublasBf16Spec, CublasLtBf16Plan, CublasLtBf16Spec, DenseMatmulPlan,
    DenseMatmulSpec, Driver, MemoryPool, Stream, bf16,
};

const ITERATIONS: u16 = 10;
const ROWS: &[usize] =
    &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 2_560, 4_096, 5_120, 8_192];

#[derive(Clone, Copy)]
struct Projection {
    name: &'static str,
    n: usize,
    k: usize,
    max_m: usize,
}

const PROJECTIONS: &[Projection] = &[
    Projection {
        name: "qkv",
        n: 6_144,
        k: 2_560,
        max_m: 8_192,
    },
    Projection {
        name: "attention-output",
        n: 2_560,
        k: 4_096,
        max_m: 8_192,
    },
    Projection {
        name: "gate-up",
        n: 19_456,
        k: 2_560,
        max_m: 8_192,
    },
    Projection {
        name: "down",
        n: 2_560,
        k: 9_728,
        max_m: 8_192,
    },
    Projection {
        name: "output-head",
        n: 151_936,
        k: 2_560,
        max_m: 16,
    },
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let mut output = io::stdout().lock();
    let selected = std::env::args().nth(1);
    for projection in PROJECTIONS {
        if selected.as_deref().is_some_and(|name| name != projection.name) {
            continue;
        }
        for &m in ROWS.iter().filter(|m| **m <= projection.max_m) {
            profile(&context, &stream, &pool, *projection, m, &mut output)?;
        }
    }
    Ok(())
}

fn profile(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    projection: Projection,
    m: usize,
    output: &mut impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut input = pool.allocate_zeroed::<bf16>(stream, elements(m, projection.k)?)?;
    let mut weight = pool.allocate_zeroed::<bf16>(stream, elements(projection.n, projection.k)?)?;
    let mut result = pool.allocate_zeroed::<bf16>(stream, elements(m, projection.n)?)?;
    let mut cutlass = DenseMatmulPlan::new(
        context,
        stream,
        DenseMatmulSpec::new(m, projection.n, projection.k)?,
    )?;
    let mut vendor = CublasLtBf16Plan::new(
        context,
        stream,
        CublasLtBf16Spec::new(m, projection.n, projection.k)?,
    )?;
    let mut classic =
        CublasBf16Plan::new(context, stream, CublasBf16Spec::new(m, projection.n, projection.k)?)?;
    let cutlass_us = measure(context, stream, || {
        cutlass.execute(stream, &input, &weight, &mut result, 1.0, 0.0)
    })?;
    let vendor_us = measure(context, stream, || {
        vendor.execute(stream, &input, &weight, &mut result, 1.0, 0.0)
    })?;
    let classic_us = measure(context, stream, || {
        classic.execute(stream, &input, &weight, &mut result, 1.0, 0.0)
    })?;
    writeln!(
        output,
        "{}: m={m} n={} k={} cutlass={cutlass_us:.3} us cublaslt={vendor_us:.3} us \
         cublas={classic_us:.3} us vendor_speedup={:.3}x classic_speedup={:.3}x \
         vendor_workspace={} bytes",
        projection.name,
        projection.n,
        projection.k,
        cutlass_us / vendor_us,
        cutlass_us / classic_us,
        vendor.workspace_bytes(),
    )?;
    std::hint::black_box((&mut input, &mut weight));
    Ok(())
}

fn measure(
    context: &Context,
    stream: &Stream,
    mut execute: impl FnMut() -> mircuda::Result<()>,
) -> mircuda::Result<f32> {
    for _ in 0..3 {
        execute()?;
    }
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..ITERATIONS {
        execute()?;
    }
    completed.record(stream)?;
    completed.synchronize()?;
    Ok(started.elapsed_ms(&completed)? * 1_000.0 / f32::from(ITERATIONS))
}

fn elements(rows: usize, columns: usize) -> mircuda::Result<usize> {
    rows.checked_mul(columns).ok_or(mircuda::Error::InvalidMatmulShape)
}
