use std::io::{self, Write};

use mircuda::{
    Context, DenseMatmulPlan, DenseMatmulSpec, DenseVectorPlan, DenseVectorSpec, DeviceBuffer,
    Driver, MemoryPool, Stream, bf16,
};

const CYCLES: u16 = 4;
const TARGET_BANK_BYTES: usize = 768 * 1_024 * 1_024;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    n: usize,
    k: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let cases = [
        Case { name: "router", n: 128, k: 2_816 },
        Case { name: "dense-down", n: 2_816, k: 2_112 },
        Case {
            name: "attention-output",
            n: 2_816,
            k: 4_096,
        },
        Case {
            name: "dense-gate-up",
            n: 4_224,
            k: 2_816,
        },
        Case { name: "local-qkv", n: 8_192, k: 2_816 },
        Case { name: "global-qkv", n: 10_240, k: 2_816 },
        Case {
            name: "output-head",
            n: 262_144,
            k: 2_816,
        },
    ];
    let mut stdout = io::stdout().lock();
    for case in cases {
        run(&context, &stream, &pool, case, &mut stdout)?;
    }
    Ok(())
}

fn run(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    case: Case,
    stdout: &mut impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let elements = case.n.checked_mul(case.k).ok_or(mircuda::Error::InvalidMatmulShape)?;
    let bytes = elements
        .checked_mul(size_of::<bf16>())
        .ok_or(mircuda::Error::InvalidMatmulShape)?;
    let bank_count = (TARGET_BANK_BYTES / bytes).clamp(1, 64);
    let template = values(elements)?;
    let mut banks = Vec::with_capacity(bank_count);
    for _ in 0..bank_count {
        banks.push(upload(context, stream, pool, &template)?);
    }
    let input = upload(context, stream, pool, &values(case.k)?)?;
    let mut output = pool.allocate::<bf16>(stream, case.n)?;
    let mut matrix =
        DenseMatmulPlan::new(context, stream, DenseMatmulSpec::new(1, case.n, case.k)?)?;
    let mut vector = DenseVectorPlan::new(context, stream, DenseVectorSpec::new(case.n, case.k)?)?;
    let matrix_us = measure(context, stream, &banks, |weight| {
        matrix.execute(stream, &input, weight, &mut output, 1.0, 0.0)
    })?;
    let vector_us = measure(context, stream, &banks, |weight| {
        vector.execute(stream, &input, weight, &mut output, 1.0, 0.0)
    })?;
    writeln!(
        stdout,
        "{}: n={} k={} banks={} matrix={matrix_us:.3} us vector={vector_us:.3} us speedup={:.3}x",
        case.name,
        case.n,
        case.k,
        banks.len(),
        matrix_us / vector_us
    )?;
    Ok(())
}

fn measure(
    context: &Context,
    stream: &Stream,
    banks: &[DeviceBuffer<bf16>],
    mut execute: impl FnMut(&DeviceBuffer<bf16>) -> mircuda::Result<()>,
) -> mircuda::Result<f32> {
    for weight in banks {
        execute(weight)?;
    }
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..CYCLES {
        for weight in banks {
            execute(weight)?;
        }
    }
    completed.record(stream)?;
    completed.synchronize()?;
    let operations = f32::from(CYCLES) * f32::from(u16::try_from(banks.len())?);
    Ok(started.elapsed_ms(&completed)? * 1_000.0 / operations)
}

fn values(count: usize) -> mircuda::Result<Vec<bf16>> {
    (0..count)
        .map(|index| {
            let value = f32::from(u8::try_from(index % 251)?) / 256.0 - 0.5;
            Ok(bf16::from_f32(value))
        })
        .collect()
}

fn upload(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    values: &[bf16],
) -> mircuda::Result<DeviceBuffer<bf16>> {
    let mut host = context.allocate_pinned::<bf16>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = pool.allocate::<bf16>(stream, values.len())?;
    stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}
