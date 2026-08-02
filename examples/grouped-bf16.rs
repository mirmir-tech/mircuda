use std::io::{self, Write};

use mircuda::{
    Context, DeviceBuffer, Driver, MemoryPool, Stream, VariableGroupedBf16Plan,
    VariableGroupedBf16Spec, bf16,
};

const EXPERTS: usize = 32;
const TOKENS: usize = 8_192;
const SELECTED: usize = 4;
const HIDDEN: usize = 2_880;
const INTERMEDIATE: usize = 2_880;
const CYCLES: u16 = 4;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    n: usize,
    k: usize,
}

const CASES: &[Case] = &[
    Case {
        name: "gate-up",
        n: 2 * INTERMEDIATE,
        k: HIDDEN,
    },
    Case { name: "down", n: HIDDEN, k: INTERMEDIATE },
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let row_count = TOKENS * SELECTED / EXPERTS;
    let rows = vec![u32::try_from(row_count)?; EXPERTS];
    let offsets = (0..EXPERTS)
        .map(|expert| u32::try_from(expert * row_count))
        .collect::<Result<Vec<_>, _>>()?;
    let rows = upload(&context, &stream, &pool, &rows)?;
    let offsets = upload(&context, &stream, &pool, &offsets)?;
    let mut stdout = io::stdout().lock();
    for case in CASES {
        run(&context, &stream, &pool, &rows, &offsets, *case, &mut stdout)?;
    }
    Ok(())
}

fn run(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    rows: &DeviceBuffer<u32>,
    offsets: &DeviceBuffer<u32>,
    case: Case,
    stdout: &mut impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let routes = TOKENS * SELECTED;
    let mut plan = VariableGroupedBf16Plan::new(
        context,
        stream,
        VariableGroupedBf16Spec::new(EXPERTS, TOKENS, case.n, case.k, routes)?,
    )?;
    let input = pool.allocate::<bf16>(stream, routes * case.k)?;
    let weights = pool.allocate::<bf16>(stream, EXPERTS * case.n * case.k)?;
    let mut output = pool.allocate::<bf16>(stream, routes * case.n)?;
    plan.execute(stream, &input, &weights, rows, offsets, &mut output, 0.0)?;
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..usize::from(CYCLES) {
        plan.execute(stream, &input, &weights, rows, offsets, &mut output, 0.0)?;
    }
    completed.record(stream)?;
    completed.synchronize()?;
    let elapsed = started.elapsed_ms(&completed)? / f32::from(CYCLES);
    writeln!(stdout, "{}: {elapsed:.3} ms", case.name)?;
    Ok(())
}

fn upload<T: mircuda::DeviceElement + Copy>(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    values: &[T],
) -> mircuda::Result<DeviceBuffer<T>> {
    let mut host = context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = pool.allocate(stream, values.len())?;
    stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}
