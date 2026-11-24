use mircuda::{
    CaptureMode, Context, DenseVectorKernel, DenseVectorPlan, DenseVectorSpec, DeviceBuffer,
    Stream, bf16,
};

use super::CYCLES;

/// One vector plan applied to every weight bank, captured as a single graph
/// so launch overhead does not hide router-sized kernels.
struct Banks<'a> {
    stream: &'a Stream,
    plan: DenseVectorPlan<bf16>,
    input: &'a DeviceBuffer<bf16>,
    weights: &'a [DeviceBuffer<bf16>],
    output: &'a mut DeviceBuffer<bf16>,
}

/// Average graph-replayed microseconds per GEMV for `kernel`, or `None` when
/// the shape does not admit it.
pub fn measure(
    context: &Context,
    stream: &Stream,
    spec: DenseVectorSpec,
    kernel: DenseVectorKernel,
    input: &DeviceBuffer<bf16>,
    banks: &[DeviceBuffer<bf16>],
    output: &mut DeviceBuffer<bf16>,
) -> mircuda::Result<Option<f32>> {
    let Ok(spec) = spec.with_kernel(kernel) else {
        return Ok(None);
    };
    let plan = DenseVectorPlan::new(context, stream, spec)?;
    let resources = Banks {
        stream,
        plan,
        input,
        weights: banks,
        output,
    };
    let mut graph = stream.capture_with(CaptureMode::ThreadLocal, resources, execute)?;
    graph.launch(stream)?;
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..CYCLES {
        graph.launch(stream)?;
    }
    completed.record(stream)?;
    completed.synchronize()?;
    let operations = f32::from(CYCLES) * f32::from(u16::try_from(banks.len())?);
    Ok(Some(started.elapsed_ms(&completed)? * 1_000.0 / operations))
}

fn execute(banks: &mut Banks<'_>) -> mircuda::Result<()> {
    for weight in banks.weights {
        banks.plan.execute(banks.stream, banks.input, weight, banks.output, 1.0, 0.0)?;
    }
    Ok(())
}
