//! A CUTLASS plan frees its workspace on its stream when it is destroyed. A
//! plan built before a capture of that stream waits for the capture to end
//! instead of freeing into it, whether it is dropped inside the capture or
//! on another thread.

use std::num::NonZeroU16;

use mircuda::{
    DenseVectorKernel, DenseVectorPlan, DenseVectorSpec, DeviceBuffer, LaunchConfig, Stream,
    TypedKernel, bf16,
};

use super::{Fixture, Increment};

/// A plan whose workspace the driver allocated on `stream`.
fn split_k_plan(fixture: &Fixture, stream: &Stream) -> mircuda::Result<DenseVectorPlan<bf16>> {
    let splits = NonZeroU16::new(4).ok_or(mircuda::Error::InvalidLaunch)?;
    let spec = DenseVectorSpec::new(32, 256)?.with_kernel(DenseVectorKernel::SplitK { splits })?;
    let plan = DenseVectorPlan::new(&fixture.context, stream, spec)?;
    assert!(plan.workspace_bytes() > 0, "split-K owns no workspace to free");
    stream.synchronize()?;
    Ok(plan)
}

type PlanCapture<'a> = (
    &'a TypedKernel<Increment>,
    &'a Stream,
    &'a mut DeviceBuffer<u32>,
    Option<DenseVectorPlan<bf16>>,
);

fn drop_plan_inside_the_capture(capture: &mut PlanCapture<'_>) -> mircuda::Result<()> {
    let (kernel, stream, value, plan) = capture;
    kernel.launch_captured(stream, LaunchConfig::for_elements(1, 1)?, (&mut **value, 1))?;
    drop(plan.take());
    assert_eq!(mircuda::capture_activity().deferred_teardown, 1);
    Ok(())
}

#[test]
fn a_plan_dropped_inside_its_streams_capture_waits() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    let stream = fixture.context.create_stream()?;
    let mut value = fixture.pool.allocate_zeroed::<u32>(&stream, 1)?;
    let plan = split_k_plan(&fixture, &stream)?;
    let mut graph = stream.capture(
        (&fixture.kernel, &stream, &mut value, Some(plan)),
        drop_plan_inside_the_capture,
    )?;
    assert_eq!(mircuda::capture_activity(), mircuda::CaptureActivity::default());
    graph.launch(&stream)?;
    graph.launch(&stream)?;
    drop(graph);
    let mut host = fixture.context.allocate_pinned::<u32>(1)?;
    stream.copy_to_host(&value, &mut host)?;
    assert_eq!(host.to_vec()?, [2]);
    drop(fixture);
    Ok(())
}

#[test]
fn a_plan_dropped_on_another_thread_waits_for_its_streams_capture() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    let captured = fixture.context.create_stream()?;
    let plan = split_k_plan(&fixture, &captured)?;
    fixture.capture_on(&captured, move |_| {
        drop(plan);
        assert_eq!(mircuda::capture_activity().deferred_teardown, 1);
        Ok(())
    })?;
    // The serial guard stays held until the gate is observed idle.
    assert_eq!(mircuda::capture_activity(), mircuda::CaptureActivity::default());
    drop(fixture);
    captured.synchronize()
}

/// A plan on a stream that captures nothing is destroyed at once, during
/// another stream's capture.
#[test]
fn a_plan_on_an_idle_stream_is_destroyed_during_a_capture() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    fixture.capture_during(|_| {
        let stream = fixture.context.create_stream()?;
        drop(split_k_plan(&fixture, &stream)?);
        assert_eq!(mircuda::capture_activity().deferred_teardown, 0);
        stream.synchronize()
    })
}
