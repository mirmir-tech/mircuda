//! Teardown on other threads, or inside another capture, stays out of every
//! capture in progress.

use mircuda::{DeviceBuffer, LaunchConfig, Stream, TypedKernel};

use super::{Fixture, Increment, new_context};

#[test]
fn capture_survives_a_new_context_stream_on_another_thread() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    fixture.capture_during(|_| new_context()?.create_stream().map(drop))
}

#[test]
fn capture_survives_releasing_a_cross_stream_buffer_on_another_thread() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    fixture.capture_during(|_| {
        let allocation = fixture.context.create_stream()?;
        let foreign = fixture.context.create_stream()?;
        let mut buffer = fixture.pool.allocate_zeroed::<u32>(&allocation, 1)?;
        fixture
            .kernel
            .launch(&foreign, LaunchConfig::for_elements(1, 1)?, (&mut buffer, 1))?;
        drop(buffer);
        allocation.synchronize()
    })
}

/// A buffer used on the stream another thread now captures cannot order its
/// free after that stream's work without recording into the capture, so the
/// free waits for the capture to end.
#[test]
fn capture_survives_releasing_a_buffer_used_on_the_captured_stream() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    let captured = fixture.context.create_stream()?;
    let allocation = fixture.context.create_stream()?;
    let mut buffer = fixture.pool.allocate_zeroed::<u32>(&allocation, 1)?;
    allocation.synchronize()?;
    fixture
        .kernel
        .launch(&captured, LaunchConfig::for_elements(1, 1)?, (&mut buffer, 1))?;
    fixture.capture_on(&captured, move |_| {
        drop(buffer);
        assert_eq!(mircuda::capture_activity().deferred_teardown, 1);
        Ok(())
    })?;
    // The serial guard stays held until the gate is observed idle.
    assert_eq!(mircuda::capture_activity(), mircuda::CaptureActivity::default());
    drop(fixture);
    allocation.synchronize()
}

#[cfg(feature = "cublaslt")]
#[test]
fn capture_survives_a_cublaslt_plan_on_another_thread() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    fixture.capture_during(|reached| {
        let stream = fixture.context.create_stream()?;
        let spec = mircuda::CublasLtBf16Spec::new(64, 64, 64)?;
        let plan = mircuda::CublasLtBf16Plan::new(&fixture.context, &stream, spec)?;
        // The last plan of the stream destroys its handle, synchronizing the
        // device; the capture must not see it.
        reached();
        drop(plan);
        Ok(())
    })
}

#[cfg(feature = "cublaslt")]
type PlanCapture<'a> = (
    &'a TypedKernel<Increment>,
    &'a Stream,
    &'a mut DeviceBuffer<u32>,
    Option<mircuda::CublasLtBf16Plan>,
);

#[cfg(feature = "cublaslt")]
fn drop_plan_while_capturing(capture: &mut PlanCapture<'_>) -> mircuda::Result<()> {
    let (kernel, stream, value, plan) = capture;
    kernel.launch_captured(stream, LaunchConfig::for_elements(1, 1)?, (&mut **value, 1))?;
    drop(plan.take());
    Ok(())
}

/// A plan dropped inside one thread's capture waits for every capture, not
/// only that thread's: destroying its handle would synchronize the device
/// under the other thread's capture as well.
#[cfg(feature = "cublaslt")]
#[test]
fn concurrent_captures_survive_a_cublaslt_plan_dropped_inside_one() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    fixture.capture_during(|_| {
        let spec = mircuda::CublasLtBf16Spec::new(64, 64, 64)?;
        let plan_stream = fixture.context.create_stream()?;
        let plan = mircuda::CublasLtBf16Plan::new(&fixture.context, &plan_stream, spec)?;
        let stream = fixture.context.create_stream()?;
        let mut value = fixture.pool.allocate_zeroed::<u32>(&stream, 1)?;
        stream.synchronize()?;
        let mut graph = stream.capture(
            (&fixture.kernel, &stream, &mut value, Some(plan)),
            drop_plan_while_capturing,
        )?;
        // The other thread still captures, so the teardown still waits.
        assert_eq!(mircuda::capture_activity().deferred_teardown, 1);
        graph.launch(&stream)?;
        drop(graph);
        stream.synchronize()
    })?;
    assert_eq!(mircuda::capture_activity(), mircuda::CaptureActivity::default());
    drop(fixture);
    Ok(())
}
