//! A buffer freed while its allocation stream captures joins that capture
//! only when the capture allocated it: a graph allocation is freed in its
//! graph, while a buffer allocated before the capture waits for it to end.

use mircuda::{DeviceBuffer, LaunchConfig, MemoryPool, Stream, TypedKernel};

use super::{Fixture, Increment};

#[test]
fn releasing_a_buffer_of_the_captured_stream_on_another_thread_waits() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    let captured = fixture.context.create_stream()?;
    let buffer = fixture.pool.allocate_zeroed::<u32>(&captured, 1)?;
    fixture.capture_on(&captured, move |_| {
        drop(buffer);
        assert_eq!(mircuda::capture_activity().deferred_teardown, 1);
        Ok(())
    })?;
    assert_eq!(mircuda::capture_activity(), mircuda::CaptureActivity::default());
    drop(fixture);
    captured.synchronize()
}

type Release<'a> = (
    &'a TypedKernel<Increment>,
    &'a Stream,
    &'a mut DeviceBuffer<u32>,
    Option<DeviceBuffer<u32>>,
);

fn release_inside_the_capture(capture: &mut Release<'_>) -> mircuda::Result<()> {
    let (kernel, stream, value, earlier) = capture;
    kernel.launch_captured(stream, LaunchConfig::for_elements(1, 1)?, (&mut **value, 1))?;
    drop(earlier.take());
    assert_eq!(mircuda::capture_activity().deferred_teardown, 1);
    Ok(())
}

#[test]
fn releasing_an_earlier_buffer_inside_the_capture_waits() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    let stream = fixture.context.create_stream()?;
    let mut value = fixture.pool.allocate_zeroed::<u32>(&stream, 1)?;
    let earlier = fixture.pool.allocate_zeroed::<u32>(&stream, 1)?;
    stream.synchronize()?;
    let mut graph = stream.capture(
        (&fixture.kernel, &stream, &mut value, Some(earlier)),
        release_inside_the_capture,
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

type Temporary<'a> = (
    &'a TypedKernel<Increment>,
    &'a Stream,
    &'a MemoryPool,
    &'a mut DeviceBuffer<u32>,
);

fn use_a_temporary(capture: &mut Temporary<'_>) -> mircuda::Result<()> {
    let (kernel, stream, pool, value) = capture;
    let mut temporary = pool.allocate_zeroed::<u32>(stream, 1)?;
    let launch = LaunchConfig::for_elements(1, 1)?;
    kernel.launch_captured(stream, launch, (&mut temporary, 1))?;
    kernel.launch_captured(stream, launch, (&mut **value, 1))?;
    drop(temporary);
    // The capture allocated the temporary, so its free is a node of the graph.
    assert_eq!(mircuda::capture_activity().deferred_teardown, 0);
    Ok(())
}

#[test]
fn a_graph_allocation_is_freed_inside_its_capture() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    let stream = fixture.context.create_stream()?;
    let mut value = fixture.pool.allocate_zeroed::<u32>(&stream, 1)?;
    stream.synchronize()?;
    let mut graph =
        stream.capture((&fixture.kernel, &stream, &fixture.pool, &mut value), use_a_temporary)?;
    graph.launch(&stream)?;
    graph.launch(&stream)?;
    drop(graph);
    let mut host = fixture.context.allocate_pinned::<u32>(1)?;
    stream.copy_to_host(&value, &mut host)?;
    assert_eq!(host.to_vec()?, [2]);
    assert_eq!(mircuda::capture_activity(), mircuda::CaptureActivity::default());
    drop(fixture);
    Ok(())
}

/// A free waiting for its stream's capture runs when that capture ends, even
/// while an overlapping capture on another thread keeps going.
#[test]
fn a_waiting_free_runs_when_its_own_capture_ends() -> mircuda::Result<()> {
    let fixture = Fixture::new()?;
    fixture.capture_during(|_| {
        let stream = fixture.context.create_stream()?;
        let mut value = fixture.pool.allocate_zeroed::<u32>(&stream, 1)?;
        let earlier = fixture.pool.allocate_zeroed::<u32>(&stream, 1)?;
        stream.synchronize()?;
        let graph = stream.capture(
            (&fixture.kernel, &stream, &mut value, Some(earlier)),
            release_inside_the_capture,
        )?;
        assert_eq!(mircuda::capture_activity().capturing_threads, 1);
        assert_eq!(mircuda::capture_activity().deferred_teardown, 0);
        drop(graph);
        stream.synchronize()
    })?;
    drop(fixture);
    Ok(())
}
