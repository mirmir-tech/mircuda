#![cfg(target_os = "linux")]

//! Work on other threads of the same device must not invalidate a
//! thread-local capture: neither another runtime starting up, nor the release
//! of a buffer used on several streams, nor the release of a cuBLAS handle.

use std::{
    sync::{
        Barrier,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use mircuda::{
    CompileOptions, Compiler, Context, DeviceBuffer, Driver, LaunchConfig, MemoryPool, Stream,
    TypedKernel, cuda_export, cuda_kernel,
};

cuda_export!(Increment = "increment"(value: &mut DeviceBuffer<u32>, delta: u32));

struct Sequence<'a> {
    kernel: &'a TypedKernel<Increment>,
    stream: &'a Stream,
    value: &'a mut DeviceBuffer<u32>,
    started: &'a Barrier,
    finished: &'a Barrier,
}

fn capture_while_another_thread_works(sequence: &mut Sequence<'_>) -> mircuda::Result<()> {
    sequence.kernel.launch_captured(
        sequence.stream,
        LaunchConfig::for_elements(1, 1)?,
        (&mut *sequence.value, 1),
    )?;
    sequence.started.wait();
    sequence.finished.wait();
    // Teardown the other thread starts now either completes or waits for this
    // capture to end; give it the time to reach the driver.
    std::thread::sleep(Duration::from_millis(50));
    Ok(())
}

struct Fixture {
    context: Context,
    pool: MemoryPool,
    kernel: TypedKernel<Increment>,
}

fn new_context() -> mircuda::Result<Context> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    driver.create_context(device)
}

impl Fixture {
    fn new() -> mircuda::Result<Self> {
        let context = new_context()?;
        let pool = context.default_memory_pool()?;
        let compiler = Compiler::new(context.clone())?;
        let source = cuda_kernel!(
            r#"extern "C" __global__ void increment(unsigned int* value, unsigned int delta) {
                if (threadIdx.x == 0) value[0] += delta;
            }"#
        );
        let kernel = compiler.compile(source, &CompileOptions::default())?.kernel::<Increment>()?;
        Ok(Self { context, pool, kernel })
    }

    /// Captures one increment while `other` runs on another thread, then
    /// replays the graph and checks its result. The capture ends shortly
    /// after `other` calls its argument, which it does before any teardown
    /// that may wait for the capture.
    fn capture_during(
        &self,
        other: impl FnOnce(&dyn Fn()) -> mircuda::Result<()> + Send,
    ) -> mircuda::Result<()> {
        let stream = self.context.create_stream()?;
        let mut value = self.pool.allocate_zeroed::<u32>(&stream, 1)?;
        stream.synchronize()?;
        let started = Barrier::new(2);
        let finished = Barrier::new(2);
        let mut graph = std::thread::scope(|scope| {
            let other = scope.spawn(|| {
                let signalled = AtomicBool::new(false);
                let reached = || {
                    if !signalled.swap(true, Ordering::AcqRel) {
                        finished.wait();
                    }
                };
                started.wait();
                let result = other(&reached);
                reached();
                result
            });
            let sequence = Sequence {
                kernel: &self.kernel,
                stream: &stream,
                value: &mut value,
                started: &started,
                finished: &finished,
            };
            let graph = stream.capture(sequence, capture_while_another_thread_works);
            other.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))?;
            graph
        })?;
        graph.launch(&stream)?;
        drop(graph);
        let mut host = self.context.allocate_pinned::<u32>(1)?;
        stream.copy_to_host(&value, &mut host)?;
        assert_eq!(host.to_vec()?, [1]);
        Ok(())
    }
}

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
