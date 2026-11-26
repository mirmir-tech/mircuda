#![cfg(target_os = "linux")]

//! Work on other threads of the same device must not invalidate a
//! thread-local capture: neither another runtime starting up, nor the release
//! of a buffer used on several streams, nor the release of a cuBLAS handle.
//! A capture that ends early, by a panic or a failed begin, must release the
//! capture gate so that later teardown runs instead of staying deferred.

use std::{
    panic::AssertUnwindSafe,
    sync::{
        Barrier, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use mircuda::{
    CompileOptions, Compiler, Context, DeviceBuffer, Driver, LaunchConfig, MemoryPool, Stream,
    TypedKernel, cuda_export, cuda_kernel,
};

mod allocation;
mod lifecycle;
mod release;

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

/// The capture gate is process-wide: tests that observe it run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

struct Fixture {
    context: Context,
    pool: MemoryPool,
    kernel: TypedKernel<Increment>,
    _serial: MutexGuard<'static, ()>,
}

fn new_context() -> mircuda::Result<Context> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    driver.create_context(device)
}

impl Fixture {
    fn new() -> mircuda::Result<Self> {
        let serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let context = new_context()?;
        let pool = context.default_memory_pool()?;
        let compiler = Compiler::new(context.clone())?;
        let source = cuda_kernel!(
            r#"extern "C" __global__ void increment(unsigned int* value, unsigned int delta) {
                if (threadIdx.x == 0) value[0] += delta;
            }"#
        );
        let kernel = compiler.compile(source, &CompileOptions::default())?.kernel::<Increment>()?;
        Ok(Self { context, pool, kernel, _serial: serial })
    }

    /// Captures one increment on a new stream while `other` runs on another
    /// thread; see [`Self::capture_on`].
    fn capture_during(
        &self,
        other: impl FnOnce(&dyn Fn()) -> mircuda::Result<()> + Send,
    ) -> mircuda::Result<()> {
        self.capture_on(&self.context.create_stream()?, other)
    }

    /// Captures one increment on `stream` while `other` runs on another
    /// thread, then replays the graph and checks its result. The capture ends
    /// shortly after `other` calls its argument, which it does before any
    /// teardown that may wait for the capture.
    fn capture_on(
        &self,
        stream: &Stream,
        other: impl FnOnce(&dyn Fn()) -> mircuda::Result<()> + Send,
    ) -> mircuda::Result<()> {
        let mut value = self.pool.allocate_zeroed::<u32>(stream, 1)?;
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
                // A failure in `other` still lets the capture end.
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| other(&reached)));
                reached();
                result.unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            });
            let sequence = Sequence {
                kernel: &self.kernel,
                stream,
                value: &mut value,
                started: &started,
                finished: &finished,
            };
            let graph = stream.capture(sequence, capture_while_another_thread_works);
            other.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))?;
            graph
        })?;
        graph.launch(stream)?;
        drop(graph);
        let mut host = self.context.allocate_pinned::<u32>(1)?;
        stream.copy_to_host(&value, &mut host)?;
        assert_eq!(host.to_vec()?, [1]);
        Ok(())
    }
}
