use super::{Stream, unsupported};
use crate::Result;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CaptureMode {
    Global,
    #[default]
    ThreadLocal,
    Relaxed,
}

#[derive(Debug)]
pub struct Graph;

/// A stream capture in progress; capture is unsupported without CUDA.
#[must_use = "dropping a capture discards it"]
pub struct Capture<'stream>(std::marker::PhantomData<&'stream Stream>);

impl Capture<'_> {
    pub const fn end(self) -> Result<Graph> {
        Err(unsupported())
    }
}

/// Captures in progress and the teardown waiting for them; always none
/// without CUDA.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CaptureActivity {
    pub capturing_threads: usize,
    pub deferred_teardown: usize,
}

#[must_use]
pub const fn capture_activity() -> CaptureActivity {
    CaptureActivity {
        capturing_threads: 0,
        deferred_teardown: 0,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KernelNode;

impl Stream {
    pub const fn begin_capture(&self, _mode: CaptureMode) -> Result<Capture<'_>> {
        Err(unsupported())
    }

    pub const fn captured_kernel_node(&self, _kernel: &super::Kernel) -> Result<KernelNode> {
        Err(unsupported())
    }
}

impl Graph {
    pub const fn launch(&mut self, _stream: &Stream) -> Result<()> {
        Err(unsupported())
    }

    pub const fn kernel_nodes(&self, _kernel: &super::Kernel) -> Result<Vec<KernelNode>> {
        Err(unsupported())
    }

    pub const fn update_kernel(
        &mut self,
        _node: KernelNode,
        _kernel: &super::Kernel,
        _config: super::LaunchConfig,
        _arguments: &mut [super::KernelArgument],
    ) -> Result<()> {
        Err(unsupported())
    }
}
