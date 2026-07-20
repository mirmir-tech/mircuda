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

#[derive(Clone, Copy, Debug)]
pub struct KernelNode;

impl Stream {
    pub const fn begin_capture(&self, _mode: CaptureMode) -> Result<()> {
        Err(unsupported())
    }

    pub const fn end_capture(&self) -> Result<Graph> {
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
