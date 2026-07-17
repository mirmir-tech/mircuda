use std::sync::Arc;

use cudarc::driver::{CudaStream, result, sys};

use super::{
    compiler::{Kernel, KernelArgument, LaunchConfig},
    driver::Stream,
};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CaptureMode {
    Global,
    #[default]
    ThreadLocal,
    Relaxed,
}

impl CaptureMode {
    const fn native(self) -> sys::CUstreamCaptureMode {
        match self {
            Self::Global => sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            Self::ThreadLocal => sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            Self::Relaxed => sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        }
    }
}

pub struct Graph {
    raw: sys::CUgraph,
    executable: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

#[derive(Clone, Copy, Debug)]
pub struct KernelNode {
    raw: sys::CUgraphNode,
    graph: sys::CUgraph,
}

// SAFETY: CUDA graph handles are context-owned, graph methods bind that
// context to the calling thread, and mutable graph access serializes use.
unsafe impl Send for Graph {}

// SAFETY: a node is a non-owning handle retained by its owning graph and is
// only consumed while mutably updating that graph.
unsafe impl Send for KernelNode {}

impl std::fmt::Debug for Graph {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Graph").finish_non_exhaustive()
    }
}

impl Stream {
    pub fn begin_capture(&self, mode: CaptureMode) -> Result<()> {
        Ok(self.inner.begin_capture(mode.native())?)
    }

    pub fn end_capture(&self) -> Result<Graph> {
        self.inner.context().bind_to_thread()?;
        // SAFETY: capture is active on this live stream.
        let raw = unsafe { result::stream::end_capture(self.inner.cu_stream())? };
        if raw.is_null() {
            return Err(Error::EmptyGraph);
        }
        let mut executable = std::ptr::null_mut();
        // SAFETY: raw is a live captured graph and zero requests default instantiation.
        let instantiated = unsafe { sys::cuGraphInstantiateWithFlags(&raw mut executable, raw, 0) };
        if let Err(source) = instantiated.result() {
            // SAFETY: raw was returned by the successful capture above.
            let _result = unsafe { result::graph::destroy(raw) };
            return Err(source.into());
        }
        // SAFETY: executable and stream are live and belong to the bound context.
        if let Err(source) = unsafe { result::graph::upload(executable, self.inner.cu_stream()) } {
            // SAFETY: both handles were created successfully in this function.
            let _exec_result = unsafe { result::graph::exec_destroy(executable) };
            // SAFETY: raw remains live until this matching destroy.
            let _graph_result = unsafe { result::graph::destroy(raw) };
            return Err(source.into());
        }
        Ok(Graph {
            raw,
            executable,
            stream: self.inner.clone(),
        })
    }

    pub fn captured_kernel_node(&self, kernel: &Kernel) -> Result<KernelNode> {
        self.inner.context().bind_to_thread()?;
        let mut status = std::mem::MaybeUninit::uninit();
        let mut graph = std::ptr::null_mut();
        let mut dependencies = std::ptr::null();
        let mut edge_data = std::ptr::null();
        let mut count = 0;
        // SAFETY: every output points to writable storage and the stream is live.
        unsafe {
            sys::cuStreamGetCaptureInfo_v3(
                self.inner.cu_stream(),
                status.as_mut_ptr(),
                std::ptr::null_mut(),
                &raw mut graph,
                &raw mut dependencies,
                &raw mut edge_data,
                &raw mut count,
            )
        }
        .result()?;
        // SAFETY: CUDA initialized status on success.
        let active = unsafe { status.assume_init() }
            == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE;
        if !active || graph.is_null() || dependencies.is_null() || count == 0 {
            return Err(Error::MissingCapturedKernel);
        }
        // SAFETY: CUDA returned `count` dependencies valid for the active capture.
        let dependencies = unsafe { std::slice::from_raw_parts(dependencies, count) };
        let mut matched = None;
        for raw in dependencies.iter().copied() {
            if kernel_function(raw)? == Some(kernel.handle()) && matched.replace(raw).is_some() {
                return Err(Error::MissingCapturedKernel);
            }
        }
        let raw = matched.ok_or(Error::MissingCapturedKernel)?;
        Ok(KernelNode { raw, graph })
    }
}

impl Graph {
    pub fn launch(&mut self, stream: &Stream) -> Result<()> {
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        self.stream.context().bind_to_thread()?;
        // SAFETY: executable is live and launch is serialized by mutable access.
        Ok(unsafe { result::graph::launch(self.executable, self.stream.cu_stream()) }?)
    }

    pub fn kernel_nodes(&self, kernel: &Kernel) -> Result<Vec<KernelNode>> {
        self.stream.context().bind_to_thread()?;
        let mut count = 0;
        // SAFETY: null nodes requests the required element count.
        unsafe { sys::cuGraphGetNodes(self.raw, std::ptr::null_mut(), &raw mut count) }.result()?;
        let mut nodes = vec![std::ptr::null_mut(); count];
        // SAFETY: nodes has storage for the count returned by the first call.
        unsafe { sys::cuGraphGetNodes(self.raw, nodes.as_mut_ptr(), &raw mut count) }.result()?;
        nodes.truncate(count);
        nodes
            .into_iter()
            .filter_map(|raw| match kernel_function(raw) {
                Ok(Some(function)) if function == kernel.handle() => {
                    Some(Ok(KernelNode { raw, graph: self.raw }))
                },
                Ok(_) => None,
                Err(source) => Some(Err(source)),
            })
            .collect()
    }

    pub fn update_kernel(
        &mut self,
        node: KernelNode,
        kernel: &Kernel,
        config: LaunchConfig,
        arguments: &mut [KernelArgument],
    ) -> Result<()> {
        if node.graph != self.raw {
            return Err(Error::GraphNodeMismatch);
        }
        let mut pointers =
            kernel.argument_pointers(&Stream { inner: self.stream.clone() }, arguments)?;
        let params = sys::CUDA_KERNEL_NODE_PARAMS {
            func: kernel.handle(),
            gridDimX: config.grid.0,
            gridDimY: config.grid.1,
            gridDimZ: config.grid.2,
            blockDimX: config.block.0,
            blockDimY: config.block.1,
            blockDimZ: config.block.2,
            sharedMemBytes: config.shared_memory_bytes,
            kernelParams: pointers.as_mut_ptr(),
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(),
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: node belongs to this executable and params point to live ABI values.
        unsafe {
            sys::cuGraphExecKernelNodeSetParams_v2(self.executable, node.raw, &raw const params)
        }
        .result()?;
        Ok(())
    }
}

fn kernel_function(node: sys::CUgraphNode) -> Result<Option<sys::CUfunction>> {
    let mut node_type = std::mem::MaybeUninit::uninit();
    // SAFETY: node_type is writable and node came from a live graph.
    unsafe { sys::cuGraphNodeGetType(node, node_type.as_mut_ptr()) }.result()?;
    // SAFETY: CUDA initialized node_type on success.
    if unsafe { node_type.assume_init() } != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
        return Ok(None);
    }
    let mut params = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::zeroed();
    // SAFETY: this is a kernel node and params is writable.
    unsafe { sys::cuGraphKernelNodeGetParams_v2(node, params.as_mut_ptr()) }.result()?;
    // SAFETY: CUDA initialized params on success.
    Ok(Some(unsafe { params.assume_init() }.func))
}

impl Drop for Graph {
    fn drop(&mut self) {
        let context = self.stream.context();
        context.record_err(context.bind_to_thread());
        // SAFETY: this object solely owns both live graph handles.
        context.record_err(unsafe { result::graph::exec_destroy(self.executable) });
        // SAFETY: executable was destroyed first and raw is still live.
        context.record_err(unsafe { result::graph::destroy(self.raw) });
    }
}
