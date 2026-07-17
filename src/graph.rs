use std::marker::PhantomData;

use crate::{KernelSignature, LaunchConfig, Result, Stream, TypedKernel};

/// CUDA stream-capture isolation policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CaptureMode {
    /// Capture interactions with every thread.
    Global,
    /// Capture operations submitted by the current thread.
    #[default]
    ThreadLocal,
    /// Permit otherwise unsafe interactions with non-captured streams.
    Relaxed,
}

impl CaptureMode {
    const fn native(self) -> mircuda_sys::CaptureMode {
        match self {
            Self::Global => mircuda_sys::CaptureMode::Global,
            Self::ThreadLocal => mircuda_sys::CaptureMode::ThreadLocal,
            Self::Relaxed => mircuda_sys::CaptureMode::Relaxed,
        }
    }
}

/// Instantiated CUDA graph owning every resource used by captured work.
#[derive(Debug)]
pub struct Graph<Resources> {
    native: mircuda_sys::Graph,
    resources: Resources,
}

/// A captured node whose argument ABI is tied to one kernel signature.
#[derive(Debug)]
pub struct KernelNode<Signature: KernelSignature> {
    native: mircuda_sys::KernelNode,
    signature: PhantomData<Signature>,
}

impl<Signature: KernelSignature> Clone for KernelNode<Signature> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Signature: KernelSignature> Copy for KernelNode<Signature> {}

impl<Signature: KernelSignature> KernelNode<Signature> {
    pub(crate) const fn new(native: mircuda_sys::KernelNode) -> Self {
        Self { native, signature: PhantomData }
    }
}

impl Stream {
    /// Captures a thread-local sequence and transfers its resources into the graph.
    pub fn capture<Resources, CaptureError>(
        &self,
        resources: Resources,
        operation: fn(&mut Resources) -> std::result::Result<(), CaptureError>,
    ) -> std::result::Result<Graph<Resources>, CaptureError>
    where
        CaptureError: From<crate::Error>,
    {
        self.capture_with(CaptureMode::ThreadLocal, resources, operation)
    }

    /// Captures a sequence with an explicit isolation policy and resource set.
    pub fn capture_with<Resources, CaptureError>(
        &self,
        mode: CaptureMode,
        mut resources: Resources,
        operation: fn(&mut Resources) -> std::result::Result<(), CaptureError>,
    ) -> std::result::Result<Graph<Resources>, CaptureError>
    where
        CaptureError: From<crate::Error>,
    {
        self.begin_capture(mode)?;
        let operation_result = operation(&mut resources);
        let graph_result = self.end_capture();
        operation_result?;
        Ok(Graph { native: graph_result?, resources })
    }

    fn begin_capture(&self, mode: CaptureMode) -> Result<()> {
        Ok(self.native.begin_capture(mode.native())?)
    }

    fn end_capture(&self) -> Result<mircuda_sys::Graph> {
        Ok(self.native.end_capture()?)
    }
}

impl<Resources> Graph<Resources> {
    /// Enqueues one replay on the stream used during capture.
    pub fn launch(&mut self, stream: &Stream) -> Result<()> {
        Ok(self.native.launch(&stream.native)?)
    }

    /// Returns immutable capture metadata and resources owned by this graph.
    #[must_use]
    pub const fn resources(&self) -> &Resources {
        &self.resources
    }

    /// Borrows retained resources without replacing or destroying the native graph.
    ///
    /// Work enqueued by `operation` must use the same explicit stream ordering as
    /// graph replay when it touches captured allocations.
    pub fn with_resources_mut<T>(&mut self, operation: impl FnOnce(&mut Resources) -> T) -> T {
        operation(&mut self.resources)
    }

    /// Destroys the native graph and returns its retained resources.
    ///
    /// This does not synchronize the capture stream or copy device memory.
    /// Callers can use the resources to capture a replacement graph.
    #[must_use]
    pub fn into_resources(self) -> Resources {
        let Self { native: _native, resources } = self;
        resources
    }

    /// Returns captured nodes executing `kernel`, preserving its argument ABI.
    pub fn kernel_nodes<Signature: KernelSignature>(
        &self,
        kernel: &TypedKernel<Signature>,
    ) -> Result<Vec<KernelNode<Signature>>> {
        Ok(self
            .native
            .kernel_nodes(kernel.native())?
            .into_iter()
            .map(KernelNode::new)
            .collect())
    }

    /// Rebinds one kernel node without recapturing or synchronizing the graph.
    pub fn update_kernel<Signature, Arguments>(
        &mut self,
        node: &KernelNode<Signature>,
        kernel: &TypedKernel<Signature>,
        config: LaunchConfig,
        arguments: Arguments,
    ) -> Result<()>
    where
        Signature: KernelSignature,
        Arguments: for<'borrow> FnOnce(&'borrow mut Resources) -> Signature::Arguments<'borrow>,
    {
        let mut encoded = Signature::encode(arguments(&mut self.resources));
        Ok(self.native.update_kernel(
            node.native,
            kernel.native(),
            config.native(),
            &mut encoded.native,
        )?)
    }

    /// Rebinds a node when preparing typed arguments can fail.
    pub fn try_update_kernel<Signature, Arguments, UpdateError>(
        &mut self,
        node: &KernelNode<Signature>,
        kernel: &TypedKernel<Signature>,
        config: LaunchConfig,
        arguments: Arguments,
    ) -> std::result::Result<(), UpdateError>
    where
        Signature: KernelSignature,
        Arguments:
            for<'borrow> FnOnce(
                &'borrow mut Resources,
            )
                -> std::result::Result<Signature::Arguments<'borrow>, UpdateError>,
        UpdateError: From<crate::Error>,
    {
        let mut encoded = Signature::encode(arguments(&mut self.resources)?);
        let update: crate::Result<()> = (|| {
            self.native.update_kernel(
                node.native,
                kernel.native(),
                config.native(),
                &mut encoded.native,
            )?;
            Ok(())
        })();
        update?;
        Ok(())
    }
}
