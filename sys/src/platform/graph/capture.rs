//! A stream capture in progress, held by the thread that began it.

use std::marker::PhantomData;

use cudarc::driver::{result, sys};

use super::{super::capture_gate, CaptureMode, Graph};
use crate::{Error, Result, platform::driver::Stream};

/// A capture in progress on one stream.
///
/// [`Self::end`] instantiates its graph; dropping it on any other exit path,
/// a panic included, ends and discards the capture. Either way the capture
/// gate is released, and since the guard stays on its thread, no other
/// thread can end the capture.
#[must_use = "dropping a capture discards it"]
pub struct Capture<'stream> {
    stream: &'stream Stream,
    open: bool,
    /// Not `Send`: the capture gate counts captures per thread.
    thread: PhantomData<*const ()>,
}

impl Stream {
    /// Begins capturing this stream on the calling thread.
    pub fn begin_capture(&self, mode: CaptureMode) -> Result<Capture<'_>> {
        capture_gate::enter();
        if let Err(error) = self.inner.begin_capture(mode.native()) {
            capture_gate::leave();
            return Err(error.into());
        }
        Ok(Capture {
            stream: self,
            open: true,
            thread: PhantomData,
        })
    }
}

impl Capture<'_> {
    /// Ends the capture and instantiates its graph.
    pub fn end(mut self) -> Result<Graph> {
        self.open = false;
        let stream = &self.stream.inner;
        stream.context().bind_to_thread()?;
        // SAFETY: capture is active on this live stream.
        let raw = unsafe { result::stream::end_capture(stream.cu_stream())? };
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
        if let Err(source) = unsafe { result::graph::upload(executable, stream.cu_stream()) } {
            // SAFETY: both handles were created successfully in this function.
            let _exec_result = unsafe { result::graph::exec_destroy(executable) };
            // SAFETY: raw remains live until this matching destroy.
            let _graph_result = unsafe { result::graph::destroy(raw) };
            return Err(source.into());
        }
        Ok(Graph { raw, executable, stream: stream.clone() })
    }
}

impl Drop for Capture<'_> {
    fn drop(&mut self) {
        if self.open {
            let stream = &self.stream.inner;
            let context = stream.context();
            context.record_err(context.bind_to_thread());
            // SAFETY: the capture this guard began is still active on the stream.
            match unsafe { result::stream::end_capture(stream.cu_stream()) } {
                Ok(raw) if !raw.is_null() => {
                    // SAFETY: raw is the discarded capture's graph, owned here.
                    context.record_err(unsafe { result::graph::destroy(raw) });
                },
                Ok(_) => {},
                Err(error) => context.record_err::<()>(Err(error)),
            }
        }
        capture_gate::leave();
    }
}
