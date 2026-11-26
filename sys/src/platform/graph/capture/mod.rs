//! A stream capture in progress, held by the thread that began it.

use std::marker::PhantomData;

use cudarc::driver::{result, sys};

use super::{
    super::{capture_gate, driver::make_current},
    CaptureMode, Graph,
};
use crate::{Error, Result, platform::driver::Stream};

/// A capture in progress on one stream.
///
/// [`Self::end`] instantiates its graph; dropping the capture instead, a
/// panic included, ends and discards it. Either way the capture gate is
/// released, and since the guard stays on its thread, no other thread can end
/// the capture.
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
        capture_gate::enter(|| self.inner.begin_capture(mode.native()))?;
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
        let raw = self.finish()?;
        if raw.is_null() {
            return Err(Error::EmptyGraph);
        }
        let stream = &self.stream.inner;
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

    /// Makes the context current and ends the capture. The capture stays
    /// open, for [`Drop`] to end, until the end is attempted. Raw driver calls
    /// leave an error another operation recorded on the context in place.
    fn finish(&mut self) -> Result<sys::CUgraph> {
        let stream = &self.stream.inner;
        make_current(stream.context())?;
        self.open = false;
        // SAFETY: capture is active on this live stream.
        Ok(unsafe { result::stream::end_capture(stream.cu_stream())? })
    }
}

impl Drop for Capture<'_> {
    /// Ends and discards a capture still open. A failure to end it is
    /// recorded on the context for its next caller, unless the capture was
    /// invalidated: that only reports why it is being discarded.
    fn drop(&mut self) {
        if self.open {
            let stream = &self.stream.inner;
            // SAFETY: the capture this guard began is still active on the stream.
            let ended = make_current(stream.context())
                .and_then(|()| unsafe { result::stream::end_capture(stream.cu_stream()) })
                .and_then(|raw| {
                    if raw.is_null() {
                        return Ok(());
                    }
                    // SAFETY: raw is the discarded capture's graph, owned here.
                    unsafe { result::graph::destroy(raw) }
                });
            if ended.is_err_and(|error| {
                error.0 != sys::cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
            }) {
                stream.context().record_err(ended);
            }
        }
        capture_gate::leave();
    }
}

#[cfg(test)]
mod tests;
