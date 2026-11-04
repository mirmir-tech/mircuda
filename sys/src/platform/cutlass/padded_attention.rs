use std::{ffi::c_void, sync::Arc};

use cudarc::driver::CudaStream;

use super::super::{
    driver::{Context, Stream},
    memory::{DeviceBuffer, ensure_stream},
};
use crate::{Error, Result};

unsafe extern "C" {
    fn mircuda_fmha_padded_execute(
        data_type: i32,
        qkv: *const c_void,
        output: *mut c_void,
        lengths: *const i32,
        sequences: i32,
        length: i32,
        heads: i32,
        head_dim: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
}

/// Element type of every operand of padded fused attention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PaddedAttentionDataType {
    Bf16,
    F32,
}

impl PaddedAttentionDataType {
    const fn native(self) -> i32 {
        match self {
            Self::Bf16 => 0,
            Self::F32 => 1,
        }
    }

    const fn bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PaddedAttentionSpec {
    pub heads: usize,
    pub head_dim: usize,
    pub data_type: PaddedAttentionDataType,
}

#[derive(Debug)]
pub struct PaddedAttentionPlan {
    stream: Arc<CudaStream>,
    spec: PaddedAttentionSpec,
}

impl Context {
    pub fn create_padded_attention_plan(
        &self,
        stream: &Stream,
        spec: PaddedAttentionSpec,
    ) -> Result<PaddedAttentionPlan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        if spec.heads == 0 || !matches!(spec.head_dim, 64 | 128) {
            return Err(Error::InvalidMatmulBuffer);
        }
        Ok(PaddedAttentionPlan { stream: stream.inner.clone(), spec })
    }
}

impl PaddedAttentionPlan {
    /// `lengths` holds one `u32` per sequence; the kernel clamps each to
    /// `[1, length]`, so its contents cannot move a read outside a sequence.
    pub fn execute(
        &self,
        stream: &Stream,
        (qkv, lengths, output): (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer),
        length: usize,
        scale: f32,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        for buffer in [qkv, lengths, output] {
            ensure_stream(buffer, stream)?;
        }
        let sequences = lengths.bytes() / size_of::<u32>();
        let row = self.spec.heads * self.spec.head_dim * self.spec.data_type.bytes();
        let tokens = sequences.checked_mul(length).ok_or(Error::InvalidMatmulBuffer)?;
        let valid = sequences > 0
            && length > 0
            && lengths.bytes() == sequences * size_of::<u32>()
            && tokens.checked_mul(3 * row) == Some(qkv.bytes())
            && tokens.checked_mul(row) == Some(output.bytes());
        if !valid {
            return Err(Error::InvalidMatmulBuffer);
        }
        self.stream.context().bind_to_thread()?;
        // SAFETY: the projection and output hold exactly `sequences × length`
        // rows, all buffers share the retained stream, and the kernel clamps
        // every sequence length to its padded extent.
        let status = unsafe {
            mircuda_fmha_padded_execute(
                self.spec.data_type.native(),
                qkv.pointer() as *const c_void,
                output.pointer() as *mut c_void,
                lengths.pointer() as *const i32,
                i32::try_from(sequences)?,
                i32::try_from(length)?,
                i32::try_from(self.spec.heads)?,
                i32::try_from(self.spec.head_dim)?,
                scale,
                stream.inner.cu_stream().cast(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(Error::Cutlass(status))
        }
    }
}
