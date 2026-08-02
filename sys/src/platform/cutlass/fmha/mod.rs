use std::{ffi::c_void, sync::Arc};

use cudarc::driver::CudaStream;

use super::super::{
    driver::{Context, Stream},
    memory::{DeviceBuffer, ensure_stream},
};
use crate::{Error, Result};

mod paged;
mod validation;

use validation::{check, validate, validate_varlen};

unsafe extern "C" {
    fn mircuda_fmha_bf16_execute(
        query: *const c_void,
        keys: *const c_void,
        values: *const c_void,
        output: *mut c_void,
        query_tokens: i32,
        context_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        value_head_dim: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn mircuda_fmha_bf16_varlen_execute(
        query: *const c_void,
        keys: *const c_void,
        values: *const c_void,
        output: *mut c_void,
        query_starts: *const i32,
        key_starts: *const i32,
        max_query_tokens: i32,
        max_context_tokens: i32,
        batch_size: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        value_head_dim: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FmhaBf16Spec {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_head_dim: usize,
}

#[derive(Debug)]
pub struct FmhaBf16Plan {
    stream: Arc<CudaStream>,
    spec: FmhaBf16Spec,
}

impl Context {
    pub fn create_fmha_bf16_plan(
        &self,
        stream: &Stream,
        spec: FmhaBf16Spec,
    ) -> Result<FmhaBf16Plan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        if spec.query_heads == 0
            || spec.kv_heads == 0
            || !spec.query_heads.is_multiple_of(spec.kv_heads)
            || spec.head_dim != spec.value_head_dim
            || !matches!(spec.head_dim, 64 | 128)
        {
            return Err(Error::InvalidMatmulBuffer);
        }
        self.inner.bind_to_thread()?;
        Ok(FmhaBf16Plan { stream: stream.inner.clone(), spec })
    }
}

impl FmhaBf16Plan {
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        query: &DeviceBuffer,
        keys: &DeviceBuffer,
        values: &DeviceBuffer,
        output: &DeviceBuffer,
        query_tokens: usize,
        context_tokens: usize,
        key_offset_bytes: usize,
        value_offset_bytes: usize,
        scale: f32,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        for buffer in [query, keys, values, output] {
            ensure_stream(buffer, stream)?;
        }
        validate(
            self.spec,
            query,
            keys,
            values,
            output,
            query_tokens,
            context_tokens,
            key_offset_bytes,
            value_offset_bytes,
        )?;
        self.stream.context().bind_to_thread()?;
        let key_pointer = keys
            .pointer()
            .checked_add(u64::try_from(key_offset_bytes)?)
            .ok_or(Error::InvalidMatmulBuffer)?;
        let value_pointer = values
            .pointer()
            .checked_add(u64::try_from(value_offset_bytes)?)
            .ok_or(Error::InvalidMatmulBuffer)?;
        // SAFETY: validation proves that every typed BF16 span lies within its
        // stream-owned allocation and the native kernel only enqueues work.
        let status = unsafe {
            mircuda_fmha_bf16_execute(
                query.pointer() as *const c_void,
                key_pointer as *const c_void,
                value_pointer as *const c_void,
                output.pointer() as *mut c_void,
                i32::try_from(query_tokens)?,
                i32::try_from(context_tokens)?,
                i32::try_from(self.spec.query_heads)?,
                i32::try_from(self.spec.kv_heads)?,
                i32::try_from(self.spec.head_dim)?,
                i32::try_from(self.spec.value_head_dim)?,
                scale,
                stream.inner.cu_stream().cast(),
            )
        };
        check(status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_varlen(
        &self,
        stream: &Stream,
        query: &DeviceBuffer,
        keys: &DeviceBuffer,
        values: &DeviceBuffer,
        output: &DeviceBuffer,
        query_starts: &DeviceBuffer,
        key_starts: &DeviceBuffer,
        batch_size: usize,
        total_query_tokens: usize,
        total_context_tokens: usize,
        max_query_tokens: usize,
        max_context_tokens: usize,
        scale: f32,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        for buffer in [query, keys, values, output, query_starts, key_starts] {
            ensure_stream(buffer, stream)?;
        }
        validate_varlen(
            self.spec,
            query,
            keys,
            values,
            output,
            query_starts,
            key_starts,
            batch_size,
            total_query_tokens,
            total_context_tokens,
            max_query_tokens,
            max_context_tokens,
        )?;
        self.stream.context().bind_to_thread()?;
        // SAFETY: validation proves all packed BF16 and prefix-sum spans lie
        // within stream-owned allocations; the native call only enqueues work.
        let status = unsafe {
            mircuda_fmha_bf16_varlen_execute(
                query.pointer() as *const c_void,
                keys.pointer() as *const c_void,
                values.pointer() as *const c_void,
                output.pointer() as *mut c_void,
                query_starts.pointer() as *const i32,
                key_starts.pointer() as *const i32,
                i32::try_from(max_query_tokens)?,
                i32::try_from(max_context_tokens)?,
                i32::try_from(batch_size)?,
                i32::try_from(self.spec.query_heads)?,
                i32::try_from(self.spec.kv_heads)?,
                i32::try_from(self.spec.head_dim)?,
                i32::try_from(self.spec.value_head_dim)?,
                scale,
                stream.inner.cu_stream().cast(),
            )
        };
        check(status)
    }
}
