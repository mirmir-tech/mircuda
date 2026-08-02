use super::FmhaBf16Spec;
use crate::{Error, Result, platform::memory::DeviceBuffer};

#[allow(clippy::too_many_arguments)]
pub(super) fn validate(
    spec: FmhaBf16Spec,
    query: &DeviceBuffer,
    keys: &DeviceBuffer,
    values: &DeviceBuffer,
    output: &DeviceBuffer,
    query_tokens: usize,
    context_tokens: usize,
    key_offset: usize,
    value_offset: usize,
) -> Result<()> {
    let bf16_bytes = size_of::<u16>();
    let query_bytes = elements(query_tokens, spec.query_heads, spec.head_dim, bf16_bytes)?;
    let key_bytes = elements(context_tokens, spec.kv_heads, spec.head_dim, bf16_bytes)?;
    let value_bytes = elements(context_tokens, spec.kv_heads, spec.value_head_dim, bf16_bytes)?;
    let output_bytes = elements(query_tokens, spec.query_heads, spec.value_head_dim, bf16_bytes)?;
    let valid = query_tokens > 0
        && context_tokens >= query_tokens
        && query.bytes() == query_bytes
        && output.bytes() == output_bytes
        && key_offset.checked_add(key_bytes).is_some_and(|end| end <= keys.bytes())
        && value_offset.checked_add(value_bytes).is_some_and(|end| end <= values.bytes());
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidMatmulBuffer)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_varlen(
    spec: FmhaBf16Spec,
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
) -> Result<()> {
    let bf16_bytes = size_of::<u16>();
    let starts_bytes = batch_size
        .checked_add(1)
        .and_then(|value| value.checked_mul(size_of::<u32>()))
        .ok_or(Error::InvalidMatmulBuffer)?;
    let valid = batch_size > 0
        && total_query_tokens >= max_query_tokens
        && total_context_tokens >= max_context_tokens
        && query.bytes()
            == elements(total_query_tokens, spec.query_heads, spec.head_dim, bf16_bytes)?
        && output.bytes()
            == elements(total_query_tokens, spec.query_heads, spec.value_head_dim, bf16_bytes)?
        && keys.bytes()
            >= elements(total_context_tokens, spec.kv_heads, spec.head_dim, bf16_bytes)?
        && values.bytes()
            >= elements(total_context_tokens, spec.kv_heads, spec.value_head_dim, bf16_bytes)?
        && query_starts.bytes() >= starts_bytes
        && key_starts.bytes() >= starts_bytes;
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidMatmulBuffer)
    }
}

fn elements(rows: usize, heads: usize, columns: usize, bytes: usize) -> Result<usize> {
    rows.checked_mul(heads)
        .and_then(|value| value.checked_mul(columns))
        .and_then(|value| value.checked_mul(bytes))
        .ok_or(Error::InvalidMatmulBuffer)
}

pub(super) const fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Cutlass(status))
    }
}
