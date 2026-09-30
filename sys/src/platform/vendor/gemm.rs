use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cudarc::driver::CudaStream;

use super::{
    super::{
        driver::{Context, Stream},
        memory::{DeviceBuffer, ensure_stream},
    },
    cublas::CublasDataType,
};
use crate::{Error, Result};

#[repr(C)]
struct Descriptor {
    transpose_first: i32,
    transpose_second: i32,
    m: i32,
    n: i32,
    k: i32,
    leading_first: i64,
    leading_second: i64,
    leading_output: i64,
    stride_first: i64,
    stride_second: i64,
    stride_output: i64,
    batch: i32,
    input_type: i32,
    output_type: i32,
}

unsafe extern "C" {
    fn mircuda_cublas_gemm_create(
        descriptor: *const Descriptor,
        stream: *mut c_void,
        output: *mut *mut c_void,
    ) -> i32;
    fn mircuda_cublas_gemm_execute(
        plan: *mut c_void,
        stream: *mut c_void,
        first: *const c_void,
        second: *const c_void,
        output: *mut c_void,
        alpha: f32,
        beta: f32,
    ) -> i32;
    fn mircuda_cublas_gemm_destroy(plan: *mut c_void);
}

/// One operand of a strided batched product.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CublasGemmOperand {
    /// Elements between the starts of consecutive stored rows (columns in
    /// cuBLAS terms).
    pub leading: usize,
    /// Elements between the starts of consecutive batch members.
    pub stride: usize,
    /// Whether the operand is stored transposed relative to its role.
    pub transposed: bool,
}

/// `C = op(first) · op(second)` in cuBLAS column-major terms, batched.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CublasGemmSpec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
    pub first: CublasGemmOperand,
    pub second: CublasGemmOperand,
    pub output_leading: usize,
    pub output_stride: usize,
    pub input_type: CublasDataType,
    pub output_type: CublasDataType,
}

impl CublasGemmSpec {
    /// Elements each operand spans for one launch, starting at its offset.
    fn extents(&self) -> Option<[usize; 3]> {
        let span = |rows: usize, columns: usize, leading: usize| {
            columns.checked_sub(1)?.checked_mul(leading)?.checked_add(rows)
        };
        let batched = |stride: usize, one: usize| {
            self.batch.checked_sub(1)?.checked_mul(stride)?.checked_add(one)
        };
        let (m, n, k) = (self.m, self.n, self.k);
        let first = if self.first.transposed {
            span(k, m, self.first.leading)
        } else {
            span(m, k, self.first.leading)
        };
        let second = if self.second.transposed {
            span(n, k, self.second.leading)
        } else {
            span(k, n, self.second.leading)
        };
        Some([
            batched(self.first.stride, first?)?,
            batched(self.second.stride, second?)?,
            batched(self.output_stride, span(m, n, self.output_leading)?)?,
        ])
    }
}

#[derive(Debug)]
pub struct CublasGemmPlan {
    raw: NonNull<c_void>,
    stream: Arc<CudaStream>,
    spec: CublasGemmSpec,
    extents: [usize; 3],
}

// SAFETY: the plan retains its CUDA stream, binds its context before native
// use or destruction, and execution requires exclusive mutable access.
unsafe impl Send for CublasGemmPlan {}

impl Context {
    pub fn create_cublas_gemm_plan(
        &self,
        stream: &Stream,
        spec: CublasGemmSpec,
    ) -> Result<CublasGemmPlan> {
        if !Arc::ptr_eq(&self.inner, stream.inner.context()) {
            return Err(Error::ContextMismatch);
        }
        let extents = spec.extents().ok_or(Error::InvalidMatmulBuffer)?;
        let descriptor = Descriptor {
            transpose_first: i32::from(spec.first.transposed),
            transpose_second: i32::from(spec.second.transposed),
            m: i32::try_from(spec.m)?,
            n: i32::try_from(spec.n)?,
            k: i32::try_from(spec.k)?,
            leading_first: i64::try_from(spec.first.leading)?,
            leading_second: i64::try_from(spec.second.leading)?,
            leading_output: i64::try_from(spec.output_leading)?,
            stride_first: i64::try_from(spec.first.stride)?,
            stride_second: i64::try_from(spec.second.stride)?,
            stride_output: i64::try_from(spec.output_stride)?,
            batch: i32::try_from(spec.batch)?,
            input_type: spec.input_type.native(),
            output_type: spec.output_type.native(),
        };
        self.inner.bind_to_thread()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: the descriptor lives for the call and output is writable.
        let status = unsafe {
            mircuda_cublas_gemm_create(
                &raw const descriptor,
                stream.inner.cu_stream().cast(),
                &raw mut raw,
            )
        };
        check(status)?;
        Ok(CublasGemmPlan {
            raw: NonNull::new(raw).ok_or(Error::NullAllocation)?,
            stream: stream.inner.clone(),
            spec,
            extents,
        })
    }
}

impl CublasGemmPlan {
    /// `buffers` and `offsets` list the first operand, the second and the
    /// output; offsets count elements.
    pub fn execute(
        &mut self,
        stream: &Stream,
        buffers: [&DeviceBuffer; 3],
        offsets: [usize; 3],
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.stream, &stream.inner) {
            return Err(Error::StreamMismatch);
        }
        let input = self.spec.input_type.bytes();
        let sizes = [input, input, self.spec.output_type.bytes()];
        let mut pointers = [0_u64; 3];
        for index in 0..3 {
            ensure_stream(buffers[index], stream)?;
            let end = offsets[index]
                .checked_add(self.extents[index])
                .and_then(|end| end.checked_mul(sizes[index]));
            if end.is_none_or(|end| end > buffers[index].bytes()) {
                return Err(Error::InvalidMatmulBuffer);
            }
            let offset = u64::try_from(offsets[index] * sizes[index])?;
            pointers[index] = buffers[index].pointer() + offset;
        }
        self.stream.context().bind_to_thread()?;
        // SAFETY: every operand, with its offset and batch stride, was checked
        // to lie inside its buffer, and all buffers share the retained stream.
        check(unsafe {
            mircuda_cublas_gemm_execute(
                self.raw.as_ptr(),
                stream.inner.cu_stream().cast(),
                pointers[0] as *const c_void,
                pointers[1] as *const c_void,
                pointers[2] as *mut c_void,
                alpha,
                beta,
            )
        })
    }
}

impl Drop for CublasGemmPlan {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.context().bind_to_thread());
        // SAFETY: this is the sole owner of the native plan.
        unsafe { mircuda_cublas_gemm_destroy(self.raw.as_ptr()) };
    }
}

const fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Cublas(status))
    }
}
