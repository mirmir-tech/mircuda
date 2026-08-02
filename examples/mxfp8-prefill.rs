use std::io::{self, Write};

use mircuda::{
    CompileOptions, Compiler, Context, DeviceBuffer, Driver, LaunchConfig, MemoryPool, MxFp8Spec,
    MxFp8TensorCore, Stream, TypedKernel, bf16, cuda_export, cuda_kernel_file,
};

const N: usize = 3_072;
const K: usize = 1_024;
const BANKS: usize = 8;
const CYCLES: u16 = 4;
const ROWS: &[usize] = &[1, 2, 3, 4, 8, 16, 32, 64];

cuda_export!(
    Portable = "mircuda_mxfp8_bf16"(
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        tokens: u32,
        input_features: u32,
        output_features: u32,
    )
);

cuda_export!(
    Tiled = "mircuda_mxfp8_bf16_tiled"(
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        tokens: u32,
        input_features: u32,
        output_features: u32,
    )
);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let compiler = Compiler::new(context.clone())?;
    let module =
        compiler.compile(cuda_kernel_file!("../kernels/mxfp8.cu"), &CompileOptions::default())?;
    let portable: TypedKernel<Portable> = module.kernel()?;
    let tiled: TypedKernel<Tiled> = module.kernel()?;
    let weights = banks(&pool, &stream, N * K / 4)?;
    let scales = banks(&pool, &stream, N * K / 32)?;
    let mut stdout = io::stdout().lock();
    for &tokens in ROWS {
        profile(
            &compiler, &context, &stream, &pool, &portable, &tiled, &weights, &scales, tokens,
            &mut stdout,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn profile(
    compiler: &Compiler,
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    portable: &TypedKernel<Portable>,
    tiled: &TypedKernel<Tiled>,
    weights: &[DeviceBuffer<u32>],
    scales: &[DeviceBuffer<u8>],
    tokens: usize,
    stdout: &mut impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = pool.allocate_zeroed::<bf16>(stream, tokens * K)?;
    let mut output = pool.allocate_zeroed::<bf16>(stream, tokens * N)?;
    let portable_us = measure(context, stream, weights, scales, |weight, scale| {
        portable.launch(
            stream,
            config(tokens)?,
            (
                &input,
                weight,
                scale,
                &mut output,
                u32::try_from(tokens)?,
                u32::try_from(K)?,
                u32::try_from(N)?,
            ),
        )
    })?;
    let tiled_us = measure(context, stream, weights, scales, |weight, scale| {
        tiled.launch(
            stream,
            config(tokens.div_ceil(4))?,
            (
                &input,
                weight,
                scale,
                &mut output,
                u32::try_from(tokens)?,
                u32::try_from(K)?,
                u32::try_from(N)?,
            ),
        )
    })?;
    let tensor_core_us = if tokens >= 2 {
        let operation =
            MxFp8TensorCore::new(compiler, context, pool, stream, MxFp8Spec::new(tokens, K, N)?)?;
        let swizzled = scales
            .iter()
            .map(|scale| operation.swizzle_weight_scales(pool, stream, scale))
            .collect::<mircuda::Result<Vec<_>>>()?;
        let elapsed = measure(context, stream, weights, &swizzled, |weight, scale| {
            operation.execute(stream, &input, weight, scale, &mut output)
        })?;
        Some((elapsed, operation.workspace_bytes()))
    } else {
        None
    };
    let tensor_core = tensor_core_us.map_or_else(
        || "tensor_core=n/a".to_owned(),
        |(elapsed, workspace)| {
            format!(
                "tensor_core={elapsed:.3} us tiled_to_tensor_core={:.3}x workspace={workspace}",
                tiled_us / elapsed
            )
        },
    );
    writeln!(
        stdout,
        "m={tokens} n={N} k={K} portable={portable_us:.3} us tiled={tiled_us:.3} us portable_to_tiled={:.3}x {tensor_core}",
        portable_us / tiled_us,
    )?;
    Ok(())
}

fn measure(
    context: &Context,
    stream: &Stream,
    weights: &[DeviceBuffer<u32>],
    scales: &[DeviceBuffer<u8>],
    mut execute: impl FnMut(&DeviceBuffer<u32>, &DeviceBuffer<u8>) -> mircuda::Result<()>,
) -> mircuda::Result<f32> {
    for (weight, scale) in weights.iter().zip(scales) {
        execute(weight, scale)?;
    }
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..CYCLES {
        for (weight, scale) in weights.iter().zip(scales) {
            execute(weight, scale)?;
        }
    }
    completed.record(stream)?;
    completed.synchronize()?;
    let operations = f32::from(CYCLES) * f32::from(u16::try_from(weights.len())?);
    Ok(started.elapsed_ms(&completed)? * 1_000.0 / operations)
}

fn config(grid_y: usize) -> mircuda::Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (u32::try_from(N)?, u32::try_from(grid_y)?, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}

fn banks<T: mircuda::DeviceElement>(
    pool: &MemoryPool,
    stream: &Stream,
    elements: usize,
) -> mircuda::Result<Vec<DeviceBuffer<T>>> {
    (0..BANKS).map(|_| pool.allocate_zeroed::<T>(stream, elements)).collect()
}
