#[cfg(all(target_os = "linux", feature = "cutlass"))]
#[allow(clippy::print_stdout)]
mod cuda {
    use mircuda::{
        BlockScaledFp4Plan, BlockScaledFp4Spec, BlockScaledFp4VectorPlan, BlockScaledFp4VectorSpec,
        Context, DeviceBuffer, DeviceElement, Driver, MemoryPool, Stream, bf16,
    };

    const WARMUP: usize = 20;
    const CYCLES: usize = 200;
    const CYCLES_F32: f32 = 200.0;

    pub fn run() -> mircuda::Result<()> {
        let driver = Driver::initialize()?;
        let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
        let context = driver.create_context(device)?;
        let stream = context.create_stream()?;
        let pool = context.default_memory_pool()?;
        println!("       shape    GEMM us    GEMV us    ratio");
        for (n, k) in
            [(4_096, 2_816), (2_048, 2_816), (2_816, 4_096), (2_816, 2_048), (4_096, 4_096)]
        {
            let result = measure(&context, &stream, &pool, n, k)?;
            println!(
                "{n:>6}x{k:<6} {:>8.3} {:>10.3} {:>8.3}x",
                result.gemm_us,
                result.gemv_us,
                result.gemm_us / result.gemv_us
            );
        }
        Ok(())
    }

    struct Measurement {
        gemm_us: f32,
        gemv_us: f32,
    }

    fn measure(
        context: &Context,
        stream: &Stream,
        pool: &MemoryPool,
        n: usize,
        k: usize,
    ) -> mircuda::Result<Measurement> {
        let input = copy_device(context, stream, pool, &vec![0x22_u8; k / 2])?;
        let input_scales = copy_device(context, stream, pool, &vec![0x38_u8; scales(1, k)])?;
        let weight = copy_device(context, stream, pool, &vec![0x22_u8; n * k / 2])?;
        let weight_scales = copy_device(context, stream, pool, &vec![0x38_u8; scales(n, k)])?;
        let mut baseline_output = pool.allocate_zeroed::<bf16>(stream, n)?;
        let mut vector_output = pool.allocate_zeroed::<bf16>(stream, n)?;
        let mut baseline =
            BlockScaledFp4Plan::new(context, stream, BlockScaledFp4Spec::new(1, n, k)?)?;
        let mut vector =
            BlockScaledFp4VectorPlan::new(context, stream, BlockScaledFp4VectorSpec::new(n, k)?)?;
        for _ in 0..WARMUP {
            execute_gemm(
                &mut baseline,
                stream,
                &input,
                &input_scales,
                &weight,
                &weight_scales,
                &mut baseline_output,
            )?;
            execute_gemv(
                &mut vector,
                stream,
                &input,
                &input_scales,
                &weight,
                &weight_scales,
                &mut vector_output,
            )?;
        }
        stream.synchronize()?;
        let baseline_us = time(context, stream, || {
            execute_gemm(
                &mut baseline,
                stream,
                &input,
                &input_scales,
                &weight,
                &weight_scales,
                &mut baseline_output,
            )
        })?;
        let vector_us = time(context, stream, || {
            execute_gemv(
                &mut vector,
                stream,
                &input,
                &input_scales,
                &weight,
                &weight_scales,
                &mut vector_output,
            )
        })?;
        Ok(Measurement { gemm_us: baseline_us, gemv_us: vector_us })
    }

    fn time(
        context: &Context,
        stream: &Stream,
        mut execute: impl FnMut() -> mircuda::Result<()>,
    ) -> mircuda::Result<f32> {
        let started = context.create_event(true)?;
        let completed = context.create_event(true)?;
        started.record(stream)?;
        for _ in 0..CYCLES {
            execute()?;
        }
        completed.record(stream)?;
        completed.synchronize()?;
        Ok(started.elapsed_ms(&completed)? * 1_000.0 / CYCLES_F32)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_gemm(
        plan: &mut BlockScaledFp4Plan,
        stream: &Stream,
        input: &DeviceBuffer<u8>,
        input_scales: &DeviceBuffer<u8>,
        weight: &DeviceBuffer<u8>,
        weight_scales: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
    ) -> mircuda::Result<()> {
        plan.execute(stream, input, input_scales, weight, weight_scales, output, 1.0)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_gemv(
        plan: &mut BlockScaledFp4VectorPlan,
        stream: &Stream,
        input: &DeviceBuffer<u8>,
        input_scales: &DeviceBuffer<u8>,
        weight: &DeviceBuffer<u8>,
        weight_scales: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
    ) -> mircuda::Result<()> {
        plan.execute(stream, input, input_scales, weight, weight_scales, output, 1.0)
    }

    const fn scales(rows: usize, columns: usize) -> usize {
        rows.div_ceil(128) * columns / 64 * 512
    }

    fn copy_device<T: DeviceElement>(
        context: &Context,
        stream: &Stream,
        pool: &MemoryPool,
        values: &[T],
    ) -> mircuda::Result<DeviceBuffer<T>> {
        let mut host = context.allocate_pinned::<T>(values.len())?;
        host.copy_from_slice(values)?;
        let mut device = pool.allocate::<T>(stream, values.len())?;
        stream.copy_to_device(&mut host, &mut device)?;
        stream.synchronize()?;
        Ok(device)
    }
}

#[cfg(all(target_os = "linux", feature = "cutlass"))]
fn main() -> mircuda::Result<()> {
    cuda::run()
}

#[cfg(not(all(target_os = "linux", feature = "cutlass")))]
fn main() {}
