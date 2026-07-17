#[cfg(all(target_os = "linux", feature = "cutlass"))]
#[allow(clippy::print_stdout)]
mod cuda {
    use mircuda::{
        BlockScaledFp4Plan, BlockScaledFp4Spec, Context, DeviceBuffer, DeviceElement, Driver,
        IndexedGroupedFp4Plan, IndexedGroupedFp4Spec, MemoryPool, Stream, bf16,
    };

    const N: usize = 4_224;
    const K: usize = 2_816;
    const WARMUP: usize = 16;
    const CYCLES: usize = 512;
    const CYCLES_F32: f32 = 512.0;

    pub fn run() -> mircuda::Result<()> {
        let driver = Driver::initialize()?;
        let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
        let context = driver.create_context(device)?;
        let stream = context.create_stream()?;
        let pool = context.default_memory_pool()?;
        for rows in [1, 2] {
            profile(&context, &stream, &pool, rows)?;
        }
        Ok(())
    }

    fn profile(
        context: &Context,
        stream: &Stream,
        pool: &MemoryPool,
        rows: usize,
    ) -> mircuda::Result<()> {
        let a = pool.allocate_zeroed::<u8>(stream, rows * K / 2)?;
        let a_scales = pool.allocate_zeroed::<u8>(stream, scales(rows, K))?;
        let b = [
            pool.allocate_zeroed::<u8>(stream, N * K / 2)?,
            pool.allocate_zeroed::<u8>(stream, N * K / 2)?,
        ];
        let b_scales = [
            pool.allocate_zeroed::<u8>(stream, scales(N, K))?,
            pool.allocate_zeroed::<u8>(stream, scales(N, K))?,
        ];
        let bank = pool.allocate_zeroed::<u8>(stream, 2 * N * K / 2)?;
        let bank_scales = pool.allocate_zeroed::<u8>(stream, 2 * scales(N, K))?;
        let alphas = upload(context, stream, pool, &[1.0_f32, 1.0])?;
        let indices = upload(context, stream, pool, &[0_u32, 1])?;
        let mut separate_output =
            [pool.allocate::<bf16>(stream, rows * N)?, pool.allocate::<bf16>(stream, rows * N)?];
        let mut grouped_output = pool.allocate::<bf16>(stream, 2 * rows * N)?;
        let spec = BlockScaledFp4Spec::new(rows, N, K)?;
        let mut separate = [
            BlockScaledFp4Plan::new(context, stream, spec)?,
            BlockScaledFp4Plan::new(context, stream, spec)?,
        ];
        let grouped_spec = IndexedGroupedFp4Spec::new(2, 2, rows, N, K)?.with_broadcast_input();
        let mut grouped = IndexedGroupedFp4Plan::new(context, stream, grouped_spec)?;

        for _ in 0..WARMUP {
            execute_separate(
                stream,
                &mut separate,
                &a,
                &a_scales,
                &b,
                &b_scales,
                &mut separate_output,
            )?;
            grouped.execute(
                stream,
                &a,
                &a_scales,
                &bank,
                &bank_scales,
                &alphas,
                &indices,
                &mut grouped_output,
            )?;
        }
        stream.synchronize()?;

        let separate_us = measure(context, stream, || {
            execute_separate(
                stream,
                &mut separate,
                &a,
                &a_scales,
                &b,
                &b_scales,
                &mut separate_output,
            )
        })?;
        let grouped_us = measure(context, stream, || {
            grouped.execute(
                stream,
                &a,
                &a_scales,
                &bank,
                &bank_scales,
                &alphas,
                &indices,
                &mut grouped_output,
            )
        })?;
        println!(
            "rows={rows} separate={separate_us:.3} us grouped={grouped_us:.3} us speedup={:.3}x",
            separate_us / grouped_us,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_separate(
        stream: &Stream,
        plans: &mut [BlockScaledFp4Plan; 2],
        a: &DeviceBuffer<u8>,
        a_scales: &DeviceBuffer<u8>,
        b: &[DeviceBuffer<u8>; 2],
        b_scales: &[DeviceBuffer<u8>; 2],
        output: &mut [DeviceBuffer<bf16>; 2],
    ) -> mircuda::Result<()> {
        for index in 0..2 {
            plans[index].execute(
                stream,
                a,
                a_scales,
                &b[index],
                &b_scales[index],
                &mut output[index],
                1.0,
            )?;
        }
        Ok(())
    }

    fn measure(
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

    const fn scales(rows: usize, columns: usize) -> usize {
        rows.div_ceil(128) * columns / 64 * 512
    }

    fn upload<T: DeviceElement>(
        context: &Context,
        stream: &Stream,
        pool: &MemoryPool,
        values: &[T],
    ) -> mircuda::Result<DeviceBuffer<T>> {
        let mut host = context.allocate_pinned::<T>(values.len())?;
        host.copy_from_slice(values)?;
        let mut device = pool.allocate::<T>(stream, values.len())?;
        stream.copy_to_device(&mut host, &mut device)?;
        Ok(device)
    }
}

#[cfg(all(target_os = "linux", feature = "cutlass"))]
fn main() -> mircuda::Result<()> {
    cuda::run()
}

#[cfg(not(all(target_os = "linux", feature = "cutlass")))]
fn main() {}
