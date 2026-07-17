#![cfg(target_os = "linux")]

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, Driver, KernelNode, LaunchConfig, Stream, TypedKernel,
    cuda_export, cuda_kernel,
};

cuda_export!(Increment = "increment"(value: &mut DeviceBuffer<u32>, delta: u32));

struct Sequence<'a> {
    kernel: &'a TypedKernel<Increment>,
    stream: &'a Stream,
    value: &'a mut DeviceBuffer<u32>,
    delta: u32,
    node: Option<KernelNode<Increment>>,
}

fn capture_increment(sequence: &mut Sequence<'_>) -> mircuda::Result<()> {
    sequence.node = Some(sequence.kernel.launch_captured(
        sequence.stream,
        LaunchConfig::for_elements(1, 1)?,
        (&mut *sequence.value, sequence.delta),
    )?);
    Ok(())
}

const fn triple<'borrow>(
    sequence: &'borrow mut Sequence<'_>,
) -> (&'borrow mut DeviceBuffer<u32>, u32) {
    sequence.delta = 3;
    (&mut *sequence.value, sequence.delta)
}

#[test]
fn captures_and_replays_on_the_same_stream() -> mircuda::Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let compiler = Compiler::new(context.clone())?;
    let source = cuda_kernel!(
        r#"extern "C" __global__ void increment(unsigned int* value, unsigned int delta) {
            if (threadIdx.x == 0) value[0] += delta;
        }"#
    );
    let module = compiler.compile(source, &CompileOptions::default())?;
    let kernel = module.kernel::<Increment>()?;
    let mut value = pool.allocate_zeroed::<u32>(&stream, 1)?;
    {
        let sequence = Sequence {
            kernel: &kernel,
            stream: &stream,
            value: &mut value,
            delta: 1,
            node: None,
        };
        let mut graph = stream.capture(sequence, capture_increment)?;
        let node = graph.resources().node.ok_or(mircuda::Error::InvalidLaunch)?;
        graph.update_kernel(&node, &kernel, LaunchConfig::for_elements(1, 1)?, triple)?;
        graph.launch(&stream)?;
        graph.launch(&stream)?;
    }
    let mut host = context.allocate_pinned::<u32>(1)?;
    stream.copy_to_host(&value, &mut host)?;
    assert_eq!(host.to_vec()?, [6]);
    Ok(())
}
