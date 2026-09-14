use crate::{Driver, Error, Result};

#[test]
fn nested_views_retain_the_allocation_and_address_only_their_range() -> Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let base = pool.allocate_zeroed::<u32>(&stream, 16)?;
    let first = base.slice(4..12)?;
    let mut nested = first.slice(2..6)?;
    assert!(base.slice(4..4).is_err());
    assert!(base.slice(0..17).is_err());
    assert!(base.slice(usize::MAX..usize::MAX).is_err());
    assert!(base.reinterpret::<u8>()?.slice(1..9)?.reinterpret::<u32>().is_err());
    assert_eq!(nested.len(), 4);
    assert_eq!(nested.bytes(), 16);
    let mut host = context.allocate_pinned::<u32>(4)?;
    host.copy_from_slice(&[11, 22, 33, 44])?;
    stream.copy_to_device(&mut host, &mut nested)?;
    let mut all = context.allocate_pinned::<u32>(16)?;
    stream.copy_to_host(&base, &mut all)?;
    assert_eq!(all.to_vec()?, [0, 0, 0, 0, 0, 0, 11, 22, 33, 44, 0, 0, 0, 0, 0, 0]);
    drop(base);
    drop(first);
    let mut copied = pool.allocate::<u32>(&stream, 4)?;
    stream.copy_device_range(&nested, 0..4, &mut copied, 0)?;
    drop(nested);
    stream.copy_to_host(&copied, &mut host)?;
    assert_eq!(host.to_vec()?, [11, 22, 33, 44]);
    Ok(())
}

crate::cuda_export!(ViewProbe = "mircuda_probe"(
    output: &mut crate::DeviceBuffer<f32>, value: f32, length: u32,
));

#[test]
fn a_graph_retains_a_view_after_its_parent_is_dropped() -> Result<()> {
    use crate::{CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel};

    type Resources = (TypedKernel<ViewProbe>, Stream, DeviceBuffer<f32>);
    fn launch((kernel, stream, view): &mut Resources) -> Result<()> {
        kernel.launch(stream, LaunchConfig::for_elements(4, 32)?, (view, 7.0, 4))
    }
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let allocation_stream = context.create_stream()?;
    let execution_stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let compiler = Compiler::new(context.clone())?;
    let module = compiler
        .compile(crate::cuda_kernel_file!("../../kernels/probe.cu"), &CompileOptions::default())?;
    let base = pool.allocate_zeroed::<f32>(&allocation_stream, 16)?;
    let view = base.slice(4..8)?;
    allocation_stream.synchronize()?;
    let mut graph = execution_stream
        .capture((module.kernel::<ViewProbe>()?, execution_stream.clone(), view), launch)?;
    drop(base);
    graph.launch(&execution_stream)?;
    let mut host = context.allocate_pinned::<f32>(4)?;
    execution_stream.copy_to_host(&graph.resources().2, &mut host)?;
    assert!(host.to_vec()?.iter().all(|value| value.to_bits() == 7.0_f32.to_bits()));
    graph.launch(&execution_stream)?;
    drop(graph);
    execution_stream.synchronize()?;
    Ok(())
}
