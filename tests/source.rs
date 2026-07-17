use mircuda::{KernelSource, cuda_kernel, cuda_kernel_file, cuda_kernel_files};

#[test]
fn embeds_inline_source() {
    let source = cuda_kernel!(r#"extern "C" __global__ void probe() {}"#);
    assert_eq!(source.name(), "inline.cu");
    assert!(source.source().contains("probe"));
}

#[test]
fn embeds_file_source() {
    const SOURCE: KernelSource = cuda_kernel_file!("../kernels/probe.cu");
    assert_eq!(SOURCE.name(), "../kernels/probe.cu");
    assert!(SOURCE.source().contains("mircuda_probe"));
}

#[test]
fn composes_embedded_fragments_in_order() {
    const SOURCE: KernelSource = cuda_kernel_files!(
        "composed.cu";
        "../kernels/prefix.cuh",
        "../kernels/probe.cu",
    );
    let source = SOURCE.source();
    assert!(matches!(
        (source.find("MIRCUDA_PREFIX"), source.find("mircuda_probe")),
        (Some(prefix), Some(probe)) if prefix < probe
    ));
}
