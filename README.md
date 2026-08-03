# mircuda

Mircuda is a model-agnostic Rust execution layer for NVIDIA CUDA. It provides
the accelerator primitives used by `libmir` while keeping model, generation,
and application policy out of the hardware boundary.

Use mircuda when native Rust code needs explicit control over CUDA execution,
memory, compilation, and reusable plans without adopting a Python runtime or
exposing CUDA C as the public API.

## What it provides

- devices, retained contexts, and explicit non-blocking streams;
- stream-ordered device memory and pinned host memory;
- NVRTC compilation, persistent PTX caching, and typed kernel launches;
- CUDA graphs and reusable execution plans;
- cuBLASLt and CUTLASS-backed matrix multiplication;
- model-neutral low-precision and quantized operations;
- a private native boundary to the CUDA driver and vendor libraries.

## Add it to a project

```toml
[dependencies]
mircuda = "0.3.1"
```

Mircuda currently targets Linux with an NVIDIA driver and CUDA Toolkit 13.x.
The default feature set enables cuBLASLt and CUTLASS. Consumers should depend on
`mircuda`, not its private `mircuda-sys` or `mircuda-macros` implementation
crates.

## Typical uses

- compile and launch typed CUDA kernels from Rust;
- manage explicit contexts, streams, and device memory;
- capture repeatable workloads as CUDA graphs;
- build higher-level inference or numerical libraries over native CUDA.

## Links

- [crates.io](https://crates.io/crates/mircuda)
- [Rust API documentation](https://docs.rs/mircuda)
- [Source and issues](https://github.com/mirmir-tech/mircuda)
- [MiRMiR architecture](https://mirmir.tech/modules)

Licensed under Apache-2.0.
