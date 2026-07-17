# mircuda

Mircuda is a native, model-agnostic Rust gateway to NVIDIA CUDA. It gives Rust
applications explicit control over devices, contexts, streams, memory,
compilation, kernels, graphs, and vendor libraries without exposing CUDA C or a
third-party wrapper as part of the public API.

Mircuda is not an inference engine. Model loading, tensor graphs, K/V cache
policy, scheduling, tokenization, and sampling belong in clients such as
`libmir`.

## Dependency

```toml
[dependencies]
mircuda = "0.1"
```

Consumers should not depend directly on `mircuda-sys` or `mircuda-macros`.

## Runtime

The runtime discovers devices, retains primary contexts, creates non-blocking
streams, exposes stream-ordered memory pools and pinned host memory, and reports
properties needed for device-specific planning:

```rust,no_run
use mircuda::Driver;

let driver = Driver::initialize()?;
let device = driver.devices()?.into_iter().next().expect("CUDA device");
let context = driver.create_context(device)?;
let stream = context.create_stream()?;
let info = context.device_info()?;

assert!(info.compute_capability.0 >= 12);
assert!(info.multiprocessor_count > 0);
println!("integrated CUDA memory: {}", info.integrated);
stream.synchronize()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

All execution streams are explicit. Mircuda does not create a public default
stream and does not hide host synchronization or memory copies.
`DeviceInfo::integrated` exposes CUDA's neutral integrated-device attribute so
consumers can distinguish unified and discrete memory topologies without
putting inference policy in this gateway.

## Compile and launch

CUDA source can be embedded inline or loaded from a file at compile time.
`Compiler` invokes NVRTC for the current context's compute capability and caches
loaded modules by source, NVRTC version, architecture, and compile options.
`CompilerConfig::cache_directory` additionally persists validated,
NUL-terminated PTX artifacts across processes. Cache reads and writes are
best-effort: corrupt or driver-rejected artifacts are removed and rebuilt, while
`CompileCacheStats` reports memory hits, persistent hits, compilations, and
persistent-cache failures. `cuda_export!`
generates the Rust argument contract for an exported symbol:

```rust,no_run
use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, Driver, LaunchConfig, cuda_export,
    cuda_kernel,
};

let source = cuda_kernel!(r#"
    extern "C" __global__ void fill(float* output, float value) {
        output[threadIdx.x] = value;
    }
"#);
cuda_export!(Fill = "fill"(output: &mut DeviceBuffer<f32>, value: f32));

let driver = Driver::initialize()?;
let device = driver.devices()?.into_iter().next().expect("CUDA device");
let context = driver.create_context(device)?;
let stream = context.create_stream()?;
let pool = context.default_memory_pool()?;
let compiler = Compiler::new(context)?;
let module = compiler.compile(source, &CompileOptions::default())?;
let kernel = module.kernel::<Fill>()?;
let mut output = pool.allocate::<f32>(&stream, 256)?;
kernel.launch(&stream, LaunchConfig::for_elements(256, 256)?, (&mut output, 2.0))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `Compiler::with_include_paths` when consumer-owned CUDA source includes
toolkit or project headers. Include roots participate in the module cache key;
the gateway never discovers them from environment variables implicitly.
Use `Compiler::with_config` to set both include roots and an application-owned
persistent cache directory. Consumers must invalidate that directory after
changing external headers in place; embedded source and the NVRTC version are
invalidated automatically.

File-backed source uses `cuda_kernel_file!("path.cu")`.
`cuda_kernel_files!("unit.cu"; "common.cuh", "kernel.cu")` embeds ordered
fragments as one self-contained NVRTC translation unit. This composition is
appropriate for consumer-owned shared device helpers: the resulting source is
part of the compile-cache key and requires no runtime include path. External
toolkit or generated headers should continue to use explicit compiler include
paths.

Device buffers are typed
and uniquely owned. Transfers require pinned memory and an explicit stream;
`PinnedBuffer::write_prefix` updates a reusable chunk staging allocation, while
`PinnedBuffer::to_vec` is an explicit host synchronization point. CUDA events
provide stream dependencies and device-side elapsed time.
`Stream::copy_device_range` performs checked typed D2D packing on the same
stream without staging through host memory or introducing synchronization.
`Context::start_profiler_range` opens an explicit CUDA Profiler API collection
range. Its `ProfilerRange::stop` method binds the owning context and reports
driver failures, while `Drop` provides a non-panicking cleanup fallback. Tools
such as Nsight Systems can therefore capture only a prepared hot path with
`--capture-range=cudaProfilerApi`.

## Matrix multiplication

Raw CUTLASS is the default matrix backend. `DenseMatmulPlan<Input, Output>`
provides FP16 and BF16 inputs with same-type output by default, optional FP32
output, and FP32 accumulation. It uses a short-`M` Tensor Core tile
for general decode, a CUTLASS-profiler-selected `64x64x64` five-stage tile for
wide one-row projections, a bulk Tensor Core tile for larger batches, and a
CUTLASS SIMT kernel for arbitrary shapes. SM120's collective MMA builder
supports low-bit inputs but not BF16/F16, so dense plans use the compatible
`TensorOp` instruction family with geometry tuned for `SM12x` execution.
`DenseVectorPlan<T>` is the explicit `M=1` alternative for bandwidth-bound
projections. Its native CUDA kernel assigns eight output rows to each warp,
reuses one FP16/BF16 input pair across those rows, accumulates in FP32, and
never reads the previous output when `beta` is zero. Selection between the
matrix and vector plans remains the consuming runtime's responsibility.
`BlockScaledFp4Plan`
executes packed E2M1 operands with tiled E4M3 scales and BF16 output.
`BlockScaledFp4VectorPlan` is the allocation-free SM120 `M=1` counterpart.
Eight warps share each packed activation tile, decode E2M1 pairs with native
PTX conversion, apply ModelOpt-compatible UE4M3 scales, and accumulate across
`K` in FP32. It supports partial final output groups instead of requiring
`N` to be divisible by a Tensor Core tile. The consuming runtime remains
responsible for numerical and full-graph admission of individual geometries.
`BlockwiseFp8VectorPlan` exposes the SM120 CUTLASS contract for
`E4M3[1,K] x E4M3[N,K] -> BF16[N]`. It retains workspace, accepts FP32
activation scales at `1x128` and weight scales at `128x128`, and performs no
allocation or synchronization during execution. Weight preparation and the
meaning of the projection remain responsibilities of the consuming library.
`IndexedGroupedFp4Plan` keeps matrix selection, scale pointers, metadata, and
workspace device-resident for fixed-row grouped execution.
The product count may exceed the matrix-bank size: device indices may repeat,
and metadata setup spans as many CUDA blocks as the fixed plan requires.
`VariableGroupedFp4Plan` instead accepts device-resident row counts, compact
offsets, and matrix indices. It uses CUTLASS's device-driven grouped scheduler,
retains capacity-sized workspace, supports empty groups, and neither reads
metadata nor allocates during execution. Both plans are reusable and bound to
one explicit stream.
`PairedVariableGroupedFp4Plan` submits two independent operand and scale banks
through one grouped scheduler while sharing only device-resident indices, row
counts, and offsets. It neither concatenates banks nor assumes that activation
or epilogue scales match between the two products.

```rust,no_run
use mircuda::{DenseMatmulPlan, DenseMatmulSpec, Driver, bf16};

let driver = Driver::initialize()?;
let device = driver.devices()?.into_iter().next().expect("CUDA device");
let context = driver.create_context(device)?;
let stream = context.create_stream()?;
let pool = context.default_memory_pool()?;
let mut plan = DenseMatmulPlan::<bf16>::new(
    &context,
    &stream,
    DenseMatmulSpec::new(2, 4, 3)?,
)?;
let a = pool.allocate::<bf16>(&stream, 6)?;
let b = pool.allocate::<bf16>(&stream, 12)?;
let mut c = pool.allocate_zeroed::<bf16>(&stream, 8)?;
plan.execute(&stream, &a, &b, &mut c, 1.0, 0.0)?;
# // Use `DenseMatmulPlan::<bf16, f32>` to retain FP32 output.
# Ok::<(), Box<dyn std::error::Error>>(())
```

Plan construction and workspace allocation stay outside the hot path. Mircuda
does not quantize, reorder, select experts, or interpret model weights.

## CUDA graphs

`Stream::capture` turns a stable sequence into an instantiated, pre-uploaded
CUDA graph. Capture receives one explicit resource object and a function pointer
rather than an environment-capturing closure. `Graph<Resources>` owns that
object, so buffers, modules, plans, and any references they contain cannot be
dropped while CUDA still references them.

```rust,no_run
use mircuda::{DenseMatmulPlan, DeviceBuffer, Graph, Result, Stream, bf16};

struct Projection<'a> {
    stream: &'a Stream,
    plan: &'a mut DenseMatmulPlan<bf16>,
    input: &'a DeviceBuffer<bf16>,
    weight: &'a DeviceBuffer<bf16>,
    output: &'a mut DeviceBuffer<bf16>,
}

fn project(resources: &mut Projection<'_>) -> Result<()> {
    resources.plan.execute(
        resources.stream,
        resources.input,
        resources.weight,
        resources.output,
        1.0,
        0.0,
    )
}

fn replay(stream: &Stream, resources: Projection<'_>) -> Result<()> {
    let mut graph: Graph<Projection<'_>> = stream.capture(resources, project)?;
    graph.launch(stream)?;
    graph.launch(stream)
}
```

Replay is asynchronous and valid only on the capture stream. Dynamic decode
values use `Graph::kernel_nodes` and typed `Graph::update_kernel`; the update
closure receives the graph-owned resource object and must return the original
kernel signature. `Graph::try_update_kernel` provides the same typed contract
when preparing arguments can fail and maps errors through `From`. Mircuda does
not silently recapture a graph or synchronize to
update it. `TypedKernel::launch_captured` returns the exact typed node while a
capture is active, avoiding symbol-order assumptions when the same kernel is
launched more than once in one graph. Capture-node discovery accepts branched
CUDA frontiers and selects the unique frontier node matching the launched
function handle.
`Graph::into_resources` destroys the native graph and returns the retained
resource object without synchronizing or copying device memory. Higher layers
use this explicit transition to replace graphs while preserving cache pages,
plans, and allocations.
`Graph::with_resources_mut` instead keeps the native graph alive while lending
its retained allocations to explicitly ordered work such as prefill. The
caller must enqueue that work on a stream ordered with graph replay; the method
does not synchronize or provide implicit stream selection.

## Build requirements

macOS can build and document the portable API, but CUDA execution returns an
unsupported-platform error. Native Linux execution requires:

- an NVIDIA driver exposing `libcuda.so`;
- CUDA Toolkit 13.x with `nvcc` and NVRTC;
- Rust `nightly-2026-07-13` (1.99 nightly);
- a supported NVIDIA GPU. The first production target is GB10, compute 12.1.

On the current GB10 host:

```sh
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
make check
make bootstrap-cutlass
make cutlass-check
cargo run --example inspect
make profile
make matmul-profile
make dense-vector-profile
make graph-profile
```

`make profile` verifies compilation, cache lookup, pool allocation, repeated
typed launches, event timing, pinned transfer, and result correctness on the
active CUDA device.
The `inspect` example reports compute capability, SM count, memory-pool support,
and whether CUDA identifies the device as integrated.
`make cutlass-check` builds pinned CUTLASS 4.4.2 and native vector AOT paths for
the `SM12x` family target (`120f`) and tests dense, vector, single FP4, fixed
grouped FP4, variable-row compact grouped FP4, and blockwise FP8 execution.
`MIRCUDA_CUDA_ARCH` can override that family target for development builds.
The default `cutlass` feature may be disabled only for portable gateway-only
consumers that intentionally do not need matrix execution.
`make matmul-profile` measures CUTLASS BF16/FP16 decode, generic prefill, and
the concrete small-chunk projection geometries used by the CUDA model path.
`make dense-vector-profile` rotates weight banks for the concrete decode
geometries and compares cold-weight Tensor Core plans with the explicit GEMV.
`make graph-profile` compares host enqueue and device time for direct CUTLASS
launches against replay of the same sequence as one CUDA graph.

Model and inference kernels belong to consuming runtimes such as `libmir`.
Mircuda only embeds, compiles, caches, binds, and launches their CUDA source.
