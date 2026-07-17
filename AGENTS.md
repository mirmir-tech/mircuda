# Mircuda Agent Notes

This child repository inherits the engineering and repository-boundary rules
from the Workmir `AGENTS.md`. The notes below contain only mircuda-specific
constraints needed when the child is used or released independently.

## Purpose

Mircuda is a standalone, model-agnostic Rust gateway to NVIDIA CUDA. It owns
driver contexts, explicit streams, device memory, compilation, kernel launch,
CUDA Graphs, and reusable CUTLASS plans. It never owns model assembly,
tokenization, sampling, scheduling, or K/V cache policy.

## Hard Rules

- Keep raw CUDA handles and every unsafe operation inside `mircuda-sys`.
- Do not expose cudarc or CUDA C types in the public API.
- Every allocation, transfer, kernel launch, and library call uses an explicit
  `Context` and `Stream`; never introduce an implicit default stream.
- Host transfers and stream or context synchronization must always be explicit.
- Keep tensors and execution plans accelerator-resident across prefill/decode.
- Macros may embed CUDA source, but compilation must produce actionable source
  diagnostics. No unchecked source-launch shortcut may enter the public API.
- `mircuda-sys` may use cudarc privately while mircuda fills missing Driver API,
  memory-pool, graph-capture, CUTLASS, and TMA bindings directly.
- Matrix paths use raw CUTLASS. Do not introduce cuBLASLt ownership or handles.
- Custom kernels belong in `kernels/`; model policy and tensor composition
  belong in clients such as `libmir`.
- Run rustfmt, strict Clippy, tests, and rustdoc before completion.
