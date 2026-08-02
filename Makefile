SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

.PHONY: help doctor bootstrap-cutlass bootstrap-flash-attn cutlass-check docs docs-open examples profile matmul-profile dense-vector-profile graph-profile check

CUTLASS_VERSION := v4.4.2
CUTLASS_DIR ?= $(HOME)/.cache/mircuda/cutlass-$(CUTLASS_VERSION)
FLASH_ATTN_REVISION := 2c839c33742309ec41e620bf837495ec9926c56e
FLASH_ATTN_DIR ?= $(HOME)/.cache/mircuda/flash-attention-$(FLASH_ATTN_REVISION)

help: ## Show development targets.
	@awk 'BEGIN {FS = ":.*## "; printf "Mircuda commands:\n\n"} \
		/^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

doctor: ## Verify the native Linux CUDA development toolchain.
	@test "$$(uname -s)" = Linux
	@command -v nvidia-smi >/dev/null
	@command -v nvcc >/dev/null
	@command -v rustc >/dev/null
	@nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
	@nvcc --version | tail -1
	@rustc --version

bootstrap-cutlass: ## Clone the pinned header-only CUTLASS dependency.
	@if test ! -d "$(CUTLASS_DIR)/.git"; then \
		mkdir -p "$(dir $(CUTLASS_DIR))"; \
		git clone --depth 1 --branch "$(CUTLASS_VERSION)" \
			https://github.com/NVIDIA/cutlass.git "$(CUTLASS_DIR)"; \
	fi
	@git -C "$(CUTLASS_DIR)" describe --tags --exact-match

bootstrap-flash-attn: ## Clone the pinned native FlashAttention dependency.
	@if test ! -d "$(FLASH_ATTN_DIR)/.git"; then \
		mkdir -p "$(dir $(FLASH_ATTN_DIR))"; \
		git clone --filter=blob:none \
			https://github.com/vllm-project/flash-attention.git "$(FLASH_ATTN_DIR)"; \
	fi
	@git -C "$(FLASH_ATTN_DIR)" checkout "$(FLASH_ATTN_REVISION)"
	@git -C "$(FLASH_ATTN_DIR)" submodule update --init --depth 1 csrc/cutlass
	@test "$$(git -C "$(FLASH_ATTN_DIR)" rev-parse HEAD)" = "$(FLASH_ATTN_REVISION)"
	@test -f "$(FLASH_ATTN_DIR)/csrc/cutlass/include/cutlass/cutlass.h"

cutlass-check: doctor bootstrap-cutlass bootstrap-flash-attn ## Build and test native CUTLASS attention.
	@cargo test -p mircuda --test cutlass

docs: ## Build rustdoc for the complete public workspace API.
	@RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

docs-open: ## Build and open the mircuda API documentation.
	@RUSTDOCFLAGS="-D warnings" cargo doc -p mircuda --no-deps --open

examples: ## Type-check every public mircuda example.
	@cargo check -p mircuda --examples

profile: doctor ## Run the native allocation, compile, launch, and transfer probe.
	@cargo run --release --example profile

matmul-profile: doctor bootstrap-cutlass ## Profile CUTLASS decode and prefill shapes.
	@cargo run --release --example matmul

dense-vector-profile: doctor bootstrap-cutlass ## Compare cold-weight decode GEMM and GEMV.
	@cargo run --release --example dense-vector

graph-profile: doctor bootstrap-cutlass ## Compare direct launches with CUDA Graph replay.
	@cargo run --release --example graph

check: ## Run formatting, Clippy, tests, and rustdoc.
	@if test "$$(uname -s)" = Linux; then \
		$(MAKE) --no-print-directory bootstrap-cutlass bootstrap-flash-attn; \
	fi
	@cargo fmt --all --check
	@cargo clippy --workspace --all-targets --all-features -- -D warnings
	@cargo test --workspace --all-features
	@$(MAKE) docs
