# Vendored Marlin subset

The files under `vendor/` are the Apache-2.0 Marlin implementation carried by
vLLM and adapted here to a Torch-free native boundary. The upstream source is
`vllm/csrc/libtorch_stable/{quantization/marlin,moe/marlin_moe_wna16}` in
<https://github.com/vllm-project/vllm>.

Only the BF16 activation, E2M1 weight, E4M3 scale specialization needed by the
NVFP4 weight-only MoE path is instantiated. Runtime code does not depend on
vLLM, Torch, Python, or the Workmir `research` directory.
