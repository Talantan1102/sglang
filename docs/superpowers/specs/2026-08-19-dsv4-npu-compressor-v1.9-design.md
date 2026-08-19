# DSV4 NPU Compressor v1.9 Adaptation Design

## Goal

Adapt SGLang's DeepSeek-V4 Ascend backend to the `compressor-v1.9` wheel from
`unclezhou486/sgl-kernel-npu`. Version 1.9 registers the fused operator as
`torch.ops.npu.compressor`; the current main branch calls the removed
`torch.ops.custom.compressor` namespace.

## Scope

- Change the fused runtime call in
  `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py`
  from `torch.ops.custom.compressor` to `torch.ops.npu.compressor`.
- Preserve the existing argument names, values, ordering, cache mode, rotary
  mode, metadata construction, output trimming, Hadamard transform, and cache
  writeback behavior because the v1.9 operator schema matches those inputs.
- Update stale operator-name references in
  `python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py`.
- Add a focused unit regression that fails if the backend dispatches through
  `torch.ops.custom` and verifies dispatch through `torch.ops.npu` with the
  expected compressor arguments.

## Compatibility Policy

Support `compressor-v1.9` strictly. Do not add a fallback to
`torch.ops.custom.compressor`: the release intentionally removed that custom
namespace, and a fallback could hide an incorrectly installed kernel wheel.
Importing `sgl_kernel_npu` remains responsible for registering the operator.

## Data Flow

The existing backend continues to build `state_block_table`, RoPE sine/cosine
tensors, sequence metadata, and cached weight views. It then invokes
`torch.ops.npu.compressor`, trims padded prefill output to the allocated cache
locations, optionally applies the Hadamard rotation, and writes the result to
the existing DSV4 compressed KV pool. No allocator or tensor-layout change is
part of this migration.

## Error Handling

Keep the current shape and location mismatch checks. Missing or incompatible
v1.9 registration should fail directly at `torch.ops.npu.compressor`, making
the installation problem explicit instead of silently selecting an older ABI.

## Verification

1. Add and run a focused unit test for namespace dispatch and keyword mapping.
2. Run the existing Ascend DSV4 backend unit suite.
3. Run Python syntax compilation and repository formatting/diff checks.
4. On an Ascend DSV4 host with the v1.9 wheel installed, verify operator
   registration and execute an eager warmup before graph capture, as required
   by the kernel's tiling-cache contract.

Local Windows verification cannot prove NPU execution; runtime readiness must
be reported separately from static and CPU-mocked test results.
