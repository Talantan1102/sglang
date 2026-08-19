# DSV4 NPU Compressor v1.9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DeepSeek-V4 Ascend backend dispatch the fused compressor through the `torch.ops.npu` namespace provided by compressor-v1.9.

**Architecture:** Keep the current DSV4 metadata, cache, RoPE, output-trimming, and epilog flow intact. Change only the registered operator namespace, update stale diagnostics/documentation, and protect the ABI dispatch with a focused mocked unit test.

**Tech Stack:** Python, PyTorch custom operators, `unittest`, Ascend NPU backend.

---

### Task 1: Protect the compressor-v1.9 dispatch contract

**Files:**
- Modify: `test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py`

- [ ] **Step 1: Import the backend mixin under test**

Add `CompressorAscendBackendMixin` to the existing import from
`ascend_dsv4_backend`.

- [ ] **Step 2: Write the failing namespace and ABI test**

Add a `TestCompressorV19Dispatch` test case that constructs the mixin without a
hardware backend, stubs metadata and epilog dependencies, patches both operator
namespaces, calls `forward_compress`, and asserts:

```python
mock_npu_compressor.assert_called_once()
mock_custom_compressor.assert_not_called()
args, kwargs = mock_npu_compressor.call_args
self.assertIs(args[0], x)
self.assertIs(args[1], compressor._fused_wkv_w)
self.assertIs(args[2], compressor._fused_wgate_w)
self.assertIs(args[3], state_cache)
self.assertEqual(kwargs["cmp_ratio"], 4)
self.assertEqual(kwargs["coff"], 2)
self.assertEqual(kwargs["rotary_mode"], 2)
self.assertEqual(kwargs["cache_mode"], 2)
```

- [ ] **Step 3: Run the test and verify RED**

Run:

```powershell
$env:PYTHONPATH = 'python'
<bundled-python> test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py
```

Expected: the new test fails because `torch.ops.custom.compressor` is called
and `torch.ops.npu.compressor` is not.

### Task 2: Switch the runtime operator and stale references

**Files:**
- Modify: `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py`
- Modify: `python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py`

- [ ] **Step 1: Change the production dispatch**

Replace:

```python
cmp_kv = torch.ops.custom.compressor(
```

with:

```python
cmp_kv = torch.ops.npu.compressor(
```

Keep every positional and keyword argument unchanged.

- [ ] **Step 2: Update stale operator references**

Replace the two `custom.compressor` references in `dsv4_memory_pool.py` with
`npu.compressor`, preserving the surrounding error and docstring text.

- [ ] **Step 3: Run the focused test and verify GREEN**

Run the same direct `unittest` command from Task 1.

Expected: all tests in the file pass.

- [ ] **Step 4: Run static verification**

Run:

```powershell
<bundled-python> -m py_compile python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py
git grep -n 'custom\.compressor' -- python/sglang/srt/hardware_backend/npu
git diff --check
```

Expected: compilation succeeds, the grep has no matches, and the diff check
reports no whitespace errors.

- [ ] **Step 5: Commit the implementation**

Stage only the three implementation/test files and commit:

```powershell
git add -- python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py
git commit -m "fix(npu): adapt DSV4 compressor to v1.9 namespace"
```
