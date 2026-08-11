# DSV4 NPU Cache 重构串讲

## 检视范围

- 原始 commit 范围：`1665fd83f76982f88dd1065f4aa643bb8f506616` 到 `a03bc6e9b58cbbfdbe5655917507c4c96661bd1d`
- Squash 分支：`review/dsv4-npu-refactor-squashed`
- Squash commit：`0d5167ecb`
- 本次串讲只关注生产代码，忽略测试、设计文档和 SVG。

## 一、背景

> DSV4 同时维护 Full、SWA、C4、C128 KV 和 compressor state。原来的 NPU 实现为这些数据增加了多套独立的 allocation、location 记录和释放逻辑。
>
> 比如 SWA 在公共/GPU 路径中，可以通过 Full location 和 `translate_loc_from_full_to_swa()` 得到，但 NPU 又额外维护了 `req_to_token_swa`。C4 也类似，额外维护了 `c4_attn_allocator` 和 `req_to_token_c4`。
>
> 这些重复的 location 来源让后续分配、释放和 PD 适配越来越复杂。这个 PR 的目标就是收敛这些重复逻辑，让 NPU 尽量复用现有的公共内存管理和缓存机制。

## 二、整体改动

这次重构可以拆成三个部分。

### 1. 复用 Full 地址

> 第一，复用 Full 地址。

主要涉及：

- SWA 复用 Full → SWA location 映射。
- C4 建立 Full → C4 location 映射。
- 删除不再需要的独立 allocation 和 location 记录逻辑。

#### 1.1 C4 和 SWA

> C4 和 SWA 可以合在一起看：两者都是先分配 Full location，再从 Full location 得到自己的 location。区别只是 SWA 已经有 `translate_loc_from_full_to_swa()`，而 C4 需要先将 page size 改成 32，才能建立 Full → C4 映射。

##### 代码检视顺序

1. `dsv4_memory_pool.py`

   `DSV4NPUTokenToKVPool` 负责创建并持有 NPU DSV4 使用的 SWA、C4、C128 KV pool，以及 compressor state 和 Indexer pool。

   > SWA 的映射已经具备，这里先补 C4 的前置条件。原来 Full KV 的 page size 是 128，C4 虽然每 4 个 token 生成一个 slot，但 NPU kernel 看到的 C4 page size 也是 128，所以 Full page ID 和 C4 page ID 无法直接对应，只能通过 `c4_attn_allocator` 和 `req_to_token_c4` 单独管理。

   - `DSV4NPUTokenToKVPool._make_kv_pool()`

     这个方法负责创建各类 KV sub-pool，并决定 NPU kernel 看到的物理 page 布局。本次将 C4 KV 的 `kernel_page_size` 从全局 128 改成原生 32。

   - `DSV4NPUTokenToKVPool._make_indexer_pool()`

     这个方法负责创建 C4 Indexer 使用的 K/scale pool。本次将 Indexer 的 page size 同样改成 32，使它与 C4 KV 共用相同的 location。

   > C4 和 Indexer 改成 page size 32 以后，一个 Full page 的 128 个 token 正好对应一个 C4 page 的 32 个 slot，Full page ID 和 C4 page ID 就对齐了。接下来再看 allocator 如何利用这个关系，从 Full location 直接生成 C4 location。

2. `dsv4_allocator.py`

   `DSV4NPUTokenToKVPoolAllocator` 负责组织 Full、SWA 和压缩 KV 的 location 分配，并返回 `DSV4OutCacheLoc`。

   - `DSV4NPUTokenToKVPoolAllocator._wrap_full_alloc()`

     这个方法体现了两者共用的主流程：接收 Full allocator 返回的 `out_full_loc`，通过 `translate_loc_from_full_to_swa()` 生成 `out_swa_loc`，再进入 `_alloc_compressed_kv()` 处理 C4/C128。

   - `DSV4NPUTokenToKVPoolAllocator._derive_c4_loc_from_full()`

     这个方法从本轮 `out_full_loc` 中选出完成 4-token 分组的 slot，再通过 `full_loc // 4` 得到对应的 C4 location。

   - `DSV4NPUTokenToKVPoolAllocator._alloc_compressed_kv()`

     这个方法负责组装本轮 C4/C128 location。本次改为调用 `_derive_c4_loc_from_full()` 生成 `out_c4_loc`，不再单独分配 C4 slot。

3. `ascend_dsv4_backend.py`

   > allocator 建立 Full → C4/SWA 映射以后，消费侧的改动可以快速带过。SWA 直接调用现有翻译方法；C4 backend 则把原来读取 `req_to_token_c4` 的地方，替换为读取基础 `req_to_token`，再计算 C4 location。

   - `CompressorAscendBackendMixin._compute_compress_locs()`

     从 `req_to_token` 构造 C4 page table。

   - `CompressorAscendBackendMixin._forward_compress_native()`

     通过 `full_pos` 查询 `req_to_token`，再计算 C4 write location。

   - `C4IndexerAscendBackendMixin.forward_c4_indexer_npu()`

     Indexer 按 C4 page size 32 读取。

   - `_get_kv_indices()`

     允许调用方传入 `page_size`，支持 C4 原生页。

   - `DeepseekV4AscendAttnBackend._forward_compressed()`

     校验 C4 page size 必须等于 `ori_page_size // 4`。

   > 以 `_compute_compress_locs()` 为例，原来是先取 `req_to_token_c4`，再从这张独立表生成 `c4_page_table`。现在改为直接取 `req_to_token[req_pool_64, :n_c_tokens * 4]`，每隔一个 Full page size 取出 Full page ID，作为 `c4_page_table`。
   >
   > `_forward_compress_native()` 和 Indexer 也是同一类替换：不再读 `req_to_token_c4`，而是从 `req_to_token` 取 Full location，再按 Full → C4 的关系翻译。因此这几处可以快速带过。

4. `dsv4_common_hooks.py`

   这个模块负责在公共 allocation 流程和 DSV4 NPU 特有的 location 之间做衔接，同时处理 Spec verify 和 PD 需要的地址信息。

   - `maybe_build_dsv4_verify_bundle()`

     这个方法为一次 target verify 构造当前 draft 区间的 `DSV4OutCacheLoc`。这里可以同时看到两条映射：`out_swa_loc` 通过 `translate_loc_from_full_to_swa(out_full_loc)` 得到，`out_c4_loc` 通过筛选 Full 分组尾 slot 后执行 `// 4` 得到。

##### 旧逻辑清理

> Full → C4/SWA 的分配和读写链路接通后，剩下的都是机械清理：删除 `c4_attn_allocator`、`req_to_token_c4` 和 `req_to_token_swa` 相关的初始化、释放和写回代码，这里快速带过。

- `dsv4_allocator.py`

  以 `DSV4NPUTokenToKVPoolAllocator.__init__()` 为例，删除 C4 独立 allocator：

  ```diff
  - self.c4_attn_allocator = mk(kvcache.c4_size, kvcache.c4_kv_pool)
  ```

  `free()` 和 `clear()` 中对应的 C4 独立释放、清理分支一并删除。

- `dsv4_req_to_token_pool.py` / `dsv4_common_hooks.py`

  以 `DSV4ReqToTokenTablesMixin._init_dsv4_tables()` 为例，删除 C4 独立 table：

  ```diff
  - ("req_to_token_swa", max_context_len),
  - ("req_to_token_c4", max(1, max_context_len // 4)),
  ```

  随后删除 `write_swa()`、`write_c4()`，以及 `maybe_write_dsv4_extend()`、`_write_dsv4_tables()`、`maybe_write_dsv4_decode()` 中对两张表的写回。

用 Full location 的确定性计算替代 C4 独立分配，并让 C4、SWA 都复用基础 req_to_token；最终删除 c4_attn_allocator、req_to_token_c4 和 req_to_token_swa。

### 2. 重构 compressor state

> 第二，重构 compressor state。

#### 2.1 原来的 paged state 逻辑

> 原来 NPU compressor 使用 `cache_mode=1`，`NPUCompressStatePool` 是 paged 布局。state 的物理 page 是由 allocator 动态分配的，无法根据 token position 直接算出来。
>
> 所以 NPU 单独维护了 `c4_state_attn_allocator`、`c128_state_attn_allocator`，以及 `req_to_token_c4_state`、`req_to_token_c128_state`。每轮 forward 先分配 state slot，把 location 写入这两张表，backend 再从表中构造 state page table。

原来的主调用链可以简化为：

```text
allocation.py::_compute_dsv4_state_lens()
    → DSV4NPUTokenToKVPoolAllocator.compute_dsv4_state_lens_*()
    → DSV4NPUTokenToKVPoolAllocator._alloc_state_extend()
    → DSV4OutCacheLoc.out_c4_state_loc / out_c128_state_loc
    → dsv4_common_hooks.py 写入 req_to_token_c4_state / req_to_token_c128_state
    → CompressorAscendBackendMixin._compute_compress_locs()
    → custom.compressor(cache_mode=1)
```

#### 2.2 改成 ring，复用 GPU `CompressStatePool`

> 现在 NPU compressor 支持 `cache_mode=2` 的 ring state。ring 的 ownership 是固定的：C4 attention/Indexer state 跟随 SWA physical page，C128 state 跟随 `req_pool_idx`。因此 state location 可以直接计算，不再需要 allocator 和 `req_to_token_c*_state`。

##### 代码检视顺序

1. `dsv4_memory_pool.py::NPUCompressStatePool`

   NPU 从自己管理 paged state，改为复用 GPU `CompressStatePool` 的 ring buffer 和两种 location 翻译；`NPUCompressStatePool` 只保留 NPU 算子需要的适配。

2. `ascend_dsv4_backend.py`

   - `_build_explicit_state_block_table()`：调用公共 translate 方法计算 `state_loc`，整理成 NPU 算子需要的 table。
   - `CompressorAscendBackendMixin.forward_compress()`：将 compressor 从 `cache_mode=1` 切到 `cache_mode=2`。
   - Graph/MTP 复用同一个 table 构造方法，不再分配 speculative state slot。

3. PD 代码

   ring ownership 对齐后，SWA KV/C4 state 复用 `StateType.SWA`，C128 state 复用 `StateType.C128_STATE`；Ascend 只保留 `AscendStateType.DSV4_C128` KV 特例。

#### 2.3 旧逻辑清理

> 最后删除整条 paged state 链路，检视时只看三类代表性清理：

- `dsv4_allocator.py` / `dsv4_req_to_token_pool.py`：删除两个 state allocator 和两张 `req_to_token_c*_state`。
- `forward_batch_info.py` / `allocation.py` / `dsv4_common_hooks.py`：删除 `DSV4StateLens`、`DSV4OutCacheLoc` 中的 state location，以及 alloc/write/free/evict 链路。
- `ascend_dsv4_backend.py` / PD 代码：删除 state page table、`cache_mode=1` 消费和 Ascend 专用 state payload。

##### 口述总结

> 复用的是 GPU `CompressStatePool` 的 ring 分配、ownership、location 翻译，以及 `StateType.SWA` / `StateType.C128_STATE` 的 PD 逻辑。
>
> 修改的是 NPU 适配层：三维 `state_cache`、`dummy_state_loc`、A3 显式 location table、`cache_mode=2` 和 Graph 原地刷新。
>
> 删除的是整条 paged state 管理：state allocator、`req_to_token_c*_state`、`DSV4StateLens`、state page table，以及对应的 alloc/write/free/evict/PD 分支。

### 3. 补齐 C128 Radix Cache

> 第三，补齐 C128 Radix Cache。

待补充：

- 为什么 C128 location 不能从 Full location 推导。
- 为什么将 `req_to_token_c128` 改为 `req_to_c128_sidecar`。
- C128 page 的引用计数和释放关系。
- `C128SidecarComponent` 的 insert、match 和 evict 生命周期。
- PD prefix reuse 和 EAGLE group 对齐。

## 三、总结

待补充最终总结。
