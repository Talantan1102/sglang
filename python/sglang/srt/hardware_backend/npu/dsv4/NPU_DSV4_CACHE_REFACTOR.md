# NPU DSV4 Cache 地址空间重构方案

## 1. 方案总结

本次重构包含两项改动：

1. **复用 full 地址映射**：C4 和 Indexer 的 page size 从 128 改为 32，共同复用 full page id 和 block table；SWA 复用现有 `full_to_swa_index_mapping`。删除 C4 独立 allocator、`req_to_token_c4` 和 `req_to_token_swa`。
2. **NPU fused compressor 支持 ring**，C4/C128 及 Indexer 的 compressor state 改为固定 ring buffer，不再需要 state allocator 和两张 state request table。

改动二按四个可独立 review 的子改动实施：前三项在建立 ring storage、Eager、Graph/MTP 新路径时同步删除各自替代的 paged state 逻辑，第四项在 GPU 风格 `state_loc` 已经生效后完成 PD/StateType 收敛。删除不是单独阶段。

> 当前代码索引：[六类 allocator](dsv4_allocator.py#L122-L188) · [五张辅助 request table](dsv4_req_to_token_pool.py#L43-L101) · [GPU C4 native page 布局](../../../mem_cache/deepseek_v4_memory_pool.py#L562-L645) · [GPU C4 地址派生](../../../../kernels/ops/attention/dsv4/metadata_kernel.py#L34-L50)

### SVG 矢量图索引

以下文件可单独打开并无限放大，原始 Mermaid 图仍保留在正文中：

- C4 地址链路：[重构前](npu_dsv4_cache_refactor_svgs/01-c4-address-before.svg) · [重构后](npu_dsv4_cache_refactor_svgs/02-c4-address-after.svg)
- Indexer 链路：[重构前](npu_dsv4_cache_refactor_svgs/03-indexer-before.svg) · [重构后](npu_dsv4_cache_refactor_svgs/04-indexer-after.svg)
- SWA request table：[重构前](npu_dsv4_cache_refactor_svgs/05-swa-table-before.svg) · [重构后](npu_dsv4_cache_refactor_svgs/06-swa-table-after.svg)
- Compressor state：[重构前](npu_dsv4_cache_refactor_svgs/07-compressor-state-before.svg) · [Ring 重构后](npu_dsv4_cache_refactor_svgs/08-compressor-state-ring-after.svg)
- 改动二实施拆分：[四个子改动](npu_dsv4_cache_refactor_svgs/11-compressor-ring-four-parts.svg)
- 子改动 1 · Ring pool：[重构前](npu_dsv4_cache_refactor_svgs/12-p1-ring-pool-before.svg) · [重构后](npu_dsv4_cache_refactor_svgs/13-p1-ring-pool-after.svg)
- 子改动 2 · Eager Compressor：[重构前](npu_dsv4_cache_refactor_svgs/14-p2-eager-before.svg) · [重构后](npu_dsv4_cache_refactor_svgs/15-p2-eager-after.svg)
- 子改动 3 · Graph/MTP：[重构前](npu_dsv4_cache_refactor_svgs/16-p3-graph-mtp-before.svg) · [重构后](npu_dsv4_cache_refactor_svgs/17-p3-graph-mtp-after.svg)
- 子改动 4 · PD/StateType 收敛：[重构前](npu_dsv4_cache_refactor_svgs/18-p4-pd-ring-before.svg) · [重构后](npu_dsv4_cache_refactor_svgs/19-p4-pd-ring-after.svg)
- PD 地址与传输链路：[改动一前](npu_dsv4_cache_refactor_svgs/09-pd-before-change-one.svg) · [改动一后](npu_dsv4_cache_refactor_svgs/10-pd-after-change-one.svg)


## 2. 改动一：复用 full 地址映射

### 2.1 C4 复用 full 地址空间

full page 包含 128 个 raw token，C4 每 4 个 raw token 生成一个压缩 token，因此一个 full page 刚好对应一个 32-token C4 page。

目标修改：

- C4 attention KV pool 的 physical page size 改为 32。
- C4 page id 直接使用 full page id，C4 attention 复用 full block table。
- C4 写入地址由 full loc 直接派生。
- 删除 C4 allocator、`req_to_token_c4` 和独立 C4 page-table 构造逻辑。
- C4 KV buffer 仍然保留，变化的是它的寻址和生命周期管理方式。

这项修改依赖 `npu_sparse_attn_sharedkv` 能同时使用 ori page 128 和 C4 page 32。

代码框架对比：

这条链路有两路输入和两个最终结果：

- **写地址输入**：[`DSV4NPUTokenToKVPoolAllocator.alloc_extend()` / `alloc_decode()`](dsv4_allocator.py#L573-L670) 调用父类完成 full 分配，得到本轮 `out_full_loc`；重构后统一在 `_wrap_full_alloc()` 接入 C4 地址派生。
- **读地址输入**：[`ReqToTokenPool.req_to_token`](../../../mem_cache/memory_pool.py#L250-L314) 已保存 full KV 的历史地址，由 attention backend 读取。它不是 `_wrap_full_alloc()` 的返回值。
- **结果一（写地址）**：得到 `out_c4_loc`，告诉 compressor 本轮新生成的 C4 KV 写到哪里。
- **结果二（读地址）**：得到 C4 page table，告诉 SparseAttn 到哪些 page 读取历史 C4 KV。

颜色说明：蓝色表示复用，绿色表示新增，橙色表示修改，红色表示删除；灰色是业务处理，紫色是最终结果。

重构前，两个结果都依赖 C4 独立地址管理：

```mermaid
flowchart LR
    Entry["分配入口<br/>DSV4NPUTokenToKVPoolAllocator.alloc_extend() / alloc_decode()"]:::neutral
    FullAlloc["父类完成 full 分配<br/>SWATokenToKVPoolAllocator.alloc_extend() / alloc_decode()"]:::reused
    OldJoin["进入 DSV4 C4 分配<br/>DSV4NPUTokenToKVPoolAllocator._alloc_c_and_state()"]:::changed
    Count["计算本轮完成的 4-token 分组数<br/>DSV4NPUTokenToKVPoolAllocator._compute_c_extend_counts()"]:::reused
    History["查询历史 C4 尾地址<br/>DSV4ReqToTokenTablesMixin.req_to_token_c4<br/>用于从当前 C4 page/offset 继续分配"]:::deleted
    Alloc["独立分配 C4 slot<br/>DSV4NPUTokenToKVPoolAllocator.c4_attn_allocator"]:::deleted
    WriteLoc["结果一：out_c4_loc<br/>DSV4OutCacheLoc.out_c4_loc"]:::reused
    Compressor["compressor 生成并写入本轮 C4 KV"]:::neutral
    Pool["C4 KV pool<br/>NPUDeepSeekV4SingleKVPool，page=128"]:::changed
    WriteTable["记录新地址<br/>DSV4ReqToTokenTablesMixin.write_c4()"]:::deleted
    Table["维护历史地址<br/>req_to_token_c4"]:::deleted
    BuildTable["结果二：构造独立 C4 page table<br/>CompressorAscendBackendMixin._compute_compress_locs()"]:::deleted
    Attention["最终：SparseAttn 能读取完整 C4 历史"]:::result

    Entry --> FullAlloc --> OldJoin --> Count --> Alloc --> WriteLoc
    History --> Alloc
    WriteLoc --> Compressor --> Pool --> Attention
    WriteLoc --> WriteTable --> Table --> BuildTable --> Attention

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef deleted fill:#FDECEC,stroke:#D14343,color:#7F1D1D,stroke-width:2px,stroke-dasharray:5 3;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

`alloc_extend()` 用于 prefill/chunk 一次新增多个 raw token，`alloc_decode()` 用于 decode 时每个 request 新增一个 raw token。它们只负责预留地址，不负责计算或写入 KV；一次调用会先通过父类分配 full/SWA，再通过 `_alloc_c_and_state()` 分配 C4、C128 和 compressor state，最后返回一个 `DSV4OutCacheLoc` 地址集合。

以一个 request 从 raw 长度 6 扩展到 10 为例：

1. 父类 full allocator 根据 full 尾地址分配 4 个 raw slot。假设当前 full page id 是 5，则 `out_full_loc=[646, 647, 648, 649]`。
2. C4 长度从 `6//4=1` 增长到 `10//4=2`，说明本轮只新生成 1 个 C4 KV（raw position 7 完成了 4～7 这一组）。
3. C4 使用独立地址空间。假设 `req_to_token_c4[req, 0]=256`，表示上一个 C4 KV 位于 C4 page 2、offset 0。
4. `_alloc_c_extend()` 将 256 作为 C4 尾地址交给 `c4_attn_allocator`，allocator 从同一 C4 page 继续分配，得到 `out_c4_loc=[257]`。
5. compressor 将本轮 C4 KV 写到 slot 257，`write_c4()` 再把 257 写入 `req_to_token_c4[req, 1]`。attention backend 后续通过这张表构造 C4 page table。

这里的两个地址对象作用不同：

| 地址对象 | 生命周期 | 作用 |
| --- | --- | --- |
| `out_c4_loc` | 仅本次 forward | 按 batch 顺序保存“本轮新增 C4 KV”的物理写入 slot，供 compressor 写入 |
| `req_to_token_c4` | 整个 request | 保存“request 的第几个 C4 KV→物理 slot”的完整历史映射，供后续分配和 attention 查询 |

因此，得到 `out_c4_loc=[257]` 后还必须记录新地址：一方面，下次再完成一个 C4 分组时，需要从 `req_to_token_c4[req, 1]=257` 找到尾地址并继续分配；另一方面，SparseAttn 需要根据完整的 `[256, 257, ...]` 历史地址构造 C4 page table。仅保留本轮的 `out_c4_loc` 无法完成这两件事。

查询 C4 尾地址的目的，就是找到独立 C4 地址空间中“上一次写到哪里”。paged allocator 需要它判断下一个 slot 应继续使用当前 page，还是申请新 page。由于当前 C4 与 full 地址无关，这个尾地址不能从 full 尾地址推导，只能查询 `req_to_token_c4`。

重构后，写地址从 full loc 派生，读地址直接复用 full block table：

```mermaid
flowchart LR
    Entry["分配入口<br/>DSV4NPUTokenToKVPoolAllocator.alloc_extend() / alloc_decode()"]:::neutral
    FullAlloc["复用父类 full 分配<br/>SWATokenToKVPoolAllocator.alloc_extend() / alloc_decode()"]:::reused
    Start["写地址链路起点：接收 out_full_loc<br/>DSV4NPUTokenToKVPoolAllocator._wrap_full_alloc()"]:::changed
    Derive["派生本轮 C4 写地址<br/>DSV4NPUTokenToKVPoolAllocator._derive_c4_loc_from_full()<br/>① 筛选完成 4-token 分组的 full loc　② full loc // 4"]:::added
    WriteLoc["结果一：out_c4_loc<br/>DSV4OutCacheLoc.out_c4_loc"]:::reused
    Compressor["compressor 生成并写入本轮 C4 KV"]:::neutral
    Pool["C4 KV pool<br/>NPUDeepSeekV4SingleKVPool，page=32"]:::changed
    BaseTable["复用完整历史地址<br/>ReqToTokenPool.req_to_token"]:::reused
    ReadTable["结果二：full block table 直接作为 C4 page table<br/>CompressorAscendBackendMixin._compute_compress_locs()"]:::changed
    Attention["最终：SparseAttn 能读取完整 C4 历史"]:::result

    Entry --> FullAlloc --> Start --> Derive --> WriteLoc
    BaseTable --> ReadTable
    WriteLoc --> Compressor --> Pool --> Attention
    ReadTable --> Attention

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef added fill:#E8F7EE,stroke:#22A06B,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

`DSV4OutCacheLoc.out_c4_loc` 是一次 forward 内新生成的 C4 KV 的写入位置，不是长期保存地址的 request table。重构前它由 `c4_attn_allocator` 分配；重构后保留这个字段供 compressor 使用，但字段值改由 `_derive_c4_loc_from_full()` 生成。

该转换方法从 `out_full_loc` 中选出“刚好完成一个 4-token 分组”的位置，再执行 `full loc // 4`。例如 full page 7 的 raw offset 3、7 对应 full loc 899、903，转换后得到 C4 loc 224、225，也就是 C4 page 7 的 offset 0、1。这样既保持 full/C4 page id 一致，也不再需要 C4 allocator。

SparseAttn “复用 full”指复用 full block table 中的物理 page id，不是读取 full KV buffer。假设一个 request 有 300 个 raw token，full page size=128，三个逻辑页实际分配到物理 page `[5, 9, 2]`：

| 逻辑范围 | full 物理页 | 对应 C4 范围 | C4 物理页 |
| --- | --- | --- | --- |
| raw 0～127 | 5 | C4 0～31 | 5 |
| raw 128～255 | 9 | C4 32～63 | 9 |
| raw 256～299 | 2 | C4 64～74 | 2 |

因此 full block table `[5, 9, 2]` 可以直接作为 C4 block table。假设 Indexer 选中 C4 logical index 35，SparseAttn 按 C4 page size 32 计算：page-table column=`35//32=1`，physical page=`[5,9,2][1]=9`，offset=`35%32=3`，最终从独立 C4 KV buffer 的 slot `9*32+3=291` 读取。最后一页只有 11 个有效 C4 token，由 C4 sequence length 限制读取范围。

代码处理集中在现有类中：[`_wrap_full_alloc()`](dsv4_allocator.py#L608-L670) 接入 C4 地址派生；[`_alloc_c_extend()` / `_alloc_c_and_state()`](dsv4_allocator.py#L275-L367) 移除 C4 分配分支、仅保留 C128；[`DSV4NPUTokenToKVPool._make_kv_pool()`](dsv4_memory_pool.py#L286-L315) 使用子 pool 的实际 page size；[`_compute_compress_locs()`](../attention/ascend_dsv4_backend.py#L242-L337) 改为复用 full block table。不新增 pool 或 allocator 类。

> 当前代码索引：[C4 独立分配](dsv4_allocator.py#L196-L326) · [C4 request table](dsv4_req_to_token_pool.py#L65-L98) · [C4 page table 构造](../attention/ascend_dsv4_backend.py#L323-L337) · [SparseAttn 的 ori/cmp page 限制](../attention/ascend_dsv4_backend.py#L1695-L1756)

### 2.2 Indexer 保留独立 pool，但与 C4 共享地址

Indexer 是 C4 历史的“搜索目录”：它使用独立的 index key 和 scale 为当前 query 选出 top-k C4 位置，SparseAttn 再只读取这些位置的 C4 KV。

Indexer 与 C4 attention 保留两个独立物理 pool，因为两者保存的数据和格式不同：

| Pool | 作用 | 目标管理方式 |
| --- | --- | --- |
| C4 KV pool | 保存最终参与 attention 的压缩 KV | 独立 buffer，page=32，复用 full page id |
| C4 Indexer pool | 保存用于 top-k 检索的 index K/scale | 独立 buffer，page=32，复用同一 full page id |

两个 pool 使用同一个 C4 loc 和 block table，所以 Indexer **不需要独立 allocator，也不需要独立 request table**。

`npu_quant_lightning_indexer` 已确认支持 page size 32：PA_BSND 布局的 block size 支持 16 的整数倍。因此 Indexer 可与 C4 一起切换到 full 映射模型。

代码框架对比：

Indexer pool 保存的是每个历史 C4 token 的搜索特征：量化后的 Index K（int8）和对应 scale（fp16）。它不保存 query，也不保存 attention 使用的 C4 KV。当前 query 是查询阶段的临时输入，用来和历史 Index K/scale 打分。

需要区分“生成 Index K”和“查询 Index K”：生成新的 Index K/scale 时只使用本轮 token 数据，并通过 `out_c4_loc` 写入 Indexer pool，不需要 block table；只有后续 query 查询历史 Index K/scale 时，才使用 block table 定位 Indexer pool 中的历史条目。

这条链路分为两个阶段，有两路输入和一个最终结果：

- **写入阶段输入**：每完成一组 4 个 raw token，生成一份 C4 attention KV 和一份 Index K/scale，并用同一个 `out_c4_loc` 写入两个独立 pool。
- **查询阶段输入**：当前 query 和历史 C4 block table。query 不写入 Indexer pool，只用于检索历史 Index K/scale。
- **最终结果**：Indexer 返回 top-k C4 logical index，SparseAttn 据此读取对应的历史 C4 KV。

颜色说明：蓝色表示复用，绿色表示新增，橙色表示修改，红色表示删除；灰色是业务输入，紫色是最终结果。

重构前，两个阶段都使用 C4 独立地址体系：

```mermaid
flowchart LR
    subgraph WriteBefore["阶段一：写入历史缓存"]
        direction TB
        Raw["输入：完成一组 4 个 raw token"]:::neutral
        C4Build["生成 C4 attention KV<br/>CompressorAscendBackendMixin.forward_compress()"]:::reused
        IndexBuild["生成 Index K(int8) + scale(fp16)<br/>C4IndexerAscendBackendMixin.forward_c4_indexer_npu()"]:::reused
        OldLoc["[删除] 独立分配 out_c4_loc<br/>DSV4NPUTokenToKVPoolAllocator.c4_attn_allocator"]:::deleted
        C4Pool["[修改] C4 KV pool<br/>NPUDeepSeekV4SingleKVPool，page=128"]:::changed
        IndexPool["[修改] Indexer pool<br/>NPUDeepSeekV4IndexerPool，page=128"]:::changed

        Raw --> C4Build --> C4Pool
        Raw --> IndexBuild --> IndexPool
        OldLoc --> C4Pool
        OldLoc --> IndexPool
    end

    subgraph SearchBefore["阶段二：查询历史"]
        direction TB
        Query["输入：当前 query（临时）"]:::neutral
        OldTable["[删除] 独立 C4 block table<br/>由 DSV4ReqToTokenTablesMixin.req_to_token_c4 构造"]:::deleted
        Indexer["[复用] 查询历史 Index K/scale<br/>C4IndexerAscendBackendMixin._forward_npu_fused()"]:::reused
        TopK["top-k C4 logical index"]:::reused
        Sparse["[复用] 按 top-k 读取历史 C4 KV<br/>DeepseekV4AscendAttnBackend._forward_compressed()"]:::reused
        Result["最终：选中的 C4 压缩历史参与 attention"]:::result

        Query --> Indexer --> TopK --> Sparse --> Result
        OldTable --> Indexer
        OldTable --> Sparse
    end

    IndexPool --> Indexer
    C4Pool --> Sparse

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef deleted fill:#FDECEC,stroke:#D14343,color:#7F1D1D,stroke-width:2px,stroke-dasharray:5 3;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

重构后，两个阶段继续共享 C4 地址，但地址来源改为 2.1 的 full 映射：

```mermaid
flowchart LR
    subgraph WriteAfter["阶段一：写入历史缓存"]
        direction TB
        Raw["输入：完成一组 4 个 raw token"]:::neutral
        C4Build["[复用] 生成 C4 attention KV<br/>CompressorAscendBackendMixin.forward_compress()"]:::reused
        IndexBuild["[复用] 生成 Index K(int8) + scale(fp16)<br/>C4IndexerAscendBackendMixin.forward_c4_indexer_npu()"]:::reused
        NewLoc["[新增] full loc 派生 out_c4_loc<br/>DSV4NPUTokenToKVPoolAllocator._derive_c4_loc_from_full()"]:::added
        C4Pool["[修改] C4 KV pool<br/>NPUDeepSeekV4SingleKVPool，page=32"]:::changed
        IndexPool["[修改] Indexer pool<br/>NPUDeepSeekV4IndexerPool，page=32"]:::changed

        Raw --> C4Build --> C4Pool
        Raw --> IndexBuild --> IndexPool
        NewLoc --> C4Pool
        NewLoc --> IndexPool
    end

    subgraph SearchAfter["阶段二：查询历史"]
        direction TB
        Query["输入：当前 query（临时）"]:::neutral
        NewTable["[复用] full block table<br/>由 ReqToTokenPool.req_to_token 构造"]:::reused
        Indexer["[复用] 查询历史 Index K/scale<br/>C4IndexerAscendBackendMixin._forward_npu_fused()"]:::reused
        TopK["top-k C4 logical index"]:::reused
        Sparse["[复用] 按 top-k 读取历史 C4 KV<br/>DeepseekV4AscendAttnBackend._forward_compressed()"]:::reused
        Result["最终：选中的 C4 压缩历史参与 attention"]:::result

        Query --> Indexer --> TopK --> Sparse --> Result
        NewTable --> Indexer
        NewTable --> Sparse
    end

    IndexPool --> Indexer
    C4Pool --> Sparse

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef added fill:#E8F7EE,stroke:#22A06B,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

查询历史时复用 full 的具体方式是：backend 将 full block table 作为 `npu_quant_lightning_indexer` 的 `block_table` 参数，但算子读取的是独立 Indexer pool。假设 request 有 300 个 raw token，full block table=`[5,9,2]`，则共有 `300//4=75` 个历史 C4 Index 条目。查询 C4 logical index 35 时，算子计算 logical page=`35//32=1`、offset=`35%32=3`，从 full block table 得到 physical page id 9，最终读取 Indexer pool 的 slot `9*32+3=291` 中的 Index K/scale。所有有效历史条目完成打分后，算子返回 top-k logical index；SparseAttn 再用同一 block table 到 C4 KV pool 读取这些 index 对应的 C4 KV。整个过程不读取 full KV buffer，复用的只是 full block table 中的 physical page id。

有效 Index 数量不由 block table 长度决定，因为 block table 可能包含 padding；它只负责 logical page→physical page 映射。每个 request 的有效 C4 Index 数量由实际序列长度计算：`num_c4=seq_len//4`。NPU fused 路径向 `npu_quant_lightning_indexer` 传入 `actual_seq_lengths_key` 和 `cmp_ratio=4`，算子据此限制打分范围；fallback 路径也直接使用 `seq_i//ratio`。prefill 中每个 query 还会叠加 causal 边界，只能看到该 query 位置之前已经完成的 C4 分组。

只修改 [`_make_indexer_pool()`](dsv4_memory_pool.py#L369-L393) 的 kernel page size，并在初始化时校验它与 C4 page size 一致；[`NPUDeepSeekV4IndexerPool`](dsv4_memory_pool.py#L195-L270) 及其读写方法、[`forward_c4_indexer_npu()`](../attention/ascend_dsv4_backend.py#L841-L950) 均复用。不删除 Indexer pool，也不新增 allocator、request table 或转换方法。

> 当前代码索引：[NPU Indexer pool](dsv4_memory_pool.py#L195-L272) · [GPU 中独立的 C4/Indexer pool](../../../mem_cache/deepseek_v4_memory_pool.py#L612-L645) · [NPU Indexer top-k](../attention/ascend_dsv4_backend.py#L841-L950) · [`npu_quant_lightning_indexer`](../attention/ascend_dsv4_backend.py#L988-L1021) · [top-k 传入 SparseAttn](../attention/ascend_dsv4_backend.py#L1738-L1756)

### 2.3 SWA request table 复用 full→SWA 映射

SWA 和 full 是两个独立分配和回收的地址空间，因此 SWA allocator 必须保留。但通用 allocator 已经维护 `full_to_swa_index_mapping`，GPU 和 NPU 通用 attention 路径都会通过它将 full loc 翻译为 SWA loc。

因此 NPU DSV4 不再需要将相同结果按 request/context 复制到 `req_to_token_swa`：

```text
base req_to_token → full loc → full_to_swa_index_mapping → SWA loc
```

目标修改：

- 保留 SWA allocator 和 `full_to_swa_index_mapping`。
- 删除 `req_to_token_swa` 及其写入逻辑。
- 通用 SWA attention 继续使用现有 full→SWA 映射，不改变计算逻辑。
- 原来读取 `req_to_token_swa` 的 DSV4 消费路径改为：从 base `req_to_token` 取出所需 full loc，再调用现有 `translate_loc_from_full_to_swa()` 翻译。
- SWA 页淘汰时由 allocator 同步清理映射，保证已释放页不再可达。

代码框架对比：

这项修改只调整 SWA 历史地址的保存和读取方式：本轮 `out_swa_loc` 的生成、SWA KV 写入、SWA allocator 和 mapping 均保持不变。重构前后统一以“DSV4 consumer 需要定位某个 request 的历史 SWA 地址”为入口，以“得到历史 SWA loc / page table”为结果。

颜色说明：蓝色表示复用，绿色表示新增，橙色表示修改，红色表示删除；灰色是业务输入，紫色是最终结果。

重构前，DSV4 consumer 直接从独立 request table 读取历史 SWA 地址：

```mermaid
flowchart LR
    Entry["业务入口：定位 request 的历史 SWA 地址<br/>DSV4 SWA 地址消费路径"]:::neutral
    SWATable["[删除] 读取独立 SWA 历史表<br/>DSV4ReqToTokenTablesMixin.req_to_token_swa<br/>由 write_swa() 维护"]:::deleted
    Result["结果：历史 SWA loc / page table"]:::result

    Entry --> SWATable --> Result

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef deleted fill:#FDECEC,stroke:#D14343,color:#7F1D1D,stroke-width:2px,stroke-dasharray:5 3;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

重构后保持相同业务入口和结果，只将地址来源替换为 base `req_to_token` 和现有 mapping：

```mermaid
flowchart LR
    Entry["业务入口：定位 request 的历史 SWA 地址<br/>DSV4 SWA 地址消费路径"]:::neutral
    HistoryFull["[修改] consumer 改读完整 full loc 历史<br/>ReqToTokenPool.req_to_token"]:::changed
    Translate["[复用] 现有地址翻译接口<br/>translate_loc_from_full_to_swa()"]:::reused
    Mapping["[复用] 接口内部查询地址关系<br/>full_to_swa_index_mapping"]:::reused
    Result["结果：历史 SWA loc / page table"]:::result

    Entry --> HistoryFull --> Translate --> Mapping --> Result

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

`translate_loc_from_full_to_swa()` 本身不修改：它已经支持输入任意形状的 full slot tensor，并返回相同形状的 SWA slot tensor。需要修改的是 DSV4 consumer：先按 request 和有效长度从 base `req_to_token` 截取 full loc，再调用该接口；如果算子需要 page table，则由 consumer 从 SWA slot 换算 SWA page id。

[`DSV4ReqToTokenTablesMixin._init_dsv4_tables()`](dsv4_req_to_token_pool.py#L53-L86) 删除 SWA 表，[`_write_dsv4_tables()`](dsv4_common_hooks.py#L242-L297) 删除对应写入逻辑；原有 SWA 地址消费路径复用 [`SWATokenToKVPoolAllocator.translate_loc_from_full_to_swa()`](../../../mem_cache/allocator/swa.py#L147-L150) 或 KV pool 上的同名接口。这里不新增类、allocator、映射表或翻译方法。

> 当前代码索引：[full→SWA 映射创建与注册](../../../mem_cache/allocator/swa.py#L78-L101) · [alloc 更新映射](../../../mem_cache/allocator/swa.py#L274-L315) · [SWA 回收清理映射](../../../mem_cache/allocator/swa.py#L333-L359) · [GPU DSV4 使用映射](../../../layers/attention/deepseek_v4_backend.py#L1124-L1126) · [NPU 通用路径构造 SWA block table](../attention/ascend_backend.py#L438-L462)

## 3. 改动二：Compressor state 全量切换为显式 `state_loc` ring

改动二的最终目标不是同时维护 paged/ring 两条路径，而是用 Atlas A3 新版 `cache_mode=2` Compressor 完整替代当前 `cache_mode=1`：SGLang 为历史和本轮 token 显式提供 `state_loc`，算子不再根据 request bank 自己取模，也不再使用 state token/page allocator。

### 3.1 统一地址模型

C4 Attention、C4 Indexer 和 C128 使用三类形状不同的 per-layer state pool；每个相关 layer 都有自己的 pool 对象。新版算子统一消费显式 `state_loc`，但 SGLang 生成地址时有两种规则：

```text
C4A / C4Li:
state_loc = (swa_loc // swa_page_size) * c4_ring_size
            + (swa_loc % c4_ring_size)

C128A:
state_loc = req_pool_idx * c128_ring_size
            + (absolute_position % c128_ring_size)
```

`state_block_table` 在 `cache_mode=2` 下改为二维显式地址表，shape 为 `[B, coff * cmp_ratio + input_capacity]`。前 `coff * cmp_ratio` 列描述算子需要读取的历史 state，后 `input_capacity` 列描述本轮 token 的 state 写入位置。表内元素是 flat `state_loc`，位置 0 是合法地址；idle/padding 行必须配合 `seqused=0` 并填入专用 dummy/sentinel loc。

这里的“表”只是新版 A3 算子的调用参数格式，不是新增 state ownership，也不是新的长期 request table。地址值继续由 GPU 公共路径计算：C4 复用 full→SWA mapping 和 `CompressStatePool.translate_from_swa_loc_to_state_loc()`，C128 复用 `CompressStatePool.translate_from_req_position_to_state_loc()`；NPU 只在调用算子前按 `ratio/coff/input_capacity` 把这些地址组装成二维 `state_block_table`。

| State | 物理 pool | `state_cache` 逻辑 shape | `state_loc` 来源 |
| --- | --- | --- | --- |
| C4 Attention | C4A ring pool | `[swa_state_pages, c4_ring_size, 2048]` | SWA physical loc 翻译 |
| C4 Indexer | C4Li ring pool | `[swa_state_pages, c4_ring_size, 512]` | 与 C4A 共用同一份 ratio=4 临时调用 tensor |
| C128 Attention | C128A ring pool | `[req_slots, c128_ring_size, 1024]` | `req_pool_idx + absolute_position` |

C4 Attention 与 C4 Indexer 的 `state_loc` 数值相同，但二者保存的数据和 last dim 不同，必须保留两个独立 tensor。C4A/C4Li 复用 GPU 的 SWA→state 地址转换；C128A 保留 GPU 已有的 request-position ring 地址。

颜色说明：蓝色表示复用，绿色表示新增，橙色表示修改，红色表示删除；灰色是业务输入，紫色是最终结果。

重构前，state 地址沿着“分配→写 loc→写 request table→构造 page table”的链路流动：

```mermaid
flowchart LR
    Batch["ScheduleBatch<br/>Prefill / Decode / Verify"]:::neutral
    Lens["[删除] DSV4StateLens<br/>state alloc len / offset"]:::deleted
    Alloc["[删除] C4/C128 state allocator<br/>按 token/page 分配与回收"]:::deleted
    OutLoc["[删除] out_c4_state_loc<br/>out_c128_state_loc"]:::deleted
    Tables["[删除] req_to_token_c4_state<br/>req_to_token_c128_state"]:::deleted
    PageTable["[删除] c4/c128_state_page_table<br/>INT32[B, max_pages]"]:::deleted
    PagedPool["[修改] NPUCompressStatePool<br/>当前为 paged FP32 layout"]:::changed
    Op["[修改] custom.compressor<br/>cache_mode=1"]:::changed
    Result["本轮 compressed KV<br/>并更新 paged state"]:::result

    Batch --> Lens --> Alloc --> OutLoc --> Tables --> PageTable --> Op --> Result
    PagedPool --> Op

    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef deleted fill:#FDECEC,stroke:#D14343,color:#7F1D1D,stroke-width:2px,stroke-dasharray:5 3;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

重构后，SGLang 直接把 GPU 风格的 per-token `state_loc` 表传给算子，不再存在 state 地址分配链：

```mermaid
flowchart LR
    Batch["ScheduleBatch<br/>Prefill / Decode / Verify"]:::neutral
    Full["[复用] full loc → SWA loc<br/>C4A/C4Li 地址来源"]:::reused
    Req["[复用] req_pool_indices + position<br/>C128A 地址来源"]:::reused
    Seq["[复用] start_pos / seqused<br/>cu_seqlens"]:::reused
    Addr["[复用] GPU state_loc 计算<br/>SWA / request-position"]:::reused
    LocTable["[修改] A3 调用适配<br/>临时组装二维 state_block_table"]:::changed

    subgraph Pools["[复用] GPU CompressStatePool + NPU 薄适配"]
        direction TB
        C4A["C4A state<br/>[swa_state_pages, c4_ring_size, 2048]"]:::reused
        C4Li["C4Li state<br/>[swa_state_pages, c4_ring_size, 512]"]:::reused
        C128A["C128A state<br/>[req_slots, c128_ring_size, 1024]"]:::reused
    end

    Op["[修改] custom.compressor<br/>state_block_table=state_loc table<br/>cache_mode=2"]:::changed
    Epilog["[复用] _compressor_epilog_npu()<br/>写 C4/C128 KV 或 Indexer K/scale"]:::reused
    Result["本轮 compressed KV<br/>并按绝对位置更新 ring state"]:::result

    Batch --> Full --> Addr
    Batch --> Req --> Addr
    Addr --> LocTable
    Batch --> Seq
    LocTable --> Op
    Seq --> Op
    C4A --> Op
    C4Li --> Op
    C128A --> Op
    Op --> Epilog --> Result

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
    style Pools fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:2px;
```

### 3.2 四个可独立 review 的子改动

四个子改动按 storage ownership、Eager、Graph/MTP、PD 四个责任边界拆分。每个子改动同时加入 ring 逻辑并删除被它替代的 paged 逻辑；不再设置“删除 allocator”和“删除 table”两个独立清理阶段。前三项合起来完成单机切换，第四项让跨实例传输复用 GPU 的 StateType 和地址语义。

```mermaid
flowchart LR
    P1["子改动 1<br/>Ring storage ownership<br/>+ 删除 paged allocation"]:::changed
    P2["子改动 2<br/>Eager cache_mode=2<br/>+ 删除 Eager paged metadata"]:::changed
    P3["子改动 3<br/>Graph/MTP ring<br/>+ 删除剩余 state metadata"]:::changed
    P4["子改动 4<br/>PD/StateType 收敛<br/>仅保留 NPU C128 KV 特例"]:::changed
    Final["最终：单机与 PD<br/>Compressor state 仅使用 ring"]:::result

    P1 --> P2 --> P3 --> P4 --> Final

    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

#### 3.2.1 子改动 1：Ring storage ownership 与 paged allocation 删除

- 在 `dsv4_memory_pool.py` 保留现有 `NPUCompressStatePool` 类名、factory、layer mapping 以及 `get_state_cache()` 调用边界，但将它收缩成 GPU `CompressStatePool` 的薄适配层。buffer 大小计算、内存分配、ring ownership、C4/C128 地址翻译全部直接复用 GPU 基类；NPU 只校验 FP32、请求 `ring_size` 对齐的连续三维 view，并把无效地址映射到正数 dummy row。
- 在公共 `CompressStatePool` 增加可选的 `state_cache_page_size` 物理 view 参数，默认值为 1，因此 GPU 原有 flat buffer 行为不变；NPU 传入 `ring_size`，让同一份公共分配结果可直接 view 为 `[bank_count, ring_size, last_dim]`。这不是新增 NPU 分配算法。
- 沿用现有 `compress_state_pools` 与 `indexer_compress_state_pools` 的 per-layer 实例和 layer mapping；C4A、C4Li、C128A 是三种实例类型而非全模型三个对象。C4A/C4Li 的逻辑 row 跟随 SWA physical page，C128A 的逻辑 row 跟随 request slot。
- 不新增第二个 `RingCompressorStatePool`/`NPURingCompressStatePool` 类型，避免 factory、访问接口和生命周期形成双轨抽象。
- C4A/C4Li 使用 `translate_from_swa_loc_to_state_loc()`：`state_loc = (swa_loc // swa_page_size) * c4_ring_size + swa_loc % c4_ring_size`。它们不再按新 request 的 `req_pool_idx` 清 bank，只保持 dummy/sentinel row 为 KV=0、score=`-inf`；有效 row 在对应 token 写入时覆盖。
- C128A 使用 `translate_from_req_position_to_state_loc()`：`state_loc = req_pool_idx * c128_ring_size + position % c128_ring_size`。新 request 第一次取得或复用 `req_pool_idx` 时只清对应 C128 bank：KV 置 0、score 置 `-inf`；chunked prefill、decode 和 verify 继续使用同一 request slot 时不重复清理。
- 同步删除 `npu_state_pool_size()`、paged page view、page-0 sentinel 和 KV cache configurator 中的 paged-state sizing override。
- 同步删除 `c4_state_attn_allocator/c128_state_attn_allocator`、`DSV4StateLens`、`out_c4_state_loc/out_c128_state_loc` 及其 alloc/free/clear/evict 分支；`DSV4OutCacheLoc` 只承载 KV 地址。
- 增加 shape/dtype/contiguous、C4 SWA-loc 翻译、C128 request-position 翻译、C128 slot 清理/复用、dummy/sentinel，以及 allocator/bundle 不再包含 state 的单测。本子改动只负责 storage ownership；Compressor consumer 的 Eager 和 Graph 切换分别属于子改动 2、3。

本子改动的输入是 SWA physical 地址空间、实际 request slot 容量和三种 Compressor shape，结果是 C4A/C4Li state ownership 跟随 SWA page、C128A ownership 跟随 request slot；旧 state allocator、lens、loc 和 paged sizing 在同一个子改动中删除。

重构前，pool 大小和物理页由 paged state allocator 模型决定：

```mermaid
flowchart LR
    Config["输入：max_running_requests<br/>page_size / ratio"]:::neutral
    Sizing["[删除] npu_state_pool_size()<br/>计算 paged slot 数"]:::deleted
    Pool["[修改] NPUCompressStatePool<br/>当前 [blocks, page_size, last_dim]"]:::changed
    Sentinel["[删除] page 0 skip sentinel"]:::deleted
    Alloc["[删除] state allocator / DSV4StateLens<br/>state loc / alloc/free/evict"]:::deleted
    Result["结果：state storage<br/>依赖分页分配生命周期"]:::result

    Config --> Sizing --> Pool --> Sentinel --> Result
    Alloc --> Pool

    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef deleted fill:#FDECEC,stroke:#D14343,color:#7F1D1D,stroke-width:2px,stroke-dasharray:5 3;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

重构后，C4A/C4Li 与 C128A 的 pool 分配及两种地址翻译规则均复用 GPU `CompressStatePool`；NPU 薄适配层只补算子接口差异，allocator bundle 只保留 KV 地址：

```mermaid
flowchart LR
    SWACapacity["输入一：SWA physical pages<br/>full→SWA mapping"]:::neutral
    ReqCapacity["输入二：req_to_token.shape[0]<br/>包含 slot 0 dummy"]:::neutral
    GPUBase["[复用] GPU CompressStatePool<br/>size / allocate / ring / translate"]:::reused
    NPUAdapter["[修改] NPUCompressStatePool 薄适配<br/>FP32 / 3-D view / positive dummy"]:::changed
    C4A["[修改] per-layer C4A pools<br/>SWA page × c4_ring_size × 2048"]:::changed
    C4Li["[修改] per-layer C4Li pools<br/>SWA page × c4_ring_size × 512"]:::changed
    C128A["[修改] per-layer C128A pools<br/>[req_slots, c128_ring_size, 1024]"]:::changed
    SWALoc["[复用] swa_loc → C4 state_loc<br/>不按 request 清 bank"]:::reused
    ReqLoc["[复用] req + position → C128 state_loc<br/>首次/复用时 clear C128 bank"]:::reused
    Sentinel["[复用] dummy/sentinel<br/>KV=0，score=-inf"]:::reused
    KVBundle["[修改] DSV4OutCacheLoc<br/>只包含 KV loc，无 state loc"]:::changed
    Result["结果：GPU 风格显式 state_loc ring<br/>无 state allocator/lens/request table"]:::result

    SWACapacity --> GPUBase
    ReqCapacity --> GPUBase
    GPUBase --> NPUAdapter
    NPUAdapter --> C4A --> Result
    NPUAdapter --> C4Li --> Result
    NPUAdapter --> C128A --> Result
    SWACapacity --> SWALoc --> C4A
    SWALoc --> C4Li
    ReqCapacity --> ReqLoc --> C128A
    Sentinel --> NPUAdapter
    KVBundle --> Result

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

#### 3.2.2 子改动 2：Eager ring 切换与 Eager paged metadata 删除

- `forward_compress()` 固定传 `cache_mode=2`；复用已有 `_get_state_pool()/get_state_cache(layer_id, is_in_indexer)` 路由取得当前层的 C4A、C4Li 或 C128A pool，不新增选择函数。
- 抽取一个按 `ratio` 工作的公共显式地址组装 helper。它复用 GPU `CompressStatePool` 的两种 translate 方法、full→SWA mapping 和现有 req/position metadata，只负责把结果整理为 A3 接口要求的二维 `state_block_table`，不在 NPU 重写 ring 寻址公式。
- 每个 Eager batch、每个 ratio 只组装一次临时调用 tensor：ratio=4 的结果供 C4A/C4Li 共用，ratio=128 的结果供 C128A 使用。它们可以作为本轮 `ForwardMetadata` 的临时计算结果缓存，避免逐层重复生成，但不新增任何长期 state-loc request 字段、allocator 或 ownership。
- 本轮最大输入容量在 metadata 阶段由已有 CPU 长度直接得到：decode 为 1，prefill 为最大 request chunk，verify 为 draft 宽度；地址 helper 不对 NPU `cu_seqlens` 调 `.item()`，避免每轮 decode 引入主机同步。
- 临时 tensor 的宽度是 `coff * cmp_ratio + input_capacity`；历史列和当前 token 列都填 flat INT32 `state_loc`。这个二维载体由新版 A3 ABI 决定，不能继续把一维 `req_pool_idx` 直接当作 `state_block_table`。
- 同步删除 Eager 路径的 state loc 写表 hook、`c4_state_page_table/c128_state_page_table` 构造与消费；prefill 显式设置 `seqused=cu_seqlens[1:]-cu_seqlens[:-1]`。
- 保留 RoPE、Hadamard、输出长度检查和 `_compressor_epilog_npu()`，避免把 state 重构扩散到 compressed KV 写入链路。
- 覆盖 C4A/C4Li/C128A 的 eager prefill、decode、空 batch、非连续/非恒等 `state_loc`，并断言 C4A/C4Li 共享 loc 数值、C128A 使用独立 request-position loc。

本子改动的输入是 Eager batch 的 request/sequence metadata 和子改动 1 改造后的 ring pool，结果是 Eager 计算统一切到 `cache_mode=2`，同时不再产生、写入或消费任何 Eager paged-state metadata。

重构前，backend 从两张 state request table 构造二维 page table：

```mermaid
flowchart LR
    Batch["输入：eager ForwardBatch<br/>req / seq / cu_seqlens"]:::neutral
    Tables["[删除 Eager 依赖] state 写表/查表 hook<br/>读取两张 state request table"]:::deleted
    Build["[删除] _compute_compress_locs()<br/>构造 state page ids"]:::deleted
    Metadata["[删除] c4/c128_state_page_table<br/>INT32[B, max_pages]"]:::deleted
    Pool["[删除] paged state_cache"]:::deleted
    Op["[修改] custom.compressor<br/>cache_mode=1"]:::changed
    Epilog["[复用] _compressor_epilog_npu()"]:::reused
    Result["结果：compressed KV<br/>写回 paged state"]:::result

    Batch --> Build
    Tables --> Build --> Metadata --> Op
    Pool --> Op --> Epilog --> Result

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef deleted fill:#FDECEC,stroke:#D14343,color:#7F1D1D,stroke-width:2px,stroke-dasharray:5 3;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

重构后先复用 GPU 地址计算，再由一个通用 A3 adapter 按 ratio 临时组装调用 tensor；C4A/C4Li 共用 ratio=4 的结果，C128A 使用 ratio=128 的结果：

```mermaid
flowchart LR
    Batch["输入：eager ForwardBatch<br/>full loc / req / cu_seqlens"]:::neutral
    Seq["[修改] start_pos / seqused<br/>prefill 也显式给有效长度"]:::changed
    C4Loc["[复用 GPU] C4 state_loc<br/>full→SWA→translate<br/>C4A/C4Li 共用"]:::reused
    C128Loc["[复用 GPU] C128 state_loc<br/>req + position→translate"]:::reused
    Adapter["[修改] 通用 A3 adapter<br/>按 ratio 临时组装<br/>二维 state_block_table"]:::changed
    Select["[复用] get_state_cache()<br/>layer_id / is_in_indexer 路由"]:::reused
    Cleanup["[删除] Eager state loc 写表 hook<br/>c4/c128_state_page_table"]:::deleted
    Op["[修改] custom.compressor<br/>state_block_table=显式 loc table<br/>cache_mode=2"]:::changed
    Epilog["[复用] RoPE / Hadamard / length check<br/>_compressor_epilog_npu()"]:::reused
    Result["结果：compressed KV<br/>并按表内 state_loc 更新对应 state row"]:::result

    Batch --> Seq --> Op
    Batch --> C4Loc --> Adapter
    Batch --> C128Loc --> Adapter
    Adapter --> Op
    Select --> Op --> Epilog --> Result
    Cleanup --> Result

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef deleted fill:#FDECEC,stroke:#D14343,color:#7F1D1D,stroke-width:2px,stroke-dasharray:5 3;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

#### 3.2.3 子改动 3：Graph/MTP ring 切换与剩余 paged metadata 删除

- Graph 复用子改动 2 的公共地址组装 helper；只有为了满足 capture/replay 的地址稳定约束，才分别分配固定地址的 ratio=4 和 ratio=128 `state_block_table` buffer，replay 只原地 `copy_`。它们是 graph 输入 buffer，不是 request table；idle/padding 行填专用 dummy/sentinel loc，并同步清 `start_pos/seqused`。
- target verify 使用 `start_pos=committed prefix length`、`seqused=draft token count`，按 draft positions 重新计算本轮显式 loc；C4 继续从对应 full/SWA loc 派生，C128 继续从 request-position 派生。
- ring size 按最大单次 request token 宽度计算，至少覆盖最大 verify draft 宽度；rejected suffix 不做 state allocator rollback，下一轮从 committed 位置覆盖。
- 同步删除 Graph 中固定的 `c4/c128_state_page_table`、`c4/c128_state_loc` 及其 refresh/copy 逻辑，并删除 MTP speculative state reserve/rollback/clear 分支。
- Eager 与 Graph consumer 都移除后，同步删除 `req_to_token_c4_state/req_to_token_c128_state`、`write_c4_state()/write_c128_state()` 及剩余 state write/free hooks；这些删除不再单列子改动。
- 覆盖 graph 两次动态 replay、真实/idle 混合 batch、非恒等显式 loc、accepted=0/1/中间值/全部接受，并断言单机代码中不存在 state request table/page table/loc consumer。

本子改动有两路输入：graph replay 的固定 tensor 约束，以及 MTP 的 committed/accepted 长度；结果是 Graph/MTP 与 Eager 使用完全相同的显式 `state_loc` 语义，同时删除单机路径最后的 state request table、page table、loc 和 speculative allocator hook。

重构前，graph 固定持有二维 page table/state loc，MTP 继续预留和回收 speculative state slot：

```mermaid
flowchart LR
    Capture["输入一：Graph capture/replay"]:::neutral
    PageBuf["[删除] 固定 c4/c128_state_page_table<br/>INT32[max_bs, max_pages]"]:::deleted
    LocBuf["[删除] 固定 c4/c128_state_loc"]:::deleted
    Verify["输入二：Target verify<br/>draft tokens"]:::neutral
    Reserve["[删除] 预留 speculative state slot"]:::deleted
    Rollback["[删除] rejected state allocator rollback/clear"]:::deleted
    Tables["[删除] 两张 state request table<br/>剩余 write/free hooks"]:::deleted
    Op["[修改] graph Compressor<br/>cache_mode=1"]:::changed
    Result["结果：graph/MTP state<br/>绑定 paged allocation"]:::result

    Capture --> PageBuf --> Op
    Capture --> LocBuf --> Op
    Verify --> Reserve --> Op
    Reserve --> Rollback --> Result
    Tables --> Result
    Op --> Result

    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef deleted fill:#FDECEC,stroke:#D14343,color:#7F1D1D,stroke-width:2px,stroke-dasharray:5 3;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

重构后，graph 仅为地址稳定固定两块二维 A3 调用 buffer；MTP 通过绝对起点、draft 地址和 accepted 长度表达提交/覆盖：

```mermaid
flowchart LR
    Capture["输入一：Graph capture<br/>max_graph_bs"]:::neutral
    C4Buf["[新增 Graph 输入] ratio=4<br/>state_block_table buffer"]:::added
    C128Buf["[新增 Graph 输入] ratio=128<br/>state_block_table buffer"]:::added
    Replay["[修改] replay 原地 copy_<br/>idle/padding=dummy loc"]:::changed
    Verify["输入二：Target verify<br/>committed length + draft count"]:::neutral
    Meta["[修改] start_pos=committed<br/>seqused=draft count"]:::changed
    DraftLoc["[复用] 子改动 2 公共 helper<br/>draft SWA/request-position loc"]:::reused
    Op["[修改] graph Compressor<br/>显式 loc table + cache_mode=2"]:::changed
    Accept["[新增] 下一轮 start_pos<br/>只推进 accepted"]:::added
    Overwrite["[新增] rejected suffix<br/>同一绝对位置覆盖"]:::added
    Cleanup["[删除] 旧 paged page/一维 loc buffer<br/>两张 request table 与剩余 hooks"]:::deleted
    Result["结果：graph/MTP 与 eager 共享 state_loc<br/>单机无 paged state metadata"]:::result

    Capture --> C4Buf --> Replay --> Op
    Capture --> C128Buf --> Replay
    Verify --> Meta --> DraftLoc --> Op --> Accept --> Overwrite --> Result
    Cleanup --> Result

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef added fill:#E8F7EE,stroke:#22A06B,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef deleted fill:#FDECEC,stroke:#D14343,color:#7F1D1D,stroke-width:2px,stroke-dasharray:5 3;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

#### 3.2.4 子改动 4：PD/StateType 收敛

本子改动必须放在子改动 1–3 之后：只有单机 Eager、Graph/MTP 都已经使用 GPU 风格的显式 `state_loc`，PD 才能安全复用 GPU 的 StateType 和 payload 语义。若在 paged state 仍生效时提前把 C4 state 合入 `StateType.SWA`，connector 会拿 SWA 地址访问独立 paged state，造成地址错配。

- C4 KV 和 Indexer K/scale 进入 GPU 已有的主 KV 传输列表，使用改动一收敛后的 full page index；删除 `DSV4_C4`、`DSV4_INDEXER` 两个 Ascend 专用 StateType。
- SWA KV、C4A state 和 C4Li state 合并注册为同一个 `StateType.SWA` component。三类 buffer 仍然独立，只共享同一组 SWA/state index；删除 `DSV4_SWA`、`DSV4_C4_STATE`。
- C128A compressor state 直接复用公共 `StateType.C128_STATE`；删除 `DSV4_C128_STATE`。
- C128 KV 仍由 `req_to_token_c128` 和独立 allocator 管理，不能进入使用 full index 的主 KV 列表，因此只保留一个 NPU 专用类型 `AscendStateType.DSV4_C128`。
- Python `Enum` 不能继承已有成员的 `StateType` 后再添加枚举值，因此不实现 `class AscendStateType(StateType)`。公共 component 直接使用 `StateType`；`AscendStateType` 是只包含 `DSV4_C128` 的小枚举。
- `get_contiguous_buf_infos()` 注册 C4/Indexer 主 KV buffer；`get_pd_state_components()` 只注册 `StateType.SWA`、`StateType.C128_STATE` 和 `AscendStateType.DSV4_C128`。`dsv4_state_payloads()` 同步只生成这三类 payload。
- `_DSV4_KVCACHE_STATE_TYPES` 收缩为 `(AscendStateType.DSV4_C128,)`；删除 C4/Indexer/SWA/C4-state/C128-state 的 Ascend 专用 dispatch 和 exact-index 分支，复用 GPU connector 的通用处理。
- 覆盖 StateType 顺序、C4/Indexer 主 KV buffer 注册、SWA+C4A+C4Li 多 buffer 共用 index、C128A 公共 state payload、NPU C128 KV 独立 payload，以及 PD 传输后继续 decode/MTP。

本子改动的输入是前三个子改动已经对齐 GPU 的 state 寻址，以及改动一已经对齐的 C4/Indexer full page index；结果是六个 Ascend 专用 StateType 收敛为一个 NPU C128 KV 特例。

重构前，PD state payload 仍依赖 paged request table，并要求源/目标 page index 精确相同：

```mermaid
flowchart LR
    Prefill["输入：Prefill request<br/>req_pool_idx / seq_len"]:::neutral
    SrcTable["[删除] source state request table"]:::deleted
    SrcPages["[删除] dsv4_state_payloads()<br/>构造 source state page ids"]:::deleted
    Decode["输入：Decode 预分配"]:::neutral
    DstAlloc["[删除] destination state allocator/table"]:::deleted
    Exact["[修改] Ascend connector<br/>要求 src page ids == dst page ids"]:::changed
    Transfer["[删除] item_len=one state page<br/>按页搬运"]:::deleted
    Result["结果：PD state 传输<br/>绑定分页地址体系"]:::result

    Prefill --> SrcTable --> SrcPages --> Exact
    Decode --> DstAlloc --> Exact --> Transfer --> Result

    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef deleted fill:#FDECEC,stroke:#D14343,color:#7F1D1D,stroke-width:2px,stroke-dasharray:5 3;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

重构后，公共数据直接进入 GPU 已有的传输类型和地址链路，Ascend connector 只保留 C128 KV 特例：

```mermaid
flowchart LR
    Full["输入一：full page index"]:::neutral
    SWA["输入二：full→SWA/state index"]:::neutral
    MainKV["[复用] 主 KV 传输<br/>C4 KV + Indexer K/scale"]:::reused
    SharedSWA["[复用] StateType.SWA<br/>SWA KV + C4A/C4Li state"]:::reused
    SharedC128State["[复用] StateType.C128_STATE<br/>C128A state"]:::reused
    NPUC128["[保留] AscendStateType.DSV4_C128<br/>独立 C128 KV index"]:::changed
    DeleteTypes["[删除] 其余 5 个<br/>Ascend 专用 StateType/dispatch"]:::deleted
    Connector["[复用] GPU 通用 connector<br/>按 component 传输"]:::reused
    Result["结果：公共类型全部复用 GPU<br/>Ascend 只保留 C128 KV 特例"]:::result

    Full --> MainKV --> Connector
    SWA --> SharedSWA --> Connector
    SharedC128State --> Connector
    NPUC128 --> Connector --> Result
    DeleteTypes --> Result

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef deleted fill:#FDECEC,stroke:#D14343,color:#7F1D1D,stroke-width:2px,stroke-dasharray:5 3;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
```

| 子改动 | 主要文件 | 行为切换 | 独立验收点 |
| --- | --- | --- | --- |
| 1. Ring storage ownership | `deepseek_v4_compress_state.py`、`dsv4_memory_pool.py`、`dsv4_allocator.py`、`forward_batch_info.py`、`kv_cache_configurator.py` | 直接复用 GPU `CompressStatePool` 的分配/ring/翻译；NPU 仅保留 FP32、3-D view、dummy 适配；同步删除 paged allocation | GPU/NPU 分配一致、两类地址翻译、仅清 C128 bank、sentinel、KV-only bundle |
| 2. Eager ring | `ascend_dsv4_backend.py`、Eager hooks | Eager 切新版 `cache_mode=2`；复用 GPU 地址计算，由通用 A3 adapter 按 ratio 临时组装调用 tensor | 三变体、C4A/C4Li 共用 loc、非连续 loc、无长期 loc/request table |
| 3. Graph/MTP ring | `ascend_dsv4_backend.py`、`dsv4_req_to_token_pool.py`、`dsv4_common_hooks.py`、spec runtime | 复用同一 helper；仅为 Graph 固定并刷新 ratio=4/128 调用 buffer，删除剩余 state table/hooks/rollback | replay loc、draft loc、rejected overwrite、单机无 paged metadata |
| 4. PD/StateType 收敛 | `dsv4_memory_pool.py`、`dsv4_common_hooks.py`、`disaggregation/ascend/conn.py`、`disaggregation/utils.py` | 公共 component 复用 GPU StateType/主 KV 路径；Ascend 只保留 C128 KV | 类型顺序与 payload 对齐，传输后 C4/C128/Indexer/state 一致 |

> 当前代码索引：[GPU 公共 `CompressStatePool`](../../../mem_cache/deepseek_v4_compress_state.py#L82-L213) · [NPU 薄适配与 factory](dsv4_memory_pool.py#L69-L327) · [KV-only allocator bundle](dsv4_allocator.py#L122-L390) · [过渡期 state table/graph metadata](../attention/ascend_dsv4_backend.py#L240-L1408) · [fused compressor 调用](../attention/ascend_dsv4_backend.py#L370-L430)

## 4. MTP 相关适配

MTP 的 draft、target verify 和 accepted/rejected 处理会同时使用 KV 写地址与 compressor state，因此两项改动都需要适配，但不改变 MTP 的生成、校验和 Indexer top-k 算法。

这里的 `bundle` 指 [`DSV4OutCacheLoc`](../../../model_executor/forward_batch_info.py#L261-L288)：它不是缓存或映射表，只是一次 forward 使用的 DSV4 写地址集合。改动一后，full loc 成为 SWA/C4 地址的唯一来源；改动二后，state 地址不再放入 bundle。

```text
预留 draft full loc
    → 按 MTP step 取得本步 full loc
    → 翻译 SWA loc / 派生 C4 loc
    → 写入本步 KV 和 ring state
    → target verify 使用相同地址规则
    → accepted/rejected 后提交 KV 长度和 ring state
```

| MTP 环节 | 当前方式 | 完成改动一后 | 完成改动二后 |
| --- | --- | --- | --- |
| DSV4 allocator 识别 | [`allocation.py`](../../../mem_cache/allocation.py#L198-L244) 通过 `hasattr(c4_attn_allocator)` 识别 DSV4 | 改为显式 DSV4 capability/type 标识，删除 C4 allocator 后仍能返回 DSV4 地址 bundle | 标识保留，bundle 不再包含 state loc |
| Draft 地址预留 | [`alloc_paged_token_slots_reserve_extend()`](dsv4_allocator.py#L68-L118) 预留 full/SWA/C4/C128/state 并写各自 request table | 预留 full/SWA/C128 和 paged state；SWA/C4 不再写独立 request table，C4 loc 由 full loc 派生 | 只预留 full/SWA/C128 KV；ring state 使用固定空间，不参与 token allocator 预留 |
| 多步地址获取 | [`_step_out_cache_loc_dsv4()`](../attention/ascend_dsv4_backend.py#L2007-L2074) 从预分配的 `out_swa_loc/out_c4_loc` 按 step 切片 | 每一步先切出本步 full loc，再翻译 SWA loc，并按 4-token 边界派生 C4 loc | KV 地址规则不变；C4 state loc 由本步 SWA loc 翻译，C128 state loc 由 request + position 计算 |
| Target verify | [`maybe_build_dsv4_verify_bundle()`](dsv4_common_hooks.py#L376-L410) 从 SWA/C4/state request table 截取 draft 区间 | 使用 verify 阶段已有的 full loc 重新翻译 SWA loc、派生 C4 loc；C128/state 暂时保留原表 | KV 继续使用改动一规则；按 draft positions 构造二维 C4/C128 `state_loc` 表，并传 `start_pos=committed length`、`seqused=draft count` |
| MTP metadata | 每个 step/replay 维护独立 C4 page table 以及 C4/SWA/state loc buffer | C4 page table 复用 full block table，C4/SWA loc 由本步 full loc 生成 | Graph 固定持有两张二维显式 loc tensor，replay 原地刷新；不保留长期 state request/page table |
| Accepted/rejected | allocator snapshot 回滚 full、SWA、C4、C128 和 state | 删除 C4 allocator 快照；被拒绝的 C4/Indexer 数据由有效序列长度隔离，后续按同一 full page id 覆盖 | state 不做 allocator rollback；下一轮只从 `old_start_pos + accepted` 继续，被拒绝 suffix 在相同绝对位置覆盖 |

MTP 的正确性依赖 runtime 只推进 accepted token 数，并为下一轮重新构造显式地址：C4A/C4Li 从对应 full/SWA loc 翻译，C128A 使用 `req_pool_idx` 和绝对位置取模。新版算子只按二维表中的 `state_loc` 读写，不再自行计算 ring offset。ring size 必须包含最大 verify 宽度的 guard 空间，避免同一轮 draft 覆盖仍需读取的 committed state；满足该容量约束后，被拒绝 suffix 无需 allocator rollback，下一轮会在重新计算出的相同位置覆盖。

改动一还需要保证：draft 地址预留、逐 step draft 和 target verify 使用相同的 request 顺序与 4-token 边界规则，使派生的 C4 loc 始终与 compressor 输出顺序一致。

> 当前代码索引：[MTP 预留空间](../../../speculative/eagle_utils.py#L803-L872) · [多步 full 地址切片](../../../speculative/eagle_utils.py#L58-L80) · [NPU DSV4 多步 bundle](../attention/ascend_dsv4_backend.py#L1984-L2074) · [target verify 地址](../../../speculative/eagle_utils.py#L490-L528) · [allocator 回滚](dsv4_allocator.py#L779-L810)

## 5. PD 分离相关适配

PD 按两项改动分别收敛：改动一让 SWA/C4/Indexer 的 page id 回到 full 地址来源；改动二的子改动 4 再把 Compressor state 从 paged state page list 切到 GPU 已有的显式 index 语义——C4A/C4Li 跟随 SWA index，C128A 跟随 request-position index。两部分都不改变 KV 与 state 物理 buffer 彼此独立的事实。

### 5.1 改动一后的 PD 删除边界

结论是：**需要删除的是 PD 中跟随旧地址所有权的分配、写表和读表分支，不是 C4、Indexer 或 SWA 的数据传输分支。** C4 KV、Indexer K/scale 和 SWA KV 的物理 buffer 仍然存在，因此对应的 `AscendStateType`、buffer 注册和传输请求都必须保留。

这里先统一数量口径：按本方案第 2 节的定义，改动一删除的是 **1 个 C4 allocator 和 2 张辅助 request table**（`req_to_token_c4`、`req_to_token_swa`）。Indexer 一直保留独立 pool，但与 C4 共用 `out_c4_loc`，没有第二个独立 Indexer allocator。如果把 C4 KV 和 Indexer 两个 pool 都称为“两个地址消费者”，它们确实同时失去独立地址所有权，但 runtime 中实际删除的 allocator 只有 C4 allocator。

PD 代码按以下边界处理：

| 代码/职责 | 改动一后的处理 | 原因 |
| --- | --- | --- |
| C4 allocator 的 PD 预留、free、clear/rollback 分支 | 删除 | C4 loc/page id 由 full 地址派生，不再拥有独立生命周期 |
| `write_swa()`、`write_c4()` 及 `_write_dsv4_tables()` 中对应写表分支 | 删除 | `req_to_token_swa`、`req_to_token_c4` 已删除，继续写入会形成第二份地址真相 |
| `dsv4_state_payloads()` 中读取两张旧表的分支 | 替换 | SWA 改为 base `req_to_token` + full→SWA mapping；C4/Indexer 改为 base `req_to_token` 的 full page id |
| C4 和 Indexer page-list 构造 | 合并为同一个 full-page builder/缓存结果 | 两者使用相同 page id；仍以两个 StateType 传输不同 buffer |
| `dsv4_prealloc_kwargs()`、`dsv4_unwrap_prealloc()`、`write_dsv4_prealloc_tables()` | 保留，但只服务 C128 和 paged state | 改动一后 C128 allocator 和两类 state allocator/table 仍存在；删除整个 wrapper 会导致 decode 侧没有目标页 |
| `get_pd_state_components()` 中 SWA/C4/Indexer/C128/state buffer 注册 | 保留 | 地址收敛不等于数据 buffer 合并；C4 与 Indexer 仍需分别传输 |
| `AscendStateType`、`_DSV4_KVCACHE_STATE_TYPES` 和 Ascend generic page dispatch | 保留 | connector 的 StateType 标识物理 buffer component，不标识 allocator/request table |
| `req_to_token_c128` 和两张 state table 的 page-list 构造 | 保留 | 它们要到改动二或后续方案才发生变化 |

当前实现已经完成了表格中的主要删除和替换；进一步适合做的是把 C4/Indexer 两次相同的 full page D2H 构造收敛成一次，并把 `hasattr(c128_attn_allocator)` 这类临时识别改成显式 DSV4 capability，避免能力判断继续绑定某个未来可能变化的子 allocator。

改动一前，PD 预分配和 payload 都跟随各自的独立地址表：

```mermaid
flowchart LR
    Entry["业务入口：为同一 request 构造 PD 源页/目标页<br/>Prefill sender + Decode receiver"]:::neutral

    subgraph PreallocBefore["Decode 侧预分配"]
        direction TB
        Prealloc["[复用] alloc_for_decode_prealloc()<br/>进入 DSV4 预分配"]:::reused
        ParentAlloc["[复用] full + SWA allocator<br/>生成 out_full_loc / out_swa_loc"]:::reused
        C4Alloc["[删除] C4 独立 allocator<br/>生成 out_c4_loc"]:::deleted
        OtherAlloc["[复用] C128 + paged state allocator"]:::reused
        OldWrite["[删除] _write_dsv4_tables() 中<br/>SWA/C4 独立写表分支"]:::deleted

        Prealloc --> ParentAlloc
        Prealloc --> C4Alloc
        Prealloc --> OtherAlloc
        ParentAlloc --> OldWrite
        C4Alloc --> OldWrite
    end

    subgraph PayloadBefore["Prefill/Decode 两侧 page-list 构造"]
        direction TB
        SWATable["[删除] req_to_token_swa"]:::deleted
        C4Table["[删除] req_to_token_c4"]:::deleted
        OtherTables["[复用] req_to_token_c128<br/>两张 state request table"]:::reused
        SWAPages["SWA page ids"]:::neutral
        C4Pages["C4 page ids + Indexer page ids"]:::neutral
        OtherPages["C128/state page ids"]:::neutral

        SWATable --> SWAPages
        C4Table --> C4Pages
        OtherTables --> OtherPages
    end

    Buffers["[复用] get_pd_state_components()<br/>注册各自独立数据 buffer"]:::reused
    Transfer["结果：按 StateType 传输 SWA/C4/Indexer/C128/state"]:::result

    Entry --> Prealloc
    OldWrite --> SWATable
    OldWrite --> C4Table
    OtherAlloc --> OtherTables
    SWAPages --> Transfer
    C4Pages --> Transfer
    OtherPages --> Transfer
    Buffers --> Transfer

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef deleted fill:#FDECEC,stroke:#D14343,color:#7F1D1D,stroke-width:2px,stroke-dasharray:5 3;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
    style PreallocBefore fill:#FFF9EB,stroke:#D97706,color:#0F172A,stroke-width:2px;
    style PayloadBefore fill:#FFF9EB,stroke:#D97706,color:#0F172A,stroke-width:2px;
```

改动一后，PD 保留相同传输框架，但 page id 收敛为三种来源：full、full→SWA mapping、剩余 C128/state table。

```mermaid
flowchart LR
    Entry["业务入口：为同一 request 构造 PD 源页/目标页<br/>Prefill sender + Decode receiver"]:::neutral

    subgraph PreallocAfter["Decode 侧预分配"]
        direction TB
        Prealloc["[复用] alloc_for_decode_prealloc()<br/>进入 DSV4 预分配"]:::reused
        ParentAlloc["[复用] full + SWA allocator<br/>SWA allocator 仍保留"]:::reused
        OtherAlloc["[复用] C128 + paged state allocator"]:::reused
        BaseTable["[复用] 通用路径维护 base req_to_token<br/>只记录 full loc"]:::reused
        RemainingWrite["[修改] dsv4_unwrap_prealloc()<br/>只写 C128/state table"]:::changed
        Removed["[删除] C4 allocator<br/>SWA/C4 独立写表分支"]:::deleted

        Prealloc --> ParentAlloc --> BaseTable
        Prealloc --> OtherAlloc --> RemainingWrite
        Prealloc --> Removed
    end

    subgraph PayloadAfter["Prefill/Decode 两侧 page-list 构造"]
        direction TB
        FullPages["[新增] 一次构造 full physical page ids<br/>来自 base req_to_token"]:::added
        C4Pages["[复用] C4 page ids"]:::reused
        IndexPages["[复用] Indexer page ids<br/>与 C4 使用同一组 id"]:::reused
        Translate["[复用] translate_loc_from_full_to_swa()<br/>查询 full_to_swa_index_mapping"]:::reused
        SWAPages["SWA page ids"]:::neutral
        OtherTables["[复用] req_to_token_c128<br/>两张 state request table"]:::reused
        OtherPages["C128/state page ids"]:::neutral

        BaseTable --> FullPages
        FullPages --> C4Pages
        FullPages --> IndexPages
        BaseTable --> Translate --> SWAPages
        RemainingWrite --> OtherTables --> OtherPages
    end

    Buffers["[复用] get_pd_state_components()<br/>C4/Indexer 仍注册不同 buffer"]:::reused
    StateTypes["[复用] AscendStateType<br/>每个 StateType 对应一组 ptr/item_len/index"]:::reused
    Dispatch["[复用] AscendKVManager generic page dispatch<br/>并校验源/目标 index 数量一致"]:::reused
    Transfer["结果：page id 可共享，数据 buffer 仍按 StateType 分别传输"]:::result

    Entry --> Prealloc
    C4Pages --> StateTypes
    IndexPages --> StateTypes
    SWAPages --> StateTypes
    OtherPages --> StateTypes
    Buffers --> StateTypes --> Dispatch --> Transfer

    classDef reused fill:#E8F1FB,stroke:#3B82F6,color:#0F172A,stroke-width:1.5px;
    classDef added fill:#E8F7EE,stroke:#22A06B,color:#0F172A,stroke-width:1.5px;
    classDef changed fill:#FFF4D6,stroke:#D97706,color:#0F172A,stroke-width:1.5px;
    classDef deleted fill:#FDECEC,stroke:#D14343,color:#7F1D1D,stroke-width:2px,stroke-dasharray:5 3;
    classDef neutral fill:#F4F5F7,stroke:#6B7280,color:#0F172A,stroke-width:1.5px;
    classDef result fill:#EEE8FF,stroke:#7C3AED,color:#0F172A,stroke-width:2px;
    style PreallocAfter fill:#FFF9EB,stroke:#D97706,color:#0F172A,stroke-width:2px;
    style PayloadAfter fill:#FFF9EB,stroke:#D97706,color:#0F172A,stroke-width:2px;
```

### 5.2 Ascend connector 的保留与后续收敛

`disaggregation/ascend/conn.py` 中没有直接读取 allocator 或 request table；它消费的是上游已经构造好的 `state_indices`，再把每个 StateType 对应的 buffer 指针、`item_len` 和源/目标 index 交给通用 page-indexed 传输。因此，改动一**没有可以直接删除的 connector 分支**：

| Ascend connector 对象 | 改动一后的处理 | 说明 |
| --- | --- | --- |
| `AscendStateType.DSV4_SWA` | 保留 | SWA buffer 仍独立，page id 经 full→SWA mapping 得到 |
| `AscendStateType.DSV4_C4` | 保留 | 标识 C4 KV buffer component；不代表 C4 allocator |
| `AscendStateType.DSV4_INDEXER` | 保留 | 与 C4 共用 index，但指向独立 Index K/scale buffers 和不同 `item_len` |
| `AscendStateType.DSV4_C128` | 保留 | C128 buffer 和独立地址体系不变 |
| `DSV4_C4_STATE` / `DSV4_C128_STATE` | 改动一保留 | 改动二子改动 4 删除，分别收敛到 `StateType.SWA` / `StateType.C128_STATE` |
| `_is_generic_kvcache_state_type()` 扩展 | 保留 | 六类 component 当前都走 `_send_kvcache_generic()` page-indexed 传输 |
| `_requires_exact_state_index_match()` 扩展 | 保留 | Prefill/Decode page list 必须位置对齐，不能静默截断 |
| `register_buffer_to_engine()` 的 state component 注册 | 保留 | allocator 删除不释放或合并物理数据 buffer |

改动一阶段保持六个 Ascend StateType，是因为 C4 compressor state 仍使用独立 paged 地址，不能提前合入 `StateType.SWA`。完成改动二的子改动 1–3 后，这个限制消失；子改动 4 再一次性完成类型、buffer 注册和 payload 收敛，避免中间状态混用两套地址语义。

所以改动一的范围仍然是复用 C4/Indexer page-list 构造但保持 connector 协议不变；六个 Ascend 专用 StateType 的删除明确属于改动二子改动 4。

### 5.3 改动二后的 PD/StateType 收敛

| 数据 | 完成改动一后 | 完成改动二后 |
| --- | --- | --- |
| SWA KV | `AscendStateType.DSV4_SWA` | `StateType.SWA`，与 C4A/C4Li state 共用 index |
| C4 KV | `AscendStateType.DSV4_C4` | 主 KV 传输列表，使用 full page index |
| Indexer K/scale | `AscendStateType.DSV4_INDEXER` | 主 KV 传输列表，与 C4 共用 full page index |
| C128 KV | `AscendStateType.DSV4_C128` | 保持 NPU 专用类型和 `req_to_token_c128` |
| C4A/C4Li state | `AscendStateType.DSV4_C4_STATE` | `StateType.SWA`，使用与 GPU 相同的 SWA/state index |
| C128A state | `AscendStateType.DSV4_C128_STATE` | 公共 `StateType.C128_STATE` |

`get_pd_state_components()` 允许一个 StateType component 注册多个独立物理 tensor，因此 `StateType.SWA` 可以同时包含 SWA KV、C4A state 和 C4Li state；共享类型不等于合并 buffer。C4 KV 与 Indexer 则从 state component 移入 `get_contiguous_buf_infos()` 返回的主 KV 列表。

`AscendStateType` 不做对 `StateType` 的 Python 枚举继承，因为已有成员的 `Enum` 不能被扩展。公共类型直接使用 `StateType`，Ascend 小枚举只定义 `DSV4_C128`。相应地，`_DSV4_KVCACHE_STATE_TYPES` 只保留这一项，其他 component 由 GPU 通用 connector 分发。

此时 Prefill/Decode 的 payload、item length 和 index 配对规则直接复用 GPU 实现；NPU 只为 C128 KV 构造独立 page list。PD 传输后按 committed sequence length 继续 decode，不恢复已删除的 state request table 或 cursor。

> 当前代码索引：[DSV4 PD payload](dsv4_common_hooks.py#L84-L197) · [PD 预分配适配](dsv4_common_hooks.py#L200-L290) · [PD buffer 注册](../../../disaggregation/utils.py#L960-L1001) · [Ascend StateType 与分发](../../../disaggregation/ascend/conn.py#L23-L43) · [exact-index 校验](../../../disaggregation/ascend/conn.py#L38-L43)

## 6. 分步目标结构

重构分两步：先完成 full 地址映射收敛，再将 compressor state 切换为 ring。C4 KV、Indexer 和 state 的物理 buffer 始终保留，收敛的是地址分配和 request table。

### 6.1 完成改动一后

| 对象 | 重构前 | 完成改动一后 |
| --- | --- | --- |
| SWA KV | 独立 allocator + `req_to_token_swa` | 保留 allocator，地址由 full→SWA 映射生成 |
| C4 KV | page=128 + 独立 allocator/table | 独立 buffer，page=32，复用 full page id |
| C4 Indexer | 独立 pool，page=128 | 独立 pool，page=32，与 C4 共享 full 地址 |
| C128 KV | 独立 allocator/table | 不变 |
| Compressor state | paged allocator/table | 不变，留到改动二 |

这一步完成后：

- allocator：6 个 → 5 个，删除 C4 allocator。
- DSV4 辅助 request table：5 张 → 3 张，保留 `req_to_token_c128`、`req_to_token_c4_state` 和 `req_to_token_c128_state`。

### 6.2 完成改动二后（最终）

改动二在改动一的结构上，将 compressor state 从 paged 管理改为 ring。

| 对象 | 重构前 | 完成改动二后（最终） |
| --- | --- | --- |
| full | full allocator + base `req_to_token` | 不变，作为 C4/Indexer 地址基准 |
| SWA KV | 独立 allocator + `req_to_token_swa` | 独立 allocator + full→SWA 映射 |
| C4 KV | page=128 + 独立 allocator/table | 独立 buffer，page=32，复用 full page id |
| C4 Indexer | 独立 pool，page=128 | 独立 pool，page=32，与 C4 共享 full 地址 |
| C128 KV | 独立 allocator + `req_to_token_c128` | 不变 |
| C4 attention/Indexer state | paged allocator + `req_to_token_c4_state` | 两个独立 ring tensor，按 SWA physical loc 显式派生 `state_loc` |
| C128 state | paged allocator + `req_to_token_c128_state` | 独立 request-position ring，显式派生 `state_loc`；PD 复用 `StateType.C128_STATE` |
| PD Compressor state | 按 state request table 构造 page list | C4 state 合入 `StateType.SWA`，C128 state 复用 `StateType.C128_STATE` |

这一步完成后：

- allocator：6 个 → 3 个，仅保留 full、SWA 和 C128。
- DSV4 辅助 request table：5 张 → 1 张，仅保留 `req_to_token_c128`。
- 加上通用 base `req_to_token`，最终共保留两张 request→token table。
- 单机和 PD 的 Compressor state 都只使用显式 `state_loc` ring，不再存在 paged state page list。
- Ascend connector 六个专用 StateType 收敛为一个 `DSV4_C128`，其余复用 GPU StateType 或主 KV 路径。

```text
重构前：6 个 allocator / 5 张辅助 request table
改动一：5 个 allocator / 3 张辅助 request table
改动二：3 个 allocator / 1 张辅助 request table
```

> 改动文件索引：[pool 布局](dsv4_memory_pool.py#L47-L272) · [allocator](dsv4_allocator.py#L122-L188) · [request table](dsv4_req_to_token_pool.py#L43-L101) · [table/state hooks](dsv4_common_hooks.py#L242-L373) · [forward 分配结果](../../../model_executor/forward_batch_info.py#L261-L322) · [attention/compressor metadata](../attention/ascend_dsv4_backend.py#L242-L430)

## 7. 后续任务：DSV4 全组件 Prefix Cache

以下任务在本次 mempool 重构完成后实施，不阻塞当前重构。

### 7.1 单机全组件 Prefix Cache

- Full 继续作为主 Radix Tree；SWA 通过 full→SWA mapping、C4/Indexer 通过 full page id 复用，并与对应 full 节点绑定生命周期。
- C128 使用独立 Radix Tree 保存 C128 page loc；最终命中长度取所有必需组件的共同前缀，并满足各组件 page/compress 对齐约束。
- C4/C128 compressor ring state 不能直接随 token page 复用，需要保存命中边界的 state checkpoint，或从最近 checkpoint 重算；具体方案后续确定。
- 统一处理各组件引用计数、淘汰和回收，覆盖 Full、SWA、C4、Indexer、C128 及 compressor state 的命中正确性和 page/bank 复用测试。

### 7.2 PD 分离适配

- Prefill 端传输 Full、SWA、C4、Indexer、C128 以及必要的 compressor state/checkpoint 和有效前缀元信息；Decode 端按本地地址体系恢复所有组件。
- P/D 不直接复用物理 page/bank loc；Decode 端完成本地分配、地址映射和数据传输后，再原子发布对应 Prefix Cache 条目。
- 任一组件传输失败或请求取消时回滚全部目标资源，覆盖跨实例全组件命中、传输后继续 decode 和淘汰复用测试。
