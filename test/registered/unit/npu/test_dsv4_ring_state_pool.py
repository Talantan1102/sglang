import sys
import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

sys.modules.setdefault("torch_npu", MagicMock())

from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (  # noqa: E402
    CompressorAscendBackendMixin,
    _build_explicit_state_block_table,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (  # noqa: E402
    DSV4NPUTokenToKVPoolAllocator,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (  # noqa: E402
    DSV4NPUTokenToKVPool,
    NPUCompressStatePool,
)
from sglang.srt.mem_cache.deepseek_v4_compress_state import (  # noqa: E402
    CompressStatePool,
)
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    DSV4OutCacheLoc,
)


class _FakeTokenPool:
    def __init__(self, full_to_swa, attention_state_pool, indexer_state_pool=None):
        self.full_to_swa = full_to_swa
        self.attention_state_pool = attention_state_pool
        self.indexer_state_pool = indexer_state_pool or attention_state_pool

    def translate_loc_from_full_to_swa(self, full_locs):
        return self.full_to_swa[full_locs]

    def _get_state_pool(self, layer_id, from_indexer):
        del layer_id
        return (
            self.indexer_state_pool if from_indexer else self.attention_state_pool
        )


class TestNPUCompressStatePool(unittest.TestCase):
    def test_shared_gpu_pool_keeps_default_flat_allocation(self):
        pool = CompressStatePool(
            size=32,
            ring_size=8,
            overlap=True,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=4,
            swa_page_size=128,
        )

        # The new physical-view option is opt-in. The shared GPU default keeps
        # its existing ratio-aligned flat allocation: align(32 + 8 + 1, 4).
        self.assertEqual(pool.kv_score_buffer.shape, torch.Size([44, 8]))
        self.assertEqual(pool.page_size, 1)

    def test_c4_uses_swa_owned_explicit_locations(self):
        pool = NPUCompressStatePool(
            size=32,
            ring_size=8,
            overlap=True,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=4,
            swa_page_size=128,
        )

        self.assertEqual(pool.state_cache_3d.shape, (6, 8, 8))
        self.assertTrue(pool.state_cache_3d.is_contiguous())
        swa_locs = torch.tensor([0, 7, 8, 127, 128, -1], dtype=torch.int64)
        state_locs = pool.translate_from_swa_loc_to_state_loc(swa_locs)
        self.assertEqual(
            state_locs.tolist(), [0, 7, 0, 7, 8, pool.dummy_state_loc]
        )
        self.assertEqual(
            pool.kv_score_buffer.kv[pool.dummy_state_loc].tolist(), [0] * 4
        )
        self.assertTrue(
            torch.isneginf(pool.kv_score_buffer.score[pool.dummy_state_loc]).all()
        )

    def test_c128_uses_request_position_and_clears_reused_slot(self):
        pool = NPUCompressStatePool(
            size=3 * 128,
            ring_size=128,
            overlap=False,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=128,
            swa_page_size=128,
        )

        positions = torch.tensor([0, 127, 128, -1], dtype=torch.int64)
        state_locs = pool.translate_from_req_position_to_state_loc(
            torch.tensor(2, dtype=torch.int64), positions
        )
        self.assertEqual(
            state_locs.tolist(), [256, 383, 256, pool.dummy_state_loc]
        )

        start = 128
        end = start + pool.ring_size
        pool.kv_score_buffer.kv[start:end].fill_(3)
        pool.kv_score_buffer.score[start:end].fill_(4)
        token_pool = object.__new__(DSV4NPUTokenToKVPool)
        token_pool.compress_state_pools = [pool]
        token_pool.clear_c128_req_state(1)

        self.assertTrue(
            torch.equal(
                pool.kv_score_buffer.kv[start:end],
                torch.zeros_like(pool.kv_score_buffer.kv[start:end]),
            )
        )
        self.assertTrue(torch.isneginf(pool.kv_score_buffer.score[start:end]).all())

    def test_rejects_non_fp32_state(self):
        with self.assertRaisesRegex(AssertionError, "requires FP32"):
            NPUCompressStatePool(
                size=32,
                ring_size=8,
                overlap=True,
                head_dim=2,
                dtype=torch.bfloat16,
                device="cpu",
                enable_memory_saver=False,
                ratio=4,
                swa_page_size=128,
            )


class TestExplicitStateBlockTable(unittest.TestCase):
    def test_prefill_sets_explicit_seqused_from_cu_seqlens(self):
        backend = object.__new__(CompressorAscendBackendMixin)
        backend._dsv4_unique_compress_ratios = (4, 128)
        backend.forward_metadata = SimpleNamespace(
            actual_seq_lengths_q_pa=torch.tensor([0, 2, 5], dtype=torch.int32)
        )
        forward_batch = SimpleNamespace(
            seq_lens=torch.tensor([2, 10], dtype=torch.int32),
            positions=torch.arange(5, dtype=torch.int64),
            batch_size=2,
            extend_prefix_lens=torch.tensor([0, 7], dtype=torch.int32),
            extend_prefix_lens_cpu=[0, 7],
            out_cache_loc_dsv4=None,
        )

        backend._build_npu_compress_metadata_prefill(forward_batch)

        self.assertEqual(backend.forward_metadata.start_pos.tolist(), [0, 7])
        self.assertEqual(backend.forward_metadata.seqused.tolist(), [2, 3])

    def test_eager_metadata_does_not_read_paged_state_tables(self):
        class EagerReqToTokenPool:
            req_to_token_c128 = torch.arange(256, dtype=torch.int64).view(1, -1)

            @property
            def req_to_token_c4_state(self):
                raise AssertionError("eager must not read the C4 state table")

            @property
            def req_to_token_c128_state(self):
                raise AssertionError("eager must not read the C128 state table")

        backend = object.__new__(CompressorAscendBackendMixin)
        backend._dsv4_unique_compress_ratios = (4, 128)
        backend.page_size = 128
        result = backend._compute_compress_locs(
            pool=MagicMock(),
            req_to_token=torch.arange(256, dtype=torch.int64).view(1, -1),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([4], dtype=torch.int64),
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            is_decode=False,
            bs=1,
            device=torch.device("cpu"),
            req_to_token_pool=EagerReqToTokenPool(),
            out_cache_loc_dsv4=None,
            is_graph=False,
        )

        self.assertNotIn("c4_state_page_table", result)
        self.assertNotIn("c128_state_page_table", result)

    def test_c4_reuses_full_to_swa_and_gpu_state_translation(self):
        pool = NPUCompressStatePool(
            size=32,
            ring_size=8,
            overlap=True,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=4,
            swa_page_size=128,
        )
        req_to_token = torch.arange(1, 17, dtype=torch.int64).view(1, -1)
        full_to_swa = torch.zeros(17, dtype=torch.int64)
        full_to_swa[1:] = torch.arange(128, 144, dtype=torch.int64)
        full_to_swa[1:7] = torch.tensor([128, 259, 130, 389, 132, 519])
        token_pool = _FakeTokenPool(full_to_swa, pool)

        table = _build_explicit_state_block_table(
            compress_ratio=4,
            coff=2,
            state_pool=pool,
            token_to_kv_pool=token_pool,
            req_to_token=req_to_token,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            start_pos=torch.tensor([4], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            seqused=torch.tensor([2], dtype=torch.int32),
            max_input_capacity=2,
        )

        self.assertEqual(table.dtype, torch.int32)
        self.assertTrue(table.is_contiguous())
        self.assertEqual(table.shape, (1, 10))
        self.assertEqual(
            table[0].tolist(),
            [pool.dummy_state_loc] * 4 + [8, 19, 10, 29, 12, 39],
        )

    def test_c128_uses_request_ring_and_dummies_inactive_capacity(self):
        pool = NPUCompressStatePool(
            size=4 * 128,
            ring_size=128,
            overlap=False,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=128,
            swa_page_size=128,
        )

        table = _build_explicit_state_block_table(
            compress_ratio=128,
            coff=1,
            state_pool=pool,
            token_to_kv_pool=MagicMock(),
            req_to_token=torch.zeros((4, 256), dtype=torch.int64),
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
            start_pos=torch.tensor([2, 130], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1, 3], dtype=torch.int32),
            seqused=torch.tensor([1, 0], dtype=torch.int32),
            max_input_capacity=2,
        )

        self.assertEqual(table.shape, (2, 130))
        self.assertEqual(
            table[0, -4:].tolist(),
            [128, 129, 130, pool.dummy_state_loc],
        )
        self.assertTrue(torch.all(table[1] == pool.dummy_state_loc))

    def test_eager_forward_uses_mode2_and_reuses_c4_table(self):
        attention_pool = NPUCompressStatePool(
            size=32,
            ring_size=8,
            overlap=True,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=4,
            swa_page_size=128,
        )
        indexer_pool = NPUCompressStatePool(
            size=32,
            ring_size=8,
            overlap=True,
            head_dim=1,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=4,
            swa_page_size=128,
        )
        full_to_swa = torch.zeros(17, dtype=torch.int64)
        full_to_swa[1:] = torch.arange(128, 144, dtype=torch.int64)
        token_pool = _FakeTokenPool(full_to_swa, attention_pool, indexer_pool)
        backend = object.__new__(CompressorAscendBackendMixin)
        backend.graph_mode = False
        backend.token_to_kv_pool = token_pool
        backend.req_to_token = torch.arange(1, 17, dtype=torch.int64).view(1, -1)
        backend.forward_metadata = SimpleNamespace(
            positions_cmp_padding_c4=torch.zeros(1, dtype=torch.int64),
            start_pos=torch.tensor([4], dtype=torch.int32),
            seqused=torch.tensor([1], dtype=torch.int32),
            actual_seq_lengths_q_pa=torch.tensor([0, 1], dtype=torch.int32),
            c4_loc=None,
            dsv4_explicit_state_block_tables={},
            dsv4_max_input_capacity=1,
        )
        backend._ensure_compressor_hadamard = MagicMock()
        backend._ensure_fused_caches = MagicMock()
        backend._compressor_epilog_npu = MagicMock()
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            forward_mode=SimpleNamespace(
                is_prefill=lambda: False,
                is_target_verify=lambda: False,
            ),
        )

        def make_compressor(from_indexer):
            return SimpleNamespace(
                ratio=4,
                overlap=True,
                layer_id=0,
                is_in_indexer=from_indexer,
                _fused_wkv_w=torch.zeros((1, 1), dtype=torch.bfloat16),
                _fused_wgate_w=torch.zeros((1, 1), dtype=torch.bfloat16),
                ape=torch.zeros((1,), dtype=torch.float32),
                _fused_norm_weight_fp32=torch.zeros((1,), dtype=torch.float32),
                rope_head_dim=64,
                freqs_cis=torch.zeros((1, 1), dtype=torch.float32),
                rotary_emb=None,
                norm=SimpleNamespace(variance_epsilon=1e-6),
                rotate=False,
            )

        rope = MagicMock()
        rope.get_cos_sin.return_value = (
            torch.zeros((1, 64), dtype=torch.float32),
            torch.zeros((1, 64), dtype=torch.float32),
        )
        x = torch.zeros((1, 1), dtype=torch.bfloat16)
        with (
            patch(
                "sglang.srt.hardware_backend.npu.dsv4.dsv4_rope."
                "Dsv4NpuRoPE.for_freqs",
                return_value=rope,
            ),
            patch.object(torch.ops.custom, "compressor", create=True) as op,
        ):
            op.return_value = torch.empty((0, 2), dtype=torch.bfloat16)
            backend.forward_compress(make_compressor(False), x, forward_batch)
            first_table = op.call_args.kwargs["state_block_table"]
            self.assertEqual(op.call_args.kwargs["cache_mode"], 2)

            backend.forward_compress(make_compressor(True), x, forward_batch)
            second_table = op.call_args.kwargs["state_block_table"]
            self.assertIs(first_table, second_table)
            self.assertEqual(op.call_args.kwargs["cache_mode"], 2)

            backend.graph_mode = True
            backend.forward_compress(make_compressor(False), x, forward_batch)
            self.assertEqual(op.call_args.kwargs["cache_mode"], 2)
            self.assertEqual(op.call_count, 3)

    def test_eager_forward_matrix_c4a_c4li_c128a_prefill_decode(self):
        cases = (
            ("C4A", 4, False, 2),
            ("C4Li", 4, True, 1),
            ("C128A", 128, False, 2),
        )

        rope = MagicMock()
        rope.get_cos_sin.return_value = (
            torch.zeros((1, 64), dtype=torch.float32),
            torch.zeros((1, 64), dtype=torch.float32),
        )
        with (
            patch(
                "sglang.srt.hardware_backend.npu.dsv4.dsv4_rope."
                "Dsv4NpuRoPE.for_freqs",
                return_value=rope,
            ),
            patch.object(torch.ops.custom, "compressor", create=True) as op,
        ):
            for name, ratio, from_indexer, head_dim in cases:
                for is_prefill in (False, True):
                    with self.subTest(name=name, is_prefill=is_prefill):
                        ring_size = 8 if ratio == 4 else 128
                        state_size = 64 if ratio == 4 else 4 * 128
                        state_pool = NPUCompressStatePool(
                            size=state_size,
                            ring_size=ring_size,
                            overlap=ratio == 4,
                            head_dim=head_dim,
                            dtype=torch.float32,
                            device="cpu",
                            enable_memory_saver=False,
                            ratio=ratio,
                            swa_page_size=128,
                        )
                        full_to_swa = torch.zeros(513, dtype=torch.int64)
                        full_to_swa[1:] = torch.arange(256, 768)
                        token_pool = _FakeTokenPool(
                            full_to_swa,
                            state_pool,
                            state_pool if from_indexer else None,
                        )
                        backend = object.__new__(CompressorAscendBackendMixin)
                        backend.graph_mode = False
                        backend.token_to_kv_pool = token_pool
                        backend.req_to_token = torch.zeros(
                            (4, 512), dtype=torch.int64
                        )
                        backend.req_to_token[2] = torch.arange(1, 513)
                        backend._ensure_compressor_hadamard = MagicMock()
                        backend._ensure_fused_caches = MagicMock()
                        backend._compressor_epilog_npu = MagicMock()

                        capacity = 3 if is_prefill else 1
                        start_pos = (
                            (4 if ratio == 4 else 130)
                            if is_prefill
                            else ratio - 1
                        )
                        backend.forward_metadata = SimpleNamespace(
                            **{
                                f"positions_cmp_padding_c{ratio}": torch.zeros(
                                    1, dtype=torch.int64
                                ),
                                f"c{ratio}_loc": (
                                    None
                                    if is_prefill
                                    else torch.tensor([7], dtype=torch.int32)
                                ),
                            },
                            start_pos=torch.tensor([start_pos], dtype=torch.int32),
                            seqused=torch.tensor([capacity], dtype=torch.int32),
                            actual_seq_lengths_q_pa=torch.tensor(
                                [0, capacity], dtype=torch.int32
                            ),
                            dsv4_explicit_state_block_tables={},
                            dsv4_max_input_capacity=capacity,
                        )
                        forward_batch = SimpleNamespace(
                            req_pool_indices=torch.tensor([2], dtype=torch.int64),
                            forward_mode=SimpleNamespace(
                                is_prefill=lambda value=is_prefill: value,
                                is_target_verify=lambda: False,
                            ),
                        )
                        compressor = SimpleNamespace(
                            ratio=ratio,
                            overlap=ratio == 4,
                            layer_id=0,
                            is_in_indexer=from_indexer,
                            _fused_wkv_w=torch.zeros(
                                (1, 1), dtype=torch.bfloat16
                            ),
                            _fused_wgate_w=torch.zeros(
                                (1, 1), dtype=torch.bfloat16
                            ),
                            ape=torch.zeros((1,), dtype=torch.float32),
                            _fused_norm_weight_fp32=torch.zeros(
                                (1,), dtype=torch.float32
                            ),
                            rope_head_dim=64,
                            freqs_cis=torch.zeros((1, 1), dtype=torch.float32),
                            rotary_emb=None,
                            norm=SimpleNamespace(variance_epsilon=1e-6),
                            rotate=False,
                        )

                        op.return_value = torch.empty(
                            (0 if is_prefill else 1, 2), dtype=torch.bfloat16
                        )
                        backend.forward_compress(
                            compressor,
                            torch.zeros((capacity, 1), dtype=torch.bfloat16),
                            forward_batch,
                        )

                        kwargs = op.call_args.kwargs
                        table = kwargs["state_block_table"]
                        self.assertEqual(kwargs["cache_mode"], 2)
                        self.assertEqual(kwargs["cmp_ratio"], ratio)
                        self.assertEqual(kwargs["coff"], 2 if ratio == 4 else 1)
                        self.assertEqual(
                            table.shape,
                            (1, (2 if ratio == 4 else 1) * ratio + capacity),
                        )
                        self.assertEqual(table.dtype, torch.int32)
                        self.assertEqual(
                            op.call_args.args[3].data_ptr(),
                            state_pool.state_cache_3d.data_ptr(),
                        )
                        self.assertEqual(
                            backend._compressor_epilog_npu.call_count,
                            0 if is_prefill else 1,
                        )

    def test_empty_batch_builds_empty_explicit_tables(self):
        for ratio, coff, ring_size in ((4, 2, 8), (128, 1, 128)):
            with self.subTest(ratio=ratio):
                state_pool = NPUCompressStatePool(
                    size=ring_size,
                    ring_size=ring_size,
                    overlap=ratio == 4,
                    head_dim=1,
                    dtype=torch.float32,
                    device="cpu",
                    enable_memory_saver=False,
                    ratio=ratio,
                    swa_page_size=128,
                )
                table = _build_explicit_state_block_table(
                    compress_ratio=ratio,
                    coff=coff,
                    state_pool=state_pool,
                    token_to_kv_pool=_FakeTokenPool(
                        torch.zeros(1, dtype=torch.int64), state_pool
                    ),
                    req_to_token=torch.zeros((1, 1), dtype=torch.int64),
                    req_pool_indices=torch.empty(0, dtype=torch.int64),
                    start_pos=torch.empty(0, dtype=torch.int32),
                    cu_seqlens=torch.tensor([0], dtype=torch.int32),
                    seqused=torch.empty(0, dtype=torch.int32),
                    max_input_capacity=1,
                )

                self.assertEqual(table.shape, (0, coff * ratio + 1))
                self.assertEqual(table.dtype, torch.int32)
                self.assertTrue(table.is_contiguous())

    def test_idle_batch_skips_compressor(self):
        backend = object.__new__(CompressorAscendBackendMixin)
        compressor = MagicMock()
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: True)
        )

        backend.forward_core_compressor(None, forward_batch, 0, compressor)

        compressor.assert_not_called()


class TestKVOnlyDSV4AllocationBundle(unittest.TestCase):
    def test_bundle_contains_no_state_locations(self):
        self.assertEqual(
            [field.name for field in fields(DSV4OutCacheLoc)],
            ["out_full_loc", "out_swa_loc", "out_c4_loc", "out_c128_loc"],
        )

    def test_allocator_only_allocates_c128_compressed_kv(self):
        allocator = object.__new__(DSV4NPUTokenToKVPoolAllocator)
        allocator.c128_attn_allocator = MagicMock()
        allocator._alloc_c_extend = MagicMock(
            return_value=torch.tensor([900], dtype=torch.int64)
        )

        bundle = allocator._alloc_compressed_kv(
            out_full_loc=torch.tensor([646, 647, 648, 649], dtype=torch.int64),
            out_swa_loc=torch.tensor([100, 101, 102, 103], dtype=torch.int64),
            prefix_lens=torch.tensor([6], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([6], dtype=torch.int64),
            seq_lens=torch.tensor([10], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10], dtype=torch.int64),
            last_loc_dtype=torch.int64,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
        )

        self.assertEqual(bundle.out_c4_loc.tolist(), [161])
        self.assertEqual(bundle.out_c128_loc.tolist(), [900])
        allocator._alloc_c_extend.assert_called_once()
        self.assertEqual(allocator._alloc_c_extend.call_args.kwargs["ratio"], 128)


if __name__ == "__main__":
    unittest.main()
