import sys
import unittest
from dataclasses import fields
from unittest.mock import MagicMock

import torch

sys.modules.setdefault("torch_npu", MagicMock())

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
