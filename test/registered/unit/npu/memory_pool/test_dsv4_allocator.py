import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
    DSV4NPUTokenToKVPoolAllocator,
    get_last_loc,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_build_dsv4_verify_bundle,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_req_to_token_pool import (
    DSV4ReqToTokenTablesMixin,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


class TestC128SidecarAllocator(unittest.TestCase):
    def test_c128_last_loc_uses_sidecar_page_and_slot(self):
        table = torch.tensor([[7, 11]], dtype=torch.int32)

        last_loc = get_last_loc(
            table,
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([0, 3, 16, 17]),
        )

        self.assertEqual(
            last_loc.tolist(), [-1, 7 * 16 + 2, 7 * 16 + 15, 11 * 16]
        )

    def test_c128_extend_reuses_page_and_opens_new_page(self):
        allocator = object.__new__(DSV4NPUTokenToKVPoolAllocator)
        allocator._empty_loc = torch.empty(0, dtype=torch.int64)
        allocator._cur_req_to_token_pool = SimpleNamespace(
            req_to_c128_sidecar=torch.tensor([[7, 0]], dtype=torch.int32)
        )
        c128_allocator = MagicMock()
        c128_allocator.alloc_extend.side_effect = [
            torch.tensor([7 * 16 + 3], dtype=torch.int32),
            torch.tensor([11 * 16], dtype=torch.int32),
        ]

        in_page = allocator._alloc_c_extend(
            c128_allocator,
            torch.tensor([384]),
            torch.tensor([384]),
            torch.tensor([512]),
            torch.tensor([512]),
            torch.tensor([0]),
            torch.int64,
            ratio=128,
        )
        new_page = allocator._alloc_c_extend(
            c128_allocator,
            torch.tensor([2048]),
            torch.tensor([2048]),
            torch.tensor([2176]),
            torch.tensor([2176]),
            torch.tensor([0]),
            torch.int64,
            ratio=128,
        )

        self.assertEqual(in_page.tolist(), [7 * 16 + 3])
        self.assertEqual(new_page.tolist(), [11 * 16])
        self.assertEqual(
            c128_allocator.alloc_extend.call_args_list[0].args[4].tolist(),
            [7 * 16 + 2],
        )
        self.assertEqual(
            c128_allocator.alloc_extend.call_args_list[1].args[4].tolist(),
            [7 * 16 + 15],
        )

    def test_write_c128_records_new_sidecar_pages(self):
        pool = object.__new__(DSV4ReqToTokenTablesMixin)
        pool.req_to_c128_sidecar = torch.tensor([[7, 0, 0]], dtype=torch.int32)
        allocator = object.__new__(DSV4NPUTokenToKVPoolAllocator)
        allocator.release_c128_pages = MagicMock()
        allocator.retain_c128_pages = MagicMock()
        pool._dsv4_allocator = allocator
        values = torch.cat(
            (
                torch.arange(11 * 16, 12 * 16, dtype=torch.int32),
                torch.tensor([12 * 16], dtype=torch.int32),
            )
        )

        pool.write_c128((0, slice(16, 33)), values)

        self.assertEqual(pool.req_to_c128_sidecar.tolist(), [[7, 11, 12]])
        self.assertEqual(
            allocator.retain_c128_pages.call_args.args[0].tolist(), [11, 12]
        )

    def test_c128_page_released_after_last_owner(self):
        allocator = object.__new__(DSV4NPUTokenToKVPoolAllocator)
        allocator.c128_page_refcount = torch.zeros(8, dtype=torch.int32)
        allocator.c128_attn_allocator = MagicMock()

        page = torch.tensor([3], dtype=torch.int32)
        allocator.retain_c128_pages(page)  # request
        allocator.retain_c128_pages(page)  # radix node
        allocator.release_c128_pages(page)
        allocator.c128_attn_allocator.free.assert_not_called()

        allocator.release_c128_pages(page)
        allocator.c128_attn_allocator.free.assert_called_once()
        self.assertEqual(
            allocator.c128_attn_allocator.free.call_args.args[0].tolist(), [48]
        )

    def test_resize_updates_c128_physical_page_capacity(self):
        allocator = object.__new__(DSV4NPUTokenToKVPoolAllocator)
        allocator.c128_attn_allocator = SimpleNamespace(size=320, num_pages=20)
        config = SimpleNamespace(c128_max_total_num_tokens=160)

        with patch.object(SWATokenToKVPoolAllocator, "resize") as parent_resize:
            allocator.resize(config)

        self.assertEqual(allocator.c128_attn_allocator.size, 160)
        self.assertEqual(allocator.c128_attn_allocator.num_pages, 10)
        parent_resize.assert_called_once_with(config)


class TestC128SidecarMTP(unittest.TestCase):
    def test_verify_bundle_uses_sidecar_page_and_slot(self):
        empty = torch.empty(0, dtype=torch.int32)
        reserve_bundle = DSV4OutCacheLoc(
            out_full_loc=empty,
            out_swa_loc=empty,
            out_c4_loc=empty,
            out_c128_loc=empty,
        )
        pool = SimpleNamespace(
            req_to_c128_sidecar=torch.tensor([[7], [9]], dtype=torch.int32),
        )
        batch = SimpleNamespace(
            req_to_token_pool=pool,
            out_cache_loc_dsv4=reserve_bundle,
            req_pool_indices_cpu=torch.tensor([0, 1]),
            seq_lens_cpu=torch.tensor([127, 2047]),
            out_cache_loc=empty,
            token_to_kv_pool_allocator=SimpleNamespace(
                translate_loc_from_full_to_swa=lambda loc: loc,
            ),
        )

        bundle = maybe_build_dsv4_verify_bundle(batch, draft_token_num=3)

        self.assertEqual(bundle.out_c128_loc.tolist(), [7 * 16, 9 * 16 + 15])


if __name__ == "__main__":
    unittest.main()
