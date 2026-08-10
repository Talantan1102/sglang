import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    C128SidecarComponent,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


class _Allocator(SWATokenToKVPoolAllocator):
    device = "cpu"
    page_size = 128
    c128_attn_allocator = SimpleNamespace(free_pages=torch.empty(0, dtype=torch.int64))

    def __init__(self):
        self.refs = {}
        self.freed_full = []

    def retain_c128_pages(self, page_ids):
        for page_id in page_ids.tolist():
            self.refs[page_id] = self.refs.get(page_id, 0) + 1

    def release_c128_pages(self, page_ids):
        for page_id in page_ids.tolist():
            self.refs[page_id] -= 1

    def free_segment(self, indices, *, start_pos):
        self.freed_full.extend(indices.tolist())

    def free_segments(self, segments):
        for indices, start_pos in segments:
            self.free_segment(indices, start_pos=start_pos)

    def free(self, indices):
        self.freed_full.extend(indices.tolist())

    @staticmethod
    def translate_loc_from_full_to_swa(loc):
        return loc + 10000


class _ReqPool:
    def __init__(self):
        self.req_to_c128_sidecar = torch.tensor(
            [[11, 12, 13]], dtype=torch.int64
        )

    def set_c128_prefix_pages(self, req, page_ids):
        req.matched_c128_pages = page_ids.clone()


class TestC128SidecarComponent(unittest.TestCase):
    def setUp(self):
        self.allocator = _Allocator()
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=_ReqPool(),
            token_to_kv_pool_allocator=self.allocator,
            page_size=128,
            tree_components=(ComponentType.FULL, ComponentType.C128),
            component_registry_override={ComponentType.C128: C128SidecarComponent},
        )
        self.cache = UnifiedRadixCache(params)

    def _insert(self, token_ids, full_start, c128_pages):
        if isinstance(c128_pages, int):
            c128_pages = [c128_pages]
        self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=torch.arange(full_start, full_start + len(token_ids)),
                c128_value=torch.tensor(c128_pages, dtype=torch.int64),
            )
        )

    def _match(self, token_ids):
        req = SimpleNamespace(session=None)
        result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids), req=req)
        )
        return result, req.matched_c128_pages

    def test_partial_group_is_not_a_cache_hit(self):
        token_ids = array("q", range(2048))
        self._insert(token_ids, 0, 11)

        result, pages = self._match(array("q", range(128)))

        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(pages.numel(), 0)

    def test_branching_inside_group_keeps_distinct_sidecars(self):
        first = array("q", range(2048))
        second = array("q", list(range(128)) + list(range(10000, 11920)))
        self._insert(first, 0, 11)
        self._insert(second, 3000, 12)

        first_result, first_pages = self._match(first)
        second_result, second_pages = self._match(second)

        self.assertEqual(len(first_result.device_indices), 2048)
        self.assertEqual(len(second_result.device_indices), 2048)
        self.assertEqual(first_pages.tolist(), [11])
        self.assertEqual(second_pages.tolist(), [12])
        self.assertEqual(self.allocator.refs, {11: 1, 12: 1})

    def test_long_leaf_materializes_every_group_boundary(self):
        token_ids = array("q", range(4096))
        self._insert(token_ids, 0, [11, 12])

        first_group, first_pages = self._match(token_ids[:2048])
        partial_second, partial_pages = self._match(token_ids[:3072])
        full, full_pages = self._match(token_ids)

        self.assertEqual(len(first_group.device_indices), 2048)
        self.assertEqual(first_pages.tolist(), [11])
        self.assertEqual(len(partial_second.device_indices), 2048)
        self.assertEqual(partial_pages.tolist(), [11])
        self.assertEqual(len(full.device_indices), 4096)
        self.assertEqual(full_pages.tolist(), [11, 12])

    def test_eagle_cache_length_uses_complete_logical_groups(self):
        component = self.cache.components[ComponentType.C128]
        self.cache.tree_core.is_eagle = True
        req = SimpleNamespace(req_pool_idx=0)

        for raw_len, expected_len, expected_pages in (
            (2048, 0, []),
            (2049, 2049, [11]),
            (4096, 2049, [11]),
            (4097, 4097, [11, 12]),
        ):
            with self.subTest(raw_len=raw_len):
                params = InsertParams()
                cache_len = component.prepare_for_caching_req(
                    req=req,
                    insert_params=params,
                    token_ids_len=raw_len,
                    is_finished=False,
                )
                self.assertEqual(cache_len, expected_len)
                self.assertEqual(params.c128_value.tolist(), expected_pages)

    def test_non_eagle_cache_length_is_unchanged(self):
        component = self.cache.components[ComponentType.C128]
        params = InsertParams()

        cache_len = component.prepare_for_caching_req(
            req=SimpleNamespace(req_pool_idx=0),
            insert_params=params,
            token_ids_len=4096,
            is_finished=False,
        )

        self.assertEqual(cache_len, 4096)
        self.assertEqual(params.c128_value.tolist(), [11, 12])

    def test_eagle_partial_second_group_stays_out_of_radix(self):
        component = self.cache.components[ComponentType.C128]
        self.cache.tree_core.is_eagle = True
        token_ids = array("q", range(4096))
        params = InsertParams()

        cache_len = component.prepare_for_caching_req(
            req=SimpleNamespace(req_pool_idx=0),
            insert_params=params,
            token_ids_len=len(token_ids),
            is_finished=False,
        )
        key = RadixKey(token_ids[:cache_len], is_bigram=True).page_aligned(128)
        params.key = key
        params.value = torch.arange(len(key))
        self.cache.insert(params)

        result, pages = self._match(token_ids)
        self.cache.inc_lock_ref(result.last_device_node)

        self.assertEqual(len(result.device_indices), 2048)
        self.assertEqual(pages.tolist(), [11])
        self.assertEqual(self.cache.total_size(), (2048, 1))
        self.assertEqual(self.cache.protected_size(), 2048)
        self.assertEqual(self.cache.evictable_size(), 0)

    def test_branch_after_complete_group_shares_only_complete_page(self):
        first = array("q", range(4096))
        second = array("q", list(range(3072)) + list(range(10000, 11024)))
        self._insert(first, 0, [11, 12])
        self._insert(second, 5000, [21, 22])

        first_result, first_pages = self._match(first)
        second_result, second_pages = self._match(second)

        self.assertEqual(len(first_result.device_indices), 4096)
        self.assertEqual(len(second_result.device_indices), 4096)
        self.assertEqual(first_pages.tolist(), [11, 12])
        self.assertEqual(second_pages.tolist(), [11, 22])

    def test_radix_eviction_releases_sidecar_reference(self):
        token_ids = array("q", range(2048))
        self._insert(token_ids, 0, 11)

        result = self.cache.evict(EvictParams(num_tokens=2048))

        self.assertEqual(result.num_tokens_evicted, 2048)
        self.assertEqual(self.allocator.refs, {11: 0})

    def test_group_splits_keep_deferred_swa_rebuilds_aligned(self):
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=_ReqPool(),
            token_to_kv_pool_allocator=self.allocator,
            page_size=128,
            sliding_window_size=128,
            tree_components=(
                ComponentType.FULL,
                ComponentType.SWA,
                ComponentType.C128,
            ),
            component_registry_override={ComponentType.C128: C128SidecarComponent},
        )
        cache = UnifiedRadixCache(params)
        token_ids = array("q", range(4096))

        cache.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=torch.arange(4096),
                c128_value=torch.tensor([11, 12]),
                swa_evicted_seqlen=0,
            )
        )

        stack = list(cache.root_node.children.values())
        while stack:
            node = stack.pop()
            full = node.component_data[ComponentType.FULL].value
            swa = node.component_data[ComponentType.SWA].value
            if swa is not None:
                self.assertEqual(len(swa), len(full))
            stack.extend(node.children.values())

        req = SimpleNamespace(session=None)
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids[:2048]), req=req)
        )
        self.assertEqual(len(result.device_indices), 2048)
        self.assertEqual(req.matched_c128_pages.tolist(), [11])


if __name__ == "__main__":
    unittest.main()
