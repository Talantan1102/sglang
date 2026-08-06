import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.modules.setdefault("torch_npu", MagicMock())

from sglang.srt.disaggregation import utils as disagg_utils  # noqa: E402
from sglang.srt.disaggregation.ascend.conn import (  # noqa: E402
    AscendKVManager,
    AscendStateType,
)
from sglang.srt.disaggregation.base.conn import StateType  # noqa: E402
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (  # noqa: E402
    dsv4_state_payloads,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (  # noqa: E402
    DSV4NPUTokenToKVPool,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_req_to_token_pool import (  # noqa: E402
    DSV4ReqToTokenTablesMixin,
)


def _state_pool(ratio: int, ring_size: int, rows: int = 256):
    return SimpleNamespace(
        ratio=ratio,
        ring_size=ring_size,
        kv_score_buffer=SimpleNamespace(
            kv_score=torch.zeros((rows, 4), dtype=torch.float32)
        ),
    )


def _fake_npu_pool(swa_mapping=None):
    pool = object.__new__(DSV4NPUTokenToKVPool)
    pool._unified_kv = False
    pool.compression_ratios = [4, 128, 4]
    pool.page_size = 32
    pool.sliding_window = 512
    pool.full_to_swa_index_mapping = swa_mapping or object()
    pool.c4_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((4, 32, 1, 2), dtype=torch.bfloat16) for _ in range(2)]
    )
    pool.c4_indexer_kv_pool = SimpleNamespace(
        index_k_buffer=[torch.zeros((4, 32, 1, 2), dtype=torch.int8) for _ in range(2)],
        index_scale_buffer=[
            torch.zeros((4, 32, 1, 1), dtype=torch.float16) for _ in range(2)
        ],
    )
    pool.swa_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((4, 128, 1, 2), dtype=torch.bfloat16) for _ in range(3)]
    )
    pool.c128_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((4, 128, 1, 2), dtype=torch.bfloat16)]
    )
    pool.compress_state_pools = [
        _state_pool(4, 8),
        _state_pool(128, 128, rows=512),
        _state_pool(4, 8),
    ]
    pool.indexer_compress_state_pools = [
        _state_pool(4, 8),
        None,
        _state_pool(4, 8),
    ]
    return pool


def _fake_nextn_npu_pool(swa_mapping):
    pool = object.__new__(DSV4NPUTokenToKVPool)
    pool._unified_kv = False
    pool.compression_ratios = [0, 0]
    pool.page_size = 32
    pool.sliding_window = 512
    pool.full_to_swa_index_mapping = swa_mapping
    pool.swa_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((4, 128, 1, 2), dtype=torch.bfloat16) for _ in range(2)]
    )
    pool.compress_state_pools = []
    pool.indexer_compress_state_pools = []
    return pool


class TestDSV4PDReuse(unittest.TestCase):
    def test_only_c128_kv_keeps_ascend_state_type(self):
        self.assertEqual(list(AscendStateType), [AscendStateType.DSV4_C128])

    def test_pool_groups_match_gpu_components(self):
        pool = _fake_npu_pool()

        main_ptrs, _, main_items = pool.get_contiguous_buf_infos()
        swa_ptrs, _, swa_items = pool.get_state_buf_infos()
        c128_state_ptrs, _, c128_state_items = pool.get_c128_state_buf_infos()
        c128_kv_ptrs, _, c128_kv_items = pool.get_c128_kv_buf_infos()

        self.assertEqual(len(main_ptrs), 6)
        self.assertEqual(len(main_items), 6)
        self.assertEqual(len(swa_ptrs), 7)
        self.assertEqual(swa_items[-1], 4 * 4 * 8)
        self.assertEqual(len(c128_state_ptrs), 1)
        self.assertEqual(c128_state_items, [4 * 4 * 128])
        self.assertEqual(len(c128_kv_ptrs), 1)
        self.assertEqual(c128_kv_items, [128 * 1 * 2 * 2])

    def test_setup_reuses_public_state_types(self):
        pool = _fake_npu_pool()
        kv_args = SimpleNamespace()

        with (
            patch.object(disagg_utils, "is_npu", return_value=True),
            patch.object(
                disagg_utils,
                "DSV4NPUTokenToKVPool",
                DSV4NPUTokenToKVPool,
                create=True,
            ),
        ):
            disagg_utils.setup_state_kv_args(kv_args, pool)

        self.assertEqual(
            kv_args.state_types,
            [StateType.SWA, StateType.C128_STATE, AscendStateType.DSV4_C128],
        )

    def test_nextn_reuses_public_swa_component_on_npu(self):
        swa_mapping = object()
        pool = _fake_npu_pool(swa_mapping)
        draft_pool = _fake_nextn_npu_pool(swa_mapping)
        kv_args = SimpleNamespace()

        with (
            patch.object(disagg_utils, "is_npu", return_value=True),
            patch.object(
                disagg_utils,
                "DSV4NPUTokenToKVPool",
                DSV4NPUTokenToKVPool,
                create=True,
            ),
        ):
            disagg_utils.setup_state_kv_args(kv_args, pool, draft_pool)

        self.assertEqual(
            kv_args.state_types,
            [
                StateType.SWA,
                StateType.C128_STATE,
                AscendStateType.DSV4_C128,
                StateType.SWA,
            ],
        )
        self.assertEqual(len(kv_args.state_data_ptrs[-1]), 2)

    def test_c128_kv_payload_is_the_only_npu_payload(self):
        req_pool = SimpleNamespace(
            req_to_token_c128=torch.zeros((1, 256), dtype=torch.int32)
        )
        req_pool.req_to_token_c128[0, 0] = 256
        req_pool.req_to_token_c128[0, 128] = 512

        payloads = dsv4_state_payloads(
            req_pool,
            req_pool_idx=0,
            seq_len=20000,
            page_size=128,
        )

        self.assertEqual(set(payloads), {AscendStateType.DSV4_C128})
        np.testing.assert_array_equal(
            payloads[AscendStateType.DSV4_C128](),
            np.array([2, 4], dtype=np.int32),
        )

    def test_pp_mapping_reuses_gpu_state_layout(self):
        manager = object.__new__(AscendKVManager)
        manager.kv_args = SimpleNamespace(
            mla_compression_ratios=[4, 128, 4, 128, 4, 128],
            prefill_start_layer=2,
            prefill_end_layer=5,
        )

        src_main = [1, 2, 3, 4, 5, 6]
        dst_main = [10, 11, 12, 20, 21, 22, 30, 31, 32]
        _, mapped_main, _ = manager.get_mla_kv_ptrs_with_pp(src_main, dst_main)
        self.assertEqual(mapped_main, [11, 12, 21, 22, 31, 32])

        _, mapped_c128, _ = manager.get_mla_kv_ptrs_with_pp(
            [7], [100, 101, 102], AscendStateType.DSV4_C128
        )
        self.assertEqual(mapped_c128, [101])

        _, mapped_c128_state, _ = manager.get_mla_kv_ptrs_with_pp(
            [8], [200, 201, 202], StateType.C128_STATE
        )
        self.assertEqual(mapped_c128_state, [201])

        src_swa = list(range(7))
        dst_swa = list(range(12))
        _, mapped_swa, _ = manager.get_mla_kv_ptrs_with_pp(
            src_swa, dst_swa, StateType.SWA
        )
        self.assertEqual(mapped_swa, [2, 3, 4, 7, 8, 10, 11])

    def test_request_pool_keeps_only_c128_kv_table(self):
        pool = object.__new__(DSV4ReqToTokenTablesMixin)
        pool._alloc_size = 2
        pool._init_dsv4_tables(256, "cpu", False)

        self.assertEqual(pool.req_to_token_c128.shape, (2, 2))
        self.assertFalse(hasattr(pool, "req_to_token_c4_state"))
        self.assertFalse(hasattr(pool, "req_to_token_c128_state"))


if __name__ == "__main__":
    unittest.main()
