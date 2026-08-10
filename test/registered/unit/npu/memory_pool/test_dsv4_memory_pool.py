import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

sys.modules.setdefault("torch_npu", MagicMock())

from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
    DSV4NPUTokenToKVPool,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


class TestNPUCompressedPoolPageSize(unittest.TestCase):
    @staticmethod
    def _pool():
        pool = object.__new__(DSV4NPUTokenToKVPool)
        pool.qk_nope_head_dim = 128
        pool.qk_rope_head_dim = 64
        pool.page_size = 128
        return pool

    @patch(
        "sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool."
        "NPUDeepSeekV4SingleKVPool"
    )
    def test_c128_uses_sidecar_page_size_16(self, mock_pool_cls):
        self._pool()._make_kv_pool(
            size=10,
            page_size=1,
            dtype=torch.bfloat16,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
            global_page_size=128,
        )

        self.assertEqual(mock_pool_cls.call_args.kwargs["kernel_page_size"], 16)

if __name__ == "__main__":
    unittest.main()
