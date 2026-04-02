import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

try:
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
        ForwardMetadata,
    )

    TORCH_NPU_AVAILABLE = True
except ImportError:
    TORCH_NPU_AVAILABLE = False


@unittest.skipUnless(TORCH_NPU_AVAILABLE, "requires torch_npu")
class TestAscendFIAChunkedPrefillUnit(unittest.TestCase):
    def _make_backend(self):
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        backend.is_dllm_model = False
        backend.use_mla = False
        backend.use_fia = True
        backend.attn_cp_size = 1
        backend.is_hybrid_swa = False
        backend.page_size = 2
        backend.fia_mask = torch.ones((8, 8), dtype=torch.bool)
        backend.forward_metadata = ForwardMetadata(
            block_tables=torch.tensor([[0, 1, 2, 0]], dtype=torch.int32),
            seq_lens_cpu_int=torch.tensor([5], dtype=torch.int32),
            extend_seq_lens_cpu_int=torch.tensor([2], dtype=torch.int32),
        )
        return backend

    def _make_layer(self):
        return SimpleNamespace(
            tp_q_head_num=4,
            tp_k_head_num=2,
            tp_v_head_num=2,
            qk_head_dim=4,
            v_head_dim=4,
            scaling=0.125,
            is_cross_attention=False,
        )

    def _make_forward_mode(self):
        return SimpleNamespace(
            is_target_verify=lambda: False,
            is_draft_extend=lambda: False,
            is_draft_extend_v2=lambda: False,
            is_context_parallel_extend=lambda: False,
        )

    def test_forward_extend_uses_paged_kv_when_prefix_exists(self):
        backend = self._make_backend()
        layer = self._make_layer()

        k_cache = torch.randn(4, 2, layer.tp_k_head_num, layer.qk_head_dim)
        v_cache = torch.randn(4, 2, layer.tp_v_head_num, layer.v_head_dim)
        pool = SimpleNamespace(
            set_kv_buffer=MagicMock(),
            get_key_buffer=MagicMock(return_value=k_cache),
            get_value_buffer=MagicMock(return_value=v_cache),
        )
        forward_batch = SimpleNamespace(
            forward_mode=self._make_forward_mode(),
            attn_cp_metadata=None,
            token_to_kv_pool=pool,
            out_cache_loc=torch.tensor([0, 1], dtype=torch.int32),
            extend_prefix_lens_cpu=[3],
            extend_seq_lens_cpu=[2],
            encoder_lens=None,
        )
        q = torch.randn(2, layer.tp_q_head_num * layer.qk_head_dim)
        k = torch.randn(2, layer.tp_k_head_num * layer.qk_head_dim)
        v = torch.randn(2, layer.tp_v_head_num * layer.v_head_dim)
        expected = torch.randn(2, layer.tp_q_head_num * layer.v_head_dim)
        backend._forward_extend_fia_paged_kv = MagicMock(return_value=expected)

        out = AscendAttnBackend.forward_extend(backend, q, k, v, layer, forward_batch)

        pool.set_kv_buffer.assert_called_once_with(layer, forward_batch.out_cache_loc, k, v)
        backend._forward_extend_fia_paged_kv.assert_called_once_with(
            q, k_cache, v_cache, layer
        )
        self.assertIs(out, expected)

    def test_forward_extend_fia_paged_kv_uses_cached_lengths(self):
        backend = self._make_backend()
        backend.forward_metadata = ForwardMetadata(
            block_tables=torch.tensor([[0, 1, 2, 0], [3, 4, 0, 0]], dtype=torch.int32),
            seq_lens_cpu_int=torch.tensor([5, 7], dtype=torch.int32),
            extend_seq_lens_cpu_int=torch.tensor([2, 3], dtype=torch.int32),
        )
        layer = self._make_layer()
        q = torch.randn(5, layer.tp_q_head_num * layer.qk_head_dim)
        k_cache = torch.randn(5, 2, layer.tp_k_head_num, layer.qk_head_dim)
        v_cache = torch.randn(5, 2, layer.tp_v_head_num, layer.v_head_dim)

        with patch(
            "torch.ops.npu.npu_fused_infer_attention_score",
            return_value=(
                torch.zeros(5, layer.tp_q_head_num, layer.v_head_dim),
                None,
            ),
        ) as mock_fia:
            out = backend._forward_extend_fia_paged_kv(q, k_cache, v_cache, layer)

        args, kwargs = mock_fia.call_args
        self.assertEqual(args[0].shape, (5, layer.tp_q_head_num, layer.qk_head_dim))
        self.assertEqual(
            args[1].shape, (5, backend.page_size, layer.tp_k_head_num * layer.qk_head_dim)
        )
        self.assertEqual(
            args[2].shape, (5, backend.page_size, layer.tp_v_head_num * layer.v_head_dim)
        )
        self.assertIs(kwargs["block_table"], backend.forward_metadata.block_tables)
        self.assertEqual(kwargs["actual_seq_lengths"], [2, 5])
        self.assertEqual(kwargs["actual_seq_lengths_kv"], [5, 7])
        self.assertIs(kwargs["atten_mask"], backend.fia_mask)
        self.assertEqual(kwargs["input_layout"], "TND")
        self.assertEqual(out.shape, (5, layer.tp_q_head_num * layer.v_head_dim))


if __name__ == "__main__":
    unittest.main()
