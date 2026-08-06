"""DSV4-NPU per-request mapping pool.

Subclass of ``ReqToTokenPool`` that adds the one auxiliary per-request table
needed by the DSV4 attention backend:

  * ``req_to_token_c128``       — slot ids in the c128 compressed-KV pool

The c128 KV pool stores 1 slot per 128 raw tokens, so its per-req table column
count is ``max_context_len // 128``. C4 locations are derived from the base
full-token table, and SWA locations use the existing full-to-SWA mapping.

The c128 address table costs 64KB for size=64 and max_context_len=32K.

The tables are populated by the ``dsv4_common_hooks`` writers (driven from
``mem_cache/common.py``) immediately after a successful alloc_extend /
alloc_decode, using the per-pool slot indices returned in ``DSV4OutCacheLoc``.
"""

from __future__ import annotations

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


class DSV4ReqToTokenTablesMixin:
    """Shared DSV4-NPU per-req table logic for the prefill/normal pool
    (:class:`DSV4NPUReqToTokenPool`) and the disagg-decode pool
    (:class:`DSV4NPUDecodeReqToTokenPool`), which differ only in their base.

    Host class must call ``super().__init__(...)`` first (so ``_alloc_size``
    exists) then ``self._init_dsv4_tables(...)``; ``free`` should call
    ``self._dsv4_free(req)`` before delegating to the base ``free``.
    """

    def _init_dsv4_tables(
        self, max_context_len: int, device: str, enable_memory_saver: bool
    ) -> None:
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        # Back-ref to DSV4NPUTokenToKVPoolAllocator, wired via
        # register_dsv4_allocator after both exist, so free(req) can release
        # c128 pages. None at construction so base clear() runs safely.
        self._dsv4_allocator = None

        # C128 KV uses one slot per 128 raw tokens.
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token_c128 = torch.zeros(
                (self._alloc_size, max(1, max_context_len // 128)),
                dtype=torch.int32,
                device=device,
            )

    # Per-pool write helpers, called by mem_cache/common.py after alloc, using
    # slot indices from DSV4OutCacheLoc. Args: (req_pool_idx, token_offset), slot.
    def write_c128(self, indices, values: torch.Tensor) -> None:
        self.req_to_token_c128[indices] = values

    def register_dsv4_allocator(self, allocator) -> None:
        """Wire the DSV4NPUTokenToKVPoolAllocator ref so ``free(req)`` can
        release C128 KV pages and clear the fixed C128 state bank."""
        self._dsv4_allocator = allocator

    def _dsv4_free(self, req) -> None:
        # Trigger C128 KV free/state clear via the allocator's unified path. May be None
        # between __init__ and register_dsv4_allocator — defensive None check.
        if self._dsv4_allocator is not None:
            self._dsv4_allocator.free(req=req, req_to_token_pool=self)


class DSV4NPUReqToTokenPool(DSV4ReqToTokenTablesMixin, ReqToTokenPool):
    """ReqToTokenPool extended with the DSV4 C128 KV table.

    Drop-in replacement for ReqToTokenPool when the model is DeepSeek-V4 on
    NPU. Selected by ``model_runner_kv_cache_mixin`` based on model arch +
    device. Non-DSV4 and non-NPU paths continue to use the base class.

    The auxiliary tables are intentionally NOT zeroed on ``clear()``: they are
    indexed only by active rows (via req_pool_idx) and only each row's
    ``[:seq_len]`` prefix is read, so stale entries past kv_committed_len are
    unreachable by the attention metadata builder.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        super().__init__(size, max_context_len, device, enable_memory_saver)
        self._init_dsv4_tables(max_context_len, device, enable_memory_saver)

    def free(self, req):
        self._dsv4_free(req)
        super().free(req)


class DSV4NPUDecodeReqToTokenPool(DSV4ReqToTokenTablesMixin, DecodeReqToTokenPool):
    """DecodeReqToTokenPool with the C128 KV table.

    The disagg-decode counterpart of DSV4NPUReqToTokenPool; DecodeReqToTokenPool
    pre-allocates extra req slots for in-flight prefill transfers.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )
        self._init_dsv4_tables(max_context_len, device, enable_memory_saver)

    def free(self, req):
        self._dsv4_free(req)
        super().free(req)
