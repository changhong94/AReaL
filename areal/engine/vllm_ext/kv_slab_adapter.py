# SPDX-License-Identifier: Apache-2.0
"""Opt-in FineServe slab policy + whole-KV physical reclaim on Ascend.

Pinned to the vLLM 0.26 / Ascend backend API. No dependency on FineServe's CUDA
service; see licenses/FineServe.txt for the adapted allocation policy.
"""

import json
import os
from functools import wraps
from typing import Any

from areal.engine.colocation_memory import (
    AscendPoolBackend,
    TaggedMemory,
    adaptive_memory_enabled,
)
from areal.engine.vllm_ext.kv_slab import SlabFreeBlockQueue, set_scheduler_capacity
from areal.engine.vllm_ext.kv_slab_memory import KVSlabMemory


def slab_blocks() -> int:
    value = int(os.environ.get("AREAL_KV_SLAB_BLOCKS", "0"))
    if value < 0:
        raise ValueError("AREAL_KV_SLAB_BLOCKS must be nonnegative")
    return value


async def transition_slab_cache(engine: Any, mapped: bool) -> None:
    """Called only with admission closed and the engine scheduler paused."""
    if not slab_blocks():
        return
    if adaptive_memory_enabled():
        # Reclaim decisions belong to the trainer's budget policy. Weight load
        # and resume still require every inference allocation to be mapped.
        if mapped:
            results = await engine.collective_rpc(
                "areal_colocation_memory", args=("restore",)
            )
            if not results or any(
                result.get("states", {}).get(tag) != "mapped"
                for result in results
                for tag in ("weights", "kv_cache")
            ):
                raise RuntimeError(
                    "Adaptive inference worker did not acknowledge restoration"
                )
        return
    stats = await engine.engine_core.call_utility_async("areal_kv_slab_stats")
    if stats["used_blocks"] != 0:
        raise RuntimeError("Cannot transition KV slabs with live blocks")
    results = await engine.collective_rpc("areal_kv_slab_transition", args=(mapped,))
    target = "mapped" if mapped else "released"
    if not results or any(r.get("state") != target for r in results):
        raise RuntimeError("KV slab workers did not acknowledge memory transition")


def install_scheduler_slab() -> None:
    """Install CPU scheduler hooks in the API and spawned EngineCore processes."""
    blocks_per_slab = slab_blocks()
    if not blocks_per_slab:
        return
    from packaging.version import Version
    from vllm import __version__
    from vllm.logger import init_logger
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.engine.core import EngineCore

    if Version(__version__).release != (0, 26, 0):
        raise RuntimeError("KV slab adapter requires vLLM 0.26.0")
    if getattr(BlockPool, "_areal_kv_slab", False):
        return
    logger = init_logger("vllm.areal_kv_slab")
    original_init = BlockPool.__init__

    @wraps(original_init)
    def initialize_pool(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.enable_caching:
            raise ValueError("KV slabs require prefix caching disabled")
        if self.null_block.block_id != 0:
            raise RuntimeError("Unexpected vLLM null block")
        self.free_block_queue = SlabFreeBlockQueue(self.blocks, blocks_per_slab)
        logger.info(
            "KV_SLAB_POOL %s", json.dumps(self.free_block_queue.table.snapshot())
        )

    def slab_stats(self):
        queue = self.scheduler.kv_cache_manager.block_pool.free_block_queue
        if not isinstance(queue, SlabFreeBlockQueue):
            raise RuntimeError("KV slab queue was not installed")
        stats = queue.table.snapshot()
        logger.info("KV_SLAB_STATS %s", json.dumps(stats))
        return stats

    def set_slab_capacity(self, num_blocks):
        return set_scheduler_capacity(
            self.scheduler.kv_cache_manager.block_pool,
            num_blocks,
            self.scheduler.kv_cache_manager.watermark_blocks,
        )

    EngineCore.areal_set_kv_capacity = set_slab_capacity

    # Patch the class objects, not module aliases captured by vLLM imports.
    BlockPool.__init__ = initialize_pool
    BlockPool._areal_kv_slab = True
    EngineCore.areal_kv_slab_stats = slab_stats


def install_kv_slab() -> None:
    """Install device hooks only after Ascend has initialized native imports."""
    blocks_per_slab = slab_blocks()
    if not blocks_per_slab:
        return
    install_scheduler_slab()
    from vllm.logger import init_logger

    logger = init_logger("vllm.areal_kv_slab")
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm_ascend.device_allocator.camem import CaMemAllocator
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if getattr(NPUModelRunner, "_areal_device_kv_slab", False):
        return
    NPUModelRunner._areal_device_kv_slab = True

    from areal.engine.vllm_ext import dynamic_kv

    if dynamic_kv.enabled():
        if not adaptive_memory_enabled():
            raise ValueError("Dynamic KV requires adaptive NPU sharing")
        dynamic_kv.install_dynamic_state_hook(NPUModelRunner)

    original_allocate = NPUModelRunner._allocate_kv_cache_tensors
    original_load_model = NPUModelRunner.load_model

    if adaptive_memory_enabled():

        @wraps(original_load_model)
        def load_model(self, *args, **kwargs):
            self._adaptive_backend = AscendPoolBackend()
            self._adaptive_memory = TaggedMemory(
                self._adaptive_backend,
                ["weights", "kv_cache"],
                retained_tags=("weights",),
            )
            with self._adaptive_backend.allocation_scope("weights"):
                result = original_load_model(self, *args, **kwargs)
            self._adaptive_backend.assert_covered(
                list(self.model.parameters()) + list(self.model.buffers())
            )
            return result

        NPUModelRunner.load_model = load_model
        original_execute_model = NPUModelRunner.execute_model

        @wraps(original_execute_model)
        def execute_with_backing(self, *args, **kwargs):
            # All publication buckets have completed before generation resumes.
            # Weight-only copies may overlap read-only inference; KV stays dirty.
            self._adaptive_memory.start_retained_backup()
            return original_execute_model(self, *args, **kwargs)

        NPUModelRunner.execute_model = execute_with_backing

    @wraps(original_allocate)
    def allocate_kv(self, cache_config):
        config = self.vllm_config
        parallel = config.parallel_config
        if (
            config.model_config.enable_sleep_mode
            or config.cache_config.enable_prefix_caching
            or config.speculative_config is not None
            or config.kv_transfer_config is not None
            or parallel.distributed_executor_backend not in ("uni", "mp")
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or config.model_config.hf_config.model_type != "qwen2"
            or config.cache_config.cache_dtype != "auto"
            or len(cache_config.kv_cache_groups) != 1
            or type(cache_config.kv_cache_groups[0].kv_cache_spec)
            is not FullAttentionSpec
        ):
            raise ValueError(
                "KV slabs require uni/mp Qwen2 full attention, "
                "native KV dtype, no prefix cache/sleep/speculation/KV transfer"
            )
        if not config.model_config.enforce_eager:
            mode = config.compilation_config.cudagraph_mode
            if getattr(mode, "name", str(mode)) not in {"FULL", "FULL_DECODE_ONLY"}:
                raise ValueError("Shared KV graphs require FULL or FULL_DECODE_ONLY")
        if dynamic_kv.enabled():
            return dynamic_kv.allocate_dynamic_kv(self, cache_config)
        allocator = CaMemAllocator.get_instance()
        adaptive = adaptive_memory_enabled()
        if allocator.pointer_to_data and not adaptive:
            raise RuntimeError("KV slabs require an exclusive CaMem KV pool")
        scope = (
            self._adaptive_backend.allocation_scope("kv_cache")
            if adaptive
            else allocator.use_memory_pool(tag="kv_cache")
        )
        with scope:
            tensors = original_allocate(self, cache_config)

        unique = {}
        for value in tensors.values():
            for tensor in value if isinstance(value, tuple) else (value,):
                unique[tensor.data_ptr()] = tensor

        def reset_cache():
            # BF16 native post-wake hook does not zero null/padding blocks.
            with torch.inference_mode():
                for tensor in unique.values():
                    tensor.zero_()
            self.post_kv_cache_wake_up()

        self._areal_reset_kv = reset_cache
        if adaptive:
            logger.info(
                "AUTO_MEMORY_INFERENCE_INIT %s",
                json.dumps(self._adaptive_memory.snapshot()),
            )
            return tensors

        self._areal_kv_slab_memory = KVSlabMemory(
            allocator,
            torch.npu.synchronize,
            reset_cache,
            lambda: torch.npu.mem_get_info()[0],
        )
        spec = cache_config.kv_cache_groups[0].kv_cache_spec
        logger.info(
            "KV_SLAB_MEMORY_INIT %s",
            json.dumps(
                {
                    "managed_bytes": self._areal_kv_slab_memory._validate_inventory(),
                    "blocks_per_slab": blocks_per_slab,
                    "logical_slab_bytes_per_layer": blocks_per_slab
                    * spec.page_size_bytes,
                    "physical_reclaim_granularity": "whole_kv_pool",
                }
            ),
        )
        return tensors

    NPUModelRunner._allocate_kv_cache_tensors = allocate_kv
