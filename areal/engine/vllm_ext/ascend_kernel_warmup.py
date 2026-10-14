# SPDX-License-Identifier: Apache-2.0
"""Warm Ascend request kernels before the worker becomes ready.

Eager execution still uses Triton JIT. Scalar 1 and aligned/non-aligned token
counts can select different specializations, even after a large profile run.
"""

import functools
import importlib
import os
import time


def request_warmup_sizes(max_tokens: int) -> tuple[int, ...]:
    return tuple(size for size in (1, 2, 16) if size <= max_tokens)


def warmup_request_kernels(worker) -> None:
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.logger import init_logger

    logger = init_logger("vllm.areal_kernel_warmup")
    runner = worker.model_runner
    batch = getattr(runner, "input_batch", None)
    table = getattr(batch, "block_table", None)
    if table is None or not hasattr(table, "block_tables"):
        logger.warning("Request kernel warmup skipped: unsupported block table API")
        return
    if batch.num_reqs:
        raise RuntimeError("Request kernel warmup must run before admitting requests")

    start = time.perf_counter()
    sizes = request_warmup_sizes(runner.scheduler_config.max_num_batched_tokens)
    # Dummy model forwards exercise the actual local heads, dtype and Q/K
    # strides. Do not guess these from global model configuration. Attention is
    # disabled, so no real KV pages or request state are updated.
    with torch.inference_mode():
        for size in sizes:
            runner._dummy_run(size, cudagraph_runtime_mode=CUDAGraphMode.NONE)

        # _dummy_run bypasses the real slot-mapping path. Exercise it separately
        # with initialized scratch rows, preserving the runner's buffers.
        saved = [
            (bt, bt.block_table.gpu[0].clone(), bt.slot_mapping.gpu.clone())
            for bt in table.block_tables
            if not bt.is_mamba_group
        ]
        try:
            for bt, _, _ in saved:
                bt.block_table.gpu[0].zero_()
            for size in sizes:
                query_start = torch.tensor(
                    [0, size], dtype=torch.int32, device=runner.device
                )
                positions = torch.zeros(size, dtype=torch.int64, device=runner.device)
                table.compute_slot_mapping(1, query_start, positions)
            torch.npu.synchronize()
        finally:
            for bt, row, slots in saved:
                bt.block_table.gpu[0].copy_(row)
                bt.slot_mapping.gpu.copy_(slots)
            torch.npu.synchronize()
    logger.info(
        "AREAL_REQUEST_KERNEL_WARMUP complete sizes=%s seconds=%.3f",
        sizes,
        time.perf_counter() - start,
    )


def install_ascend_kernel_warmup() -> None:
    """Extend vendor startup warmup; leave CPU/CUDA and disabled runs unchanged."""
    from areal.infra.platforms import current_platform

    if current_platform.device_type != "npu":
        return
    if os.environ.get("AREAL_ASCEND_KERNEL_WARMUP", "1") == "0":
        return
    module = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")

    original = module.kernel_warmup
    if getattr(original, "_areal_request_warmup", False):
        return

    @functools.wraps(original)
    def warmup(worker):
        result = original(worker)
        warmup_request_kernels(worker)
        return result

    warmup._areal_request_warmup = True
    module.kernel_warmup = warmup
