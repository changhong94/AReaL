# SPDX-License-Identifier: Apache-2.0
"""Storage invariants for persistent Ascend inference graphs.

Graph workspaces remain in the vendor allocator, outside reclaimable weight/KV
pools. Remapping physical backing is permitted only with stable virtual storage
and the existing paused-engine/device fences.
"""

from functools import wraps


def full_graph_config(max_num_batched_tokens=2048, max_num_seqs=64):
    """Bound capture count while covering prefill through the scheduler limit."""
    if max_num_batched_tokens < 1 or max_num_seqs < 1:
        raise ValueError("Graph token and sequence limits must be positive")
    dense_limit = min(max_num_batched_tokens, max_num_seqs * 2, 512)
    sizes = {n for n in (1, 2, 4) if n <= max_num_batched_tokens}
    sizes.update(range(8, dense_limit + 1, 8))
    bucket = 128
    while bucket < max_num_batched_tokens:
        sizes.add(bucket)
        bucket *= 2
    sizes.add(max_num_batched_tokens)
    return {
        "cudagraph_mode": "FULL",
        "cudagraph_capture_sizes": sorted(sizes),
        "max_cudagraph_capture_size": max_num_batched_tokens,
    }


def storage_signature(model):
    return {
        name: (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
        )
        for kind, items in (
            ("parameter", model.named_parameters()),
            ("buffer", model.named_buffers()),
        )
        for name, tensor in ((kind + ":" + name, tensor) for name, tensor in items)
    }


def runner_storage_signature(runner):
    signature = storage_signature(runner.model)
    pool = getattr(runner, "_areal_dynamic_kv", None)
    if pool is not None:
        for i, tensor in enumerate(pool.tensors):
            signature[f"dynamic_kv:{i}"] = (
                tensor.data_ptr(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                str(tensor.dtype),
            )
    return signature


def assert_graph_storage(runner):
    expected = getattr(runner, "_areal_graph_storage", None)
    if expected is None:
        return
    actual = runner_storage_signature(runner)
    changed = sorted(
        k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)
    )
    if changed:
        runner._areal_graph_broken = True
        raise RuntimeError(f"Captured inference graph storage changed: {changed[:8]}")
    if getattr(runner, "_areal_graph_broken", False):
        raise RuntimeError("Inference graph is invalid; restart the worker")


def assert_graph_storage_collective(runner):
    """All TP members fail before any peer enters a weight-update collective."""
    if runner.model_config.enforce_eager:
        return

    import torch.distributed as dist
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    if group.world_size == 1:
        return assert_graph_storage(runner)
    error = None
    try:
        assert_graph_storage(runner)
    except RuntimeError as exc:
        error = str(exc)
    errors = [None] * group.world_size
    dist.all_gather_object(errors, error, group=group.cpu_group)
    if any(errors):
        runner._areal_graph_broken = True
        raise RuntimeError(f"TP graph storage validation failed: {errors}")


def install_graph_guard(cls):
    if getattr(cls, "_areal_graph_guard", False):
        return
    original = cls.capture_model

    @wraps(original)
    def capture(self, *args, **kwargs):
        pool = getattr(self, "_areal_dynamic_kv", None)
        if pool is not None:
            if (
                self.requests
                or pool.live != {0}
                or pool.swapped
                or getattr(self, "_areal_kv_import_pins", {})
            ):
                raise RuntimeError("Graph capture requires idle, restored dynamic KV")
            pool.backend.synchronize()
            pool.reconcile({0}, trim=True)
            for table in self.input_batch.block_table.block_tables:
                table.block_table.cpu.zero_()
                table.block_table.gpu.zero_()
                table.slot_mapping.cpu.fill_(-1)
                table.slot_mapping.gpu.fill_(-1)
            pool.backend.synchronize()
        result = original(self, *args, **kwargs)
        if pool is not None:
            pool.backend.synchronize()
        if not self.model_config.enforce_eager:
            self._areal_graph_storage = runner_storage_signature(self)
            self._areal_graph_broken = False
        return result

    original_dispatch = cls._determine_batch_execution_and_padding

    @wraps(original_dispatch)
    def dispatch(self, *args, **kwargs):
        result = original_dispatch(self, *args, **kwargs)
        if self.requests:
            max_query = kwargs.get(
                "max_num_scheduled_tokens", args[3] if len(args) > 3 else 1
            )
            phase = "DECODE" if max_query == 1 else "PREFILL_MIXED"
            mode = getattr(result[0], "name", str(result[0]))
            seen = getattr(self, "_areal_graph_dispatch_logged", set())
            key = (phase, mode)
            if key not in seen:
                from vllm.logger import init_logger

                init_logger("vllm.areal_graph").info(
                    "AREAL_GRAPH_%s active_requests=%s mode=%s",
                    phase,
                    len(self.requests),
                    mode,
                )
                seen.add(key)
                self._areal_graph_dispatch_logged = seen
        return result

    cls._determine_batch_execution_and_padding = dispatch

    original_execute = cls.execute_model

    @wraps(original_execute)
    def execute(self, *args, **kwargs):
        if getattr(self, "_areal_graph_broken", False):
            raise RuntimeError("Inference graph is invalid; restart the worker")
        return original_execute(self, *args, **kwargs)

    cls.execute_model = execute
    cls.capture_model = capture
    cls._areal_graph_guard = True
