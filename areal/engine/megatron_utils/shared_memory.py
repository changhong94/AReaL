# SPDX-License-Identifier: Apache-2.0
"""Stable-address memory ownership for single-rank Megatron NPU sharing.

The engine must allocate model/DDP buffers under ``actor_weights``, construct
and step its optimizer under ``actor_optimizer``. Forward/backward temporary
workspace uses the ordinary caching allocator so pressure-triggered cache release
can reclaim it safely. Call transitions only at a quiescent boundary after
joining snapshot publication. DDP gradient buffers are persistent allocations,
not disposable gradient caches: their addresses and contents survive eviction.
"""

from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any


def _tensors(value: Any) -> Iterator[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def optimizer_leaves(optimizer: Any) -> Iterator[Any]:
    """Visit Megatron chained optimizer leaves without importing Megatron."""
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        if optimizer is not None:
            yield optimizer
    else:
        for child in children:
            yield from optimizer_leaves(child)


def persistent_tensors(engine: Any) -> list[Any]:
    """Include DDP flat storage, master parameters and lazy optimizer state."""
    tensors = []
    for model in engine.model:
        tensors.extend(model.parameters())
        # Megatron DDP shadows nn.Module.buffers() with its flat buffer list.
        buffers = getattr(model, "buffers", None)
        if callable(buffers):
            tensors.extend(buffers())
        else:
            module = getattr(model, "module", None)
            if module is not None:
                tensors.extend(module.buffers())
        for name in ("buffers", "expert_parallel_buffers"):
            buffers = getattr(model, name, ())
            if callable(buffers):
                continue
            for buffer in buffers:
                for attr in ("param_data", "grad_data"):
                    tensors.extend(_tensors(getattr(buffer, attr, None)))
        for param in model.parameters():
            tensors.extend(_tensors(getattr(param, "main_grad", None)))
            tensors.extend(_tensors(param.grad))
    for wrapper in optimizer_leaves(engine.optimizer):
        # Scalar overflow flags and loss-scale tensors are persistent too.
        tensors.extend(_tensors(vars(wrapper)))
        for name in (
            "fp32_from_float16_groups",
            "shard_fp32_from_float16_groups",
            "fp32_from_fp32_groups",
            "shard_fp32_from_fp32_groups",
            "float16_groups",
            "shard_float16_groups",
        ):
            tensors.extend(_tensors(getattr(wrapper, name, None)))
        inner = getattr(wrapper, "optimizer", wrapper)
        if inner is not None:
            for group in inner.param_groups:
                tensors.extend(group["params"])
                for param in group["params"]:
                    tensors.extend(_tensors(param.grad))
            tensors.extend(_tensors(inner.state))
        tensors.extend(
            _tensors(getattr(wrapper, "grad_scaler", None).__dict__)
            if getattr(wrapper, "grad_scaler", None) is not None
            else []
        )
    return list({id(tensor): tensor for tensor in tensors}.values())


def future_optimizer_bytes(optimizer: Any) -> int:
    """Reserve missing Adam moments; master parameters already exist at init.

    Megatron's low-precision wrapper owns FP32 master parameters, so reserve at
    least FP32 moments even when the underlying parameter is lower precision.
    Do not invoke init_state_fn or a fake optimizer step: fused optimizer state
    formats differ between MindSpeed/Megatron versions.
    """
    total = 0
    seen = set()
    for wrapper in optimizer_leaves(optimizer):
        inner = getattr(wrapper, "optimizer", wrapper)
        if inner is None:
            continue
        for group in inner.param_groups:
            for param in group["params"]:
                # Master copies can be detached tensors with requires_grad=False;
                # Megatron assigns their gradients explicitly before stepping.
                if id(param) in seen:
                    continue
                seen.add(id(param))
                state = inner.state.get(param, {})
                width = max(4, param.element_size())
                for name in ("exp_avg", "exp_avg_sq"):
                    if name not in state:
                        total += param.numel() * width
                if group.get("amsgrad", False) and "max_exp_avg_sq" not in state:
                    total += param.numel() * width
                if "step" not in state:
                    total += 8
    return int(total * 1.05) + (4 * 1024**2 if total else 0)


class MegatronSharedMemory:
    """Delegate memory transitions while leaving Megatron tensor views intact."""

    def __init__(self, engine: Any, backend: Any = None, memory: Any = None):
        from areal.engine.colocation_memory import AscendPoolBackend, TaggedMemory

        self.engine = engine
        self.backend = backend if backend is not None else AscendPoolBackend()
        self.memory = (
            memory
            if memory is not None
            else TaggedMemory(
                self.backend, ["actor_weights", "actor_optimizer", "actor_workspace"]
            )
        )

    def allocation_scope(self, tag: str):
        """Keep temporary workspace reclaimable without retiring private pools.

        torch_npu 2.10.0.post4 allocates private-pool segments through the
        pluggable allocator but releases them through aclrtFree, bypassing the
        CaMem callback. Ordinary workspace caches can be reclaimed safely by
        empty_cache under pressure. Persistent model/DDP and optimizer storage
        retain managed addresses. Workspace address parking is consequently a
        no-op for this backend; no workspace allocations enter the managed pool.
        """
        if tag == "actor_workspace":
            residency = getattr(self.backend, "residency", None)
            if residency is not None and (residency.broken or residency._missing):
                raise RuntimeError("Restore pool allocations before allocating")
            return nullcontext()
        return self.backend.allocation_scope(tag)

    def action(
        self, action: str = "status", target_free_bytes: int | None = None
    ) -> dict:
        """Run after publication has joined; never touch an unmapped tensor."""
        if action not in {
            "status",
            "finish_train",
            "release_cache",
            "park_cache",
            "offload",
            "restore",
            "reclaim",
        }:
            raise ValueError("Unsupported actor memory action")
        if action == "reclaim" and (
            type(target_free_bytes) is not int or target_free_bytes < 0
        ):
            raise ValueError("reclaim requires a nonnegative free-memory target")
        engine = self.engine
        workspace = getattr(self.backend, "workspace", None)
        if workspace is not None:
            workspace.check()
        if action == "restore":
            if workspace is not None:
                workspace.restore()
            self.memory.transition(["actor_weights", "actor_optimizer"], mapped=True)
            engine.is_offload = False
        if action == "finish_train":
            if engine.is_offload:
                raise RuntimeError("Cannot finish training on an offloaded actor")
            # zero_grad_buffer writes live contiguous storage. Never call it on
            # parked pools, and never resize its storage to discard gradients.
            engine.optimizer_zero_grad()
            self.backend.synchronize()
        if not engine.is_offload:
            self.backend.assert_covered(persistent_tensors(engine))
        if action == "reclaim":
            if (
                self.backend.free_memory()[0] < target_free_bytes
                and not self.memory.snapshot()["restore_bytes"]
                and not (workspace is not None and workspace.parked)
            ):
                # Reclaim ordinary disposable workspace before evicting live state.
                self.backend.clear_cache()
            if not engine.is_offload:
                engine.optimizer_zero_grad()
            if workspace is not None:
                workspace.park(target_free_bytes)
            status = self.memory.reclaim(
                ["actor_optimizer", "actor_weights"], target_free_bytes, preserve=True
            )
            engine.is_offload = status["restore_bytes"] > 0
        if action == "park_cache":
            if workspace is None:
                raise RuntimeError("Workspace parking is not enabled")
            self.backend.clear_cache()
            workspace.park()
        if action == "release_cache":
            if self.memory.snapshot()["restore_bytes"]:
                raise RuntimeError("Restore actor allocations before clearing cache")
            self.backend.clear_cache()
        if action == "offload":
            if not engine.is_offload:
                engine.optimizer_zero_grad()
            if (
                workspace is None or not workspace.parked
            ) and not self.memory.snapshot()["restore_bytes"]:
                self.backend.clear_cache()
            self.memory.transition(
                ["actor_weights", "actor_optimizer"], mapped=False, preserve=True
            )
            engine.is_offload = True
        status = self.memory.snapshot()
        if workspace is not None:
            status["workspace_parked_bytes"] = workspace.restore_bytes
            status["restore_bytes"] += workspace.restore_bytes
        # main_grad is already allocated in the DDP flat buffer. Reserve only
        # parameters without this persistent allocation, avoiding double count.
        status["gradient_bytes"] = sum(
            param.numel() * max(4, param.element_size())
            for model in engine.model
            for param in model.parameters()
            if param.requires_grad and getattr(param, "main_grad", None) is None
        )
        status["future_optimizer_bytes"] = future_optimizer_bytes(engine.optimizer)
        status["optimizer_id"] = id(engine.optimizer)
        engine.logger.info("AUTO_MEMORY_ACTOR %s", status)
        return status
