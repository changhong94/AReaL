# SPDX-License-Identifier: Apache-2.0
"""Selective Ascend memory pools for exclusive single-device time sharing."""

import os
from contextlib import contextmanager


def pin_camem_callbacks(allocator):
    """Retain native callback objects for the allocator singleton's lifetime."""
    allocator.python_malloc_callback = allocator.python_malloc_callback
    allocator.python_free_callback = allocator.python_free_callback


def adam_state_bytes(optimizer):
    """Conservative reservation including allocator rounding, for one actor rank."""
    raw = sum(
        (2 + int(group["amsgrad"])) * p.numel() * p.element_size() + 8
        for group in optimizer.param_groups
        for p in group["params"]
        if p.requires_grad and not optimizer.state.get(p)
    )
    return int(raw * 1.05) + (4 * 1024**2 if raw else 0)


def initialize_adam_states(optimizer):
    """Build unpublished states without gradients, weight decay, or a fake step."""
    import torch
    from torch.optim.adam import _device_dtype_check_for_fused, _get_scalar_dtype

    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("Optimizer prewarm supports standard AdamW only")
    states = {}
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad or optimizer.state.get(p):
                    continue
                if group["fused"]:
                    _device_dtype_check_for_fused(p)
                state = {
                    "step": (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=group["fused"]),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    ),
                    "exp_avg": torch.zeros_like(p, memory_format=torch.preserve_format),
                    "exp_avg_sq": torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    ),
                }
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                states[p] = state
    return states


class OptimizerPrewarm:
    """Prepare on a private stream; publish state only after the caller joins.

    All optimizer mutation and memory inventory/transition calls must wait().
    The inference workspace and device cap remain reserved while preparing.
    """

    def __init__(self, optimizer, backend, reserve_bytes, device_budget_bytes=None):
        from concurrent.futures import Future
        from threading import Lock, Thread

        self.optimizer = optimizer
        self.backend = backend
        self.reserve_bytes = reserve_bytes
        self.device_budget_bytes = device_budget_bytes
        self._future = Future()
        self._published = False
        self._publish_lock = Lock()
        self.device = optimizer.param_groups[0]["params"][0].device
        self.ready = backend.torch.npu.Event()
        self.ready.record(backend.torch.npu.current_stream(self.device))
        self._thread = Thread(target=self._run, name="optimizer-prewarm")
        self._thread.start()

    def _run(self):
        import time

        from areal.utils import request_trace

        stream = None
        start = time.perf_counter()
        try:
            npu = self.backend.torch.npu
            npu.set_device(self.device)
            stream = npu.Stream(device=self.device)
            stream.wait_event(self.ready)
            free, total = self.backend.free_memory()
            free_before = free
            if self.device_budget_bytes is not None:
                free = min(free, self.device_budget_bytes - (total - free))
            needed = adam_state_bytes(self.optimizer)
            allocate = free >= self.reserve_bytes + needed
            request_trace.emit(
                "optimizer_prewarm_begin",
                state_bytes=needed,
                reserve_bytes=self.reserve_bytes,
                available_bytes=free,
                allocate_states=allocate,
            )
            with npu.stream(stream), self.backend.allocation_scope("actor_optimizer"):
                states = initialize_adam_states(self.optimizer) if allocate else {}
                stream.synchronize()
            request_trace.emit(
                "optimizer_prewarm_end",
                allocated_states=allocate,
                free_bytes_before=free_before,
                free_bytes_after=self.backend.free_memory()[0],
                parameters=len(states),
                seconds=time.perf_counter() - start,
            )
            self._future.set_result(states)
        except BaseException as exc:
            try:
                if stream is not None:
                    stream.synchronize()
            finally:
                self._future.set_exception(exc)

    def wait(self):
        self._thread.join()
        with self._publish_lock:
            if not self._published:
                states = self._future.result()
                for param in states:
                    if self.optimizer.state.get(param):
                        raise RuntimeError("Optimizer state changed during prewarm")
                self.optimizer.state.update(states)
                self._published = True
                # Do not retain old moments if a later checkpoint replaces them.
                self._future = None
                self.optimizer = None


def adaptive_memory_enabled() -> bool:
    return os.getenv("AREAL_AUTO_MEMORY", "0") == "1"


class TaggedMemory:
    """Stable-address, allocation-granular residency for quiescent engines.

    Missing allocations are tracked individually. A partially resident tag is
    never safe to execute until restored. Failed transitions are terminal.
    """

    def __init__(self, backend, tags):
        self.backend = backend
        self.states = dict.fromkeys(tags, "mapped")
        self.broken = False
        self._missing = {}
        backend.residency = self

    def inventory(self):
        data = self.backend.inventory()
        if any(a.tag not in self.states for a in data.values()):
            raise RuntimeError("Unknown allocation in adaptive memory pool")
        for address, (allocation, _) in self._missing.items():
            if data.get(address) is not allocation:
                self.broken = True
                raise RuntimeError("Unmapped allocation changed before restoration")
        return data

    def snapshot(self):
        if self.broken:
            raise RuntimeError("Adaptive memory transition previously failed")
        sizes = dict.fromkeys(self.states, 0)
        resident = dict.fromkeys(self.states, 0)
        backup_bytes = 0
        data = self.inventory()
        for address, a in data.items():
            sizes[a.tag] += a.handle[1]
            if address not in self._missing:
                resident[a.tag] += a.handle[1]
            if a.cpu_backup_tensor is not None:
                backup_bytes += a.handle[1]
        for tag in self.states:
            modes = {mode for a, mode in self._missing.values() if a.tag == tag}
            self.states[tag] = (
                "mapped"
                if not modes
                else "partial"
                if resident[tag] or len(modes) > 1
                else next(iter(modes))
            )
        free, total = self.backend.free_memory()
        return {
            "states": dict(self.states),
            "bytes": sizes,
            "resident_bytes": resident,
            "restore_bytes": sum(a.handle[1] for a, _ in self._missing.values()),
            "cpu_backup_bytes": backup_bytes,
            "free_bytes": free,
            "total_bytes": total,
            "supports_partial_reclaim": True,
        }

    def _change(self, allocations, mapped, preserve):
        import time

        from areal.utils import request_trace

        if not allocations:
            return self.snapshot()
        started = time.perf_counter()
        self.backend.synchronize()
        self.broken = True
        if not mapped and preserve:
            for a in allocations:
                a.cpu_backup_tensor = self.backend.backup(a)
        for a in allocations:
            address = a.handle[2]
            if mapped:
                self.backend.map(a)
                if a.cpu_backup_tensor is not None:
                    self.backend.restore(a)
            else:
                self.backend.unmap(a)
        self.backend.synchronize()
        for a in allocations:
            address = a.handle[2]
            if mapped:
                a.cpu_backup_tensor = None
                del self._missing[address]
            else:
                self._missing[address] = (a, "cpu" if preserve else "discarded")
        self.broken = False
        request_trace.emit(
            "memory_blocks_restore" if mapped else "memory_blocks_reclaim",
            bytes=sum(a.handle[1] for a in allocations),
            allocations=len(allocations),
            preserve=preserve,
            tags=sorted({a.tag for a in allocations}),
            seconds=time.perf_counter() - started,
        )
        return self.snapshot()

    def transition(self, tags, mapped: bool, preserve: bool = True):
        self.snapshot()
        if not set(tags).issubset(self.states):
            raise ValueError("Unsupported memory tag")
        data = self.inventory()
        if not mapped:
            mode = "cpu" if preserve else "discarded"
            if any(a.tag in tags and old != mode for a, old in self._missing.values()):
                raise RuntimeError("Cannot change an already unmapped allocation")
        selected = [
            a
            for address, a in data.items()
            if a.tag in tags
            and (
                (address in self._missing) if mapped else (address not in self._missing)
            )
        ]
        return self._change(selected, mapped, preserve)

    def reclaim(self, tags, target_free_bytes, preserve=True):
        """Evict only enough whole allocation blocks to reach a free-memory target.

        Tag order is eviction priority. Ascend CaMem owns physical handles at
        allocation granularity; overshoot is bounded by the last selected block.
        This method must never run concurrently with tensor use or allocation.
        """
        status = self.snapshot()
        if type(target_free_bytes) is not int or target_free_bytes < 0:
            raise ValueError("target_free_bytes must be a nonnegative integer")
        if not set(tags).issubset(self.states):
            raise ValueError("Unsupported memory tag")
        deficit = target_free_bytes - status["free_bytes"]
        if deficit <= 0:
            return status
        candidates = [
            a
            for address, a in self.inventory().items()
            if a.tag in tags and address not in self._missing
        ]
        candidates.sort(key=lambda a: (tags.index(a.tag), a.handle[1], a.handle[2]))
        selected, size = [], 0
        for a in candidates:
            selected.append(a)
            size += a.handle[1]
            if size >= deficit:
                break
        return self._change(selected, mapped=False, preserve=preserve)


class ParkedWorkspace:
    """Park only whole inactive allocator segments; never discard live contents.

    The actor is quiescent throughout park/restore. No allocator use or cache
    eviction is allowed while parked. Partial transitions are terminal.
    """

    tag = "actor_workspace"

    def __init__(self, backend):
        self.backend = backend
        self.parked = []
        self.broken = False

    @property
    def restore_bytes(self):
        return sum(a.handle[1] for a in self.parked)

    def check(self):
        if self.broken:
            raise RuntimeError("Workspace transition previously failed")

    def park(self, target_free_bytes=None):
        import time

        self.check()
        if self.parked and target_free_bytes is None:
            return
        if (
            target_free_bytes is not None
            and self.backend.free_memory()[0] >= target_free_bytes
        ):
            return
        started = time.perf_counter()
        self.backend.synchronize()
        segments = {s["address"]: s for s in self.backend.segments()}
        selected = []
        for a in self.backend.inventory().values():
            if a.tag != self.tag or any(a is old for old in self.parked):
                continue
            s = segments.get(a.handle[2])
            if s is None or s["total_size"] != a.handle[1]:
                raise RuntimeError("Workspace segment lacks exact allocator evidence")
            if (
                s["active_size"] == 0
                and s["allocated_size"] == 0
                and s["blocks"]
                and all(b["state"] == "inactive" for b in s["blocks"])
                and sum(b["size"] for b in s["blocks"]) == s["total_size"]
            ):
                if a.cpu_backup_tensor is not None:
                    raise RuntimeError("Workspace unexpectedly has a live backup")
                selected.append(a)
        selected.sort(key=lambda a: (a.handle[1], a.handle[2]))
        if target_free_bytes is not None:
            deficit = target_free_bytes - self.backend.free_memory()[0]
            kept, size = [], 0
            for a in selected:
                if size >= deficit:
                    break
                kept.append(a)
                size += a.handle[1]
            selected = kept
        self.broken = True
        for a in selected:
            self.backend.unmap(a)
            self.parked.append(a)
        self.backend.synchronize()
        self.broken = False
        from areal.utils import request_trace

        request_trace.emit(
            "workspace_park",
            bytes=self.restore_bytes,
            addresses=[a.handle[2] for a in selected],
            seconds=time.perf_counter() - started,
        )

    def restore(self):
        import time

        self.check()
        if not self.parked:
            return
        started = time.perf_counter()
        count, size = len(self.parked), self.restore_bytes
        self.broken = True
        for a in self.parked:
            if self.backend.inventory().get(a.handle[2]) is not a:
                raise RuntimeError("Parked workspace allocation changed")
            self.backend.map(a)
        self.backend.synchronize()
        self.parked.clear()
        self.broken = False
        from areal.utils import request_trace

        request_trace.emit(
            "workspace_restore",
            bytes=size,
            segments=count,
            seconds=time.perf_counter() - started,
        )


class AscendPoolBackend:
    """Thin adapter around the installed vLLM-Ascend CaMem primitives."""

    def __init__(self):
        import torch
        from vllm_ascend.device_allocator import camem

        if not camem.camem_available:
            raise RuntimeError("Adaptive memory requires Ascend CaMem")
        self.torch = torch
        self.camem = camem
        self.allocator = camem.CaMemAllocator.get_instance()
        if self.allocator.pointer_to_data:
            raise RuntimeError("Adaptive memory requires exclusive process-local CaMem")
        self.pools = {}
        self.workspace = (
            ParkedWorkspace(self)
            if os.getenv("AREAL_PARK_TRAIN_CACHE", "0") == "1"
            else None
        )
        # Native init_module borrows these exact Python objects without INCREF.
        # Pin bound methods on the singleton, including future pool creation and
        # frees after this backend is gone. An instance reference alone is not enough.
        pin_camem_callbacks(self.allocator)

    @contextmanager
    def allocation_scope(self, tag):
        residency = getattr(self, "residency", None)
        if residency is not None and (residency.broken or residency._missing):
            raise RuntimeError("Restore pool allocations before allocating")
        if self.workspace is not None:
            self.workspace.check()
            if self.workspace.parked:
                raise RuntimeError("Cannot allocate while workspace is parked")
        if tag not in self.pools:
            with self.allocator.use_memory_pool(tag=tag):
                yield
            self.pools[tag] = self.allocator.allocator_and_pools[tag]
        else:
            # Reuse pools: a new pool each optimizer step loses lifetime ownership.
            pool, _ = self.pools[tag]
            with self.allocator.use_allocation_tag(tag):
                with self.torch.npu.memory.use_mem_pool(pool):
                    yield

    def inventory(self):
        return self.allocator.pointer_to_data

    def synchronize(self):
        self.torch.npu.synchronize()

    def free_memory(self):
        return self.torch.npu.mem_get_info()

    def segments(self):
        return self.torch.npu.memory._snapshot()["segments"]

    def clear_cache(self):
        residency = getattr(self, "residency", None)
        if residency is not None and (residency.broken or residency._missing):
            raise RuntimeError("Cannot evict CPU-backed allocations; restore first")
        if self.workspace is not None:
            self.workspace.check()
            if self.workspace.parked:
                raise RuntimeError("Cannot evict parked workspace; restore first")
        self.synchronize()
        self.torch.npu.empty_cache()

    def _copy(self, dst, src, size, kind):
        code = self.camem.memcpy(dst, size, src, size, kind)
        if code != 0:
            raise RuntimeError(f"Ascend memory copy failed: {code}")

    def backup(self, allocation):
        size = allocation.handle[1]
        tensor = self.torch.empty(
            size, dtype=self.torch.uint8, device="cpu", pin_memory=True
        )
        self._copy(tensor.data_ptr(), allocation.handle[2], size, 2)
        return tensor

    def restore(self, allocation):
        self._copy(
            allocation.handle[2],
            allocation.cpu_backup_tensor.data_ptr(),
            allocation.handle[1],
            1,
        )

    def unmap(self, allocation):
        self.camem.unmap_and_release(allocation.handle)

    def map(self, allocation):
        self.camem.create_and_map(allocation.handle)

    def assert_covered(self, tensors):
        ranges = [
            (a.handle[2], a.handle[2] + a.handle[1]) for a in self.inventory().values()
        ]
        for tensor in tensors:
            local = getattr(tensor, "_local_tensor", tensor)
            if local.device.type != "npu" or local.numel() == 0:
                continue
            start = local.untyped_storage().data_ptr()
            end = start + local.untyped_storage().nbytes()
            if not any(lo <= start and end <= hi for lo, hi in ranges):
                raise RuntimeError(
                    f"Persistent tensor storage is outside the adaptive pool: shape={tuple(local.shape)}, "
                    f"dtype={local.dtype}, bytes={end - start}, address={start}, allocations={len(ranges)}"
                )
