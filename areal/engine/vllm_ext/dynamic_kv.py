# SPDX-License-Identifier: Apache-2.0
"""Bounded, sparse physical backing for the pinned Ascend full-attention layout."""

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path


def enabled():
    return os.getenv("AREAL_DYNAMIC_KV", "0") == "1"


class SparseKVPages:
    """Keep every retained request mapped; only evict wholly inactive pages.

    Addresses and block IDs never change. All topology changes fence prior device
    work. Budget rejection occurs before mapping; a native failure is terminal.
    """

    def __init__(
        self,
        backend,
        tensors,
        bytes_per_block,
        num_blocks,
        reserve_bytes,
        keep_idle_pages=1,
    ):
        if (
            bytes_per_block <= 0
            or num_blocks < 2
            or reserve_bytes < 0
            or keep_idle_pages < 0
        ):
            raise ValueError("Invalid dynamic KV geometry or budget")
        self.backend = backend
        self.tensors = tensors
        self.bytes_per_block = bytes_per_block
        self.num_blocks = num_blocks
        self.reserve_bytes = reserve_bytes
        self.capacity_blocks = num_blocks
        self.keep_idle_pages = keep_idle_pages
        self.page = backend.page
        if self.page % bytes_per_block:
            raise ValueError("KV block must divide physical page size")
        if any(t.data_ptr() % self.page for t in tensors):
            raise ValueError("KV raw tensor must begin on a physical page")
        self.blocks_per_page = self.page // bytes_per_block
        self.mapped = set()
        self.live = {0}
        self.broken = False
        self.paused = False
        self.swapped = {}
        self.reconcile({0})

    def set_budget(self, budget_bytes):
        """Bound physical pages and scheduler IDs while idle or preserved-paused."""
        if (self.live != {0} and not self.paused) or self.swapped or self.broken:
            raise RuntimeError(
                "KV budget changes require idle or preserved-paused mappings"
            )
        if budget_bytes is None:
            capacity = self.num_blocks
        else:
            if type(budget_bytes) is not int or budget_bytes < 0:
                raise ValueError("Invalid KV byte budget")
            pages = budget_bytes // (len(self.tensors) * self.page)
            # Keep the final padded page out of the bounded range.
            capacity = min(self.num_blocks - 1, pages * self.blocks_per_page)
            if capacity < max(2, max(self.live) + 1):
                return 0
        self.reconcile(self.live, trim=True)
        self.capacity_blocks = capacity
        return capacity

    def reconcile(self, block_ids, trim=False):
        if self.broken:
            raise RuntimeError("Dynamic KV is unusable after a mapping failure")
        if self.swapped:
            raise RuntimeError("Restore retained KV before changing live blocks")
        ids = set(block_ids) | {0}
        if any(type(b) is not int or b < 0 or b >= self.capacity_blocks for b in ids):
            raise ValueError("Invalid KV block ID")
        live = {b // self.blocks_per_page for b in ids}
        # Ascend kernels may access allocator padding beyond the tensor end.
        # Reserve an extra page and commit it only when the final block is live.
        if self.num_blocks - 1 in ids:
            live.update(
                range(
                    (self.num_blocks * self.bytes_per_block - 1) // self.page,
                    (self.num_blocks * self.bytes_per_block + 31) // self.page + 1,
                )
            )
        idle = sorted(self.mapped - live)
        target = live | set(idle[: 0 if trim else self.keep_idle_pages])
        remove = self.mapped - target
        add = target - self.mapped
        if not remove and not add:
            self.live = ids
            return
        self.backend.synchronize()
        # No live request depends on these whole pages.
        try:
            for index in sorted(remove):
                for tensor in self.tensors:
                    self.backend.unmap(tensor.data_ptr() + index * self.page)
                self.mapped.remove(index)
            required = len(add) * len(self.tensors) * self.page
            # Pure reclamation must remain possible under memory pressure.
            if add and self.backend.free_bytes() < required + self.reserve_bytes:
                raise MemoryError(
                    "Dynamic KV physical budget exhausted before execution"
                )
            for index in sorted(add):
                for tensor in self.tensors:
                    self.backend.map(tensor.data_ptr() + index * self.page)
                self.mapped.add(index)
            self.backend.synchronize()
            self.live = ids
        except BaseException:
            self.broken = True
            raise

    def pause_preserved(self):
        if self.broken:
            raise RuntimeError("Cannot pause broken dynamic KV")
        self.backend.synchronize()
        self.paused = True

    def reclaim_preserved(self, target_free_bytes):
        """Evict only enough page groups, with live contents backed by host RAM."""
        if self.broken or not self.paused:
            raise RuntimeError("Live KV reclaim requires an acknowledged pause")
        if type(target_free_bytes) is not int or target_free_bytes < 0:
            raise ValueError("Invalid free-memory target")
        if self.backend.free_bytes() >= target_free_bytes:
            return
        # Drop disposable idle pages first; never forget the live block inventory.
        if not self.swapped:
            self.reconcile(self.live, trim=True)
        self.backend.synchronize()
        try:
            for index in sorted(self.mapped - {0}, reverse=True):
                if self.backend.free_bytes() >= target_free_bytes:
                    break
                addresses = [t.data_ptr() + index * self.page for t in self.tensors]
                backups = [self.backend.backup(a) for a in addresses]
                for address in addresses:
                    self.backend.unmap(address)
                self.swapped[index] = backups
                self.mapped.remove(index)
            self.backend.synchronize()
        except BaseException:
            self.broken = True
            raise

    def restore_preserved(self):
        if self.broken:
            raise RuntimeError("Cannot restore broken dynamic KV")
        if not self.swapped:
            return
        if not self.paused:
            raise RuntimeError("Retained KV restoration requires a pause")
        self.backend.synchronize()
        try:
            required = len(self.swapped) * len(self.tensors) * self.page
            if self.backend.free_bytes() < required + self.reserve_bytes:
                raise MemoryError("Insufficient budget to restore retained KV")
            for index, backups in self.swapped.items():
                for tensor, backup in zip(self.tensors, backups, strict=True):
                    address = tensor.data_ptr() + index * self.page
                    self.backend.map(address)
                    self.backend.restore(address, backup)
                self.mapped.add(index)
            self.backend.synchronize()
            self.swapped.clear()
        except BaseException:
            self.broken = True
            raise

    def copy_retained_blocks(self, blocks, destination):
        """Pack paused request bytes without remapping any evicted physical page.

        The engine utility dispatcher and transition lock serialize this read
        against reclaim/discard. Destination is a bounded device staging buffer.
        """
        if self.broken or not self.paused:
            raise RuntimeError("Retained export requires an acknowledged healthy pause")
        if not blocks or not set(blocks).issubset(self.live - {0}):
            raise ValueError("Retained export contains non-live blocks")
        sources = []
        for tensor_index, tensor in enumerate(self.tensors):
            for block in blocks:
                page, offset = divmod(block * self.bytes_per_block, self.page)
                backups = self.swapped.get(page)
                if backups is not None:
                    address = backups[tensor_index].data_ptr() + offset
                    host = True
                elif page in self.mapped:
                    address = tensor.data_ptr() + block * self.bytes_per_block
                    host = False
                else:
                    raise RuntimeError("Retained KV page has no physical backing")
                sources.append((address, host))
        self.backend.synchronize()
        for index, (address, host) in enumerate(sources):
            self.backend.copy_to_device(
                destination + index * self.bytes_per_block,
                address,
                self.bytes_per_block,
                host=host,
            )
        self.backend.synchronize()

    def discard_preserved(self):
        """Drop retained bytes only after the scheduler acknowledged all aborts."""
        if self.broken or not self.paused:
            raise RuntimeError("Retained discard requires a healthy paused pool")
        self.backend.synchronize()
        self.swapped.clear()
        self.reconcile({0}, trim=True)
        self.paused = False

    def resume_preserved(self):
        self.restore_preserved()
        self.backend.synchronize()
        self.paused = False

    def snapshot(self):
        return dict(
            virtual_bytes=len(self.tensors) * self.num_blocks * self.bytes_per_block,
            mapped_bytes=len(self.mapped) * len(self.tensors) * self.page,
            mapped_slabs=len(self.mapped),
            live_blocks=len(self.live - {0}),
            physical_page_bytes=self.page,
            reserve_bytes=self.reserve_bytes,
            capacity_blocks=self.capacity_blocks,
            bytes_per_block=self.bytes_per_block,
            tensor_count=len(self.tensors),
            blocks_per_page=self.blocks_per_page,
            broken=self.broken,
            paused=self.paused,
            cpu_backup_bytes=len(self.swapped) * len(self.tensors) * self.page,
            swapped_slabs=len(self.swapped),
        )


def coordinated_state_call(pool, function):
    """Propagate rank-local state failures before a peer enters a collective."""
    import torch.distributed as dist
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    if group.world_size == 1:
        return function()
    result, error = None, None
    try:
        result = function()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * group.world_size
    dist.all_gather_object(errors, error, group=group.cpu_group)
    if any(errors):
        pool.broken = True
        raise RuntimeError(f"TP KV state update failed: {errors}")
    return result


def coordinated_reconcile(pool, block_ids):
    """Fence TP peers before mapping and before any following model collective.

    Physical residency is local: each member may map a different number of pages.
    Logical block IDs and execution readiness must agree across the instance.
    """
    import torch.distributed as dist
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    if group.world_size == 1:
        return pool.reconcile(block_ids)
    ids = set(block_ids) | {0}
    error = None
    try:
        if pool.broken or pool.swapped:
            raise RuntimeError("KV pages must be restored before TP execution")
        if any(type(b) is not int or b < 0 or b >= pool.capacity_blocks for b in ids):
            raise ValueError("Invalid TP block ID")
        target = {b // pool.blocks_per_page for b in ids}
        if pool.num_blocks - 1 in ids:
            target.update(
                range(
                    (pool.num_blocks * pool.bytes_per_block - 1) // pool.page,
                    (pool.num_blocks * pool.bytes_per_block + 31) // pool.page + 1,
                )
            )
        growth = len(target - pool.mapped) * len(pool.tensors) * pool.page
        reclaimable = (
            max(0, len(pool.mapped - target) - pool.keep_idle_pages)
            * len(pool.tensors)
            * pool.page
        )
        if (
            growth
            and pool.backend.free_bytes() + reclaimable < growth + pool.reserve_bytes
        ):
            raise MemoryError(
                "TP member cannot fit the next KV mapping within its budget"
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    states = [None] * group.world_size
    digest = hashlib.sha256(str(sorted(ids)).encode()).hexdigest()
    dist.all_gather_object(states, (error, digest), group=group.cpu_group)
    if any(item[0] for item in states) or len({item[1] for item in states}) != 1:
        pool.broken = True
        raise RuntimeError(f"TP KV preflight rejected execution: {states}")
    error = None
    try:
        pool.reconcile(ids)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    states = [None] * group.world_size
    dist.all_gather_object(states, error, group=group.cpu_group)
    if any(states):
        pool.broken = True
        raise RuntimeError(
            f"TP KV mapping failed; instance must remain stopped: {states}"
        )


def load_library():
    """Compile the small ACL-only allocator once, outside the source checkout."""
    import fcntl

    source = Path(__file__).with_name("dynamic_kv_allocator.cpp")
    home = Path(
        os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    )
    override = os.getenv("AREAL_DYNAMIC_KV_LIBRARY")
    if override:
        return Path(override)
    digest = hashlib.sha256(source.read_bytes() + str(home).encode()).hexdigest()[:16]
    folder = Path(tempfile.gettempdir()) / ("areal-dynamic-kv-" + str(os.getuid()))
    folder.mkdir(mode=0o700, exist_ok=True)
    target = folder / (digest + ".so")
    with (folder / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            temporary = target.with_suffix(".building.so")
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-O2",
                    "-shared",
                    "-fPIC",
                    "-pthread",
                    str(source),
                    "-I" + str(home / "include"),
                    "-L" + str(home / "lib64"),
                    "-lascendcl",
                    "-Wl,-rpath," + str(home / "lib64"),
                    "-o",
                    str(temporary),
                ],
                check=True,
            )
            temporary.replace(target)
    return target


class AscendSparseBackend:
    def __init__(self, device):
        import torch

        self.torch = torch
        self.device = device
        self.lib = ctypes.CDLL(str(load_library()))
        self.lib.dkv_granularity.argtypes = [ctypes.c_int]
        self.lib.dkv_granularity.restype = ctypes.c_size_t
        self.lib.dkv_map.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        self.lib.dkv_map.restype = ctypes.c_int
        self.lib.dkv_unmap.argtypes = [ctypes.c_size_t]
        self.lib.dkv_unmap.restype = ctypes.c_int
        self.lib.dkv_copy.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.dkv_copy.restype = ctypes.c_int
        self.page = self.lib.dkv_granularity(device.index)
        if not self.page:
            raise RuntimeError("Failed to query Ascend allocation granularity")
        self.allocator = torch.npu.memory.NPUPluggableAllocator(
            str(load_library()), "dkv_malloc", "dkv_free"
        )
        self.pool = torch.npu.memory.MemPool(self.allocator._allocator)

    def allocate(self, size):
        with self.torch.npu.memory.use_mem_pool(self.pool):
            return self.torch.empty(
                size + self.page, dtype=self.torch.int8, device=self.device
            )[:size]

    def synchronize(self):
        self.torch.npu.synchronize(self.device)

    def free_bytes(self):
        free, total = self.torch.npu.mem_get_info(self.device)
        cap = int(os.getenv("AREAL_DYNAMIC_KV_DEVICE_BUDGET_BYTES", "0"))
        return min(free, cap - (total - free)) if cap else free

    def map(self, address):
        code = self.lib.dkv_map(address, self.page)
        if code:
            raise RuntimeError(f"Dynamic KV map failed: {code}")

    def backup(self, address):
        tensor = self.torch.empty(
            self.page, dtype=self.torch.uint8, device="cpu", pin_memory=True
        )
        code = self.lib.dkv_copy(tensor.data_ptr(), address, self.page, 2)
        if code:
            raise RuntimeError(f"Dynamic KV host backup failed: {code}")
        return tensor

    def restore(self, address, backup):
        code = self.lib.dkv_copy(address, backup.data_ptr(), self.page, 1)
        if code:
            raise RuntimeError(f"Dynamic KV host restore failed: {code}")

    def copy_to_device(self, destination, source, size, *, host):
        code = self.lib.dkv_copy(destination, source, size, 1 if host else 3)
        if code:
            raise RuntimeError(f"Dynamic KV export copy failed: {code}")

    def unmap(self, address):
        code = self.lib.dkv_unmap(address)
        if code:
            raise RuntimeError(f"Dynamic KV unmap failed: {code}")


def flatten_blocks(groups):
    if groups is None:
        return set()
    result = set()
    for group in groups:
        if isinstance(group, int):
            result.add(group)
        elif group is not None:
            result.update(flatten_blocks(group))
    return result


def pending_blocks(requests, output):
    if output.kv_cache_block_copies:
        raise ValueError("Dynamic KV does not support block copying")
    ids = set()
    for state in requests.values():
        ids.update(flatten_blocks(state.block_ids))
    for state in output.scheduled_new_reqs:
        ids.update(flatten_blocks(state.block_ids))
    ids.update(flatten_blocks(output.scheduled_cached_reqs.new_block_ids))
    ids.update(flatten_blocks(output.new_block_ids_to_zero))
    return ids


def allocate_dynamic_kv(runner, config):
    import json
    import logging

    runner.hybrid_with_attn_and_mamba = False
    spec = config.kv_cache_groups[0].kv_cache_spec
    backend = AscendSparseBackend(runner.device)
    raw = {}
    tensors = []
    # Restrict to the native equal-sized K/V full-attention layout verified by
    # the caller. The vendor reshaper produces views without touching storage.
    for desc in config.kv_cache_tensors:
        if desc.size != config.num_blocks * spec.page_size_bytes:
            raise ValueError("Unexpected full-attention KV tensor size")
        pair = tuple(backend.allocate(desc.size // 2) for _ in range(2))
        tensors.extend(pair)
        for name in desc.shared_by:
            if name in raw:
                raise ValueError("Duplicate dynamic KV layer")
            k, v = runner._get_attention_kv_cache_dims(name, spec)
            if k != v:
                raise ValueError("Dynamic KV requires identical K/V head dimensions")
            raw[name] = pair
    pool = SparseKVPages(
        backend,
        tensors,
        spec.page_size_bytes // 2,
        config.num_blocks,
        int(os.getenv("AREAL_DYNAMIC_KV_RESERVE_BYTES", str(4 * 1024**3))),
        int(os.getenv("AREAL_DYNAMIC_KV_KEEP_IDLE_PAGES", "1")),
    )
    runner._areal_dynamic_kv = pool
    # Keep allocator and MemPool alive as long as all tensor views.
    runner._areal_dynamic_kv_backend = backend
    runner._areal_reset_kv = lambda: pool.reconcile({0}, trim=True)
    logging.getLogger("vllm.areal_kv_slab").info(
        "DYNAMIC_KV_INIT %s", json.dumps(pool.snapshot())
    )
    return raw


def install_dynamic_state_hook(cls):
    import json
    import logging
    from functools import wraps

    original = cls._update_states

    @wraps(original)
    def update(self, output):
        pool = getattr(self, "_areal_dynamic_kv", None)
        if pool is None:
            return original(self, output)
        before = pool.snapshot()
        pending = coordinated_state_call(
            pool, lambda: pending_blocks(self.requests, output)
        )
        pins = getattr(self, "_areal_kv_import_pins", {})
        imported = set().union(*pins.values()) if pins else set()
        coordinated_reconcile(pool, pending | imported)
        result = coordinated_state_call(pool, lambda: original(self, output))
        live = coordinated_state_call(
            pool,
            lambda: set().union(
                *(
                    set(flatten_blocks(state.block_ids))
                    for state in self.requests.values()
                )
            ),
        )
        for rid in list(pins):
            if rid in self.requests or rid in output.finished_req_ids:
                del pins[rid]
        imported = set().union(*pins.values()) if pins else set()
        coordinated_reconcile(pool, live | imported)
        after = pool.snapshot()
        if before["mapped_bytes"] != after["mapped_bytes"]:
            logging.getLogger("vllm.areal_kv_slab").info(
                "DYNAMIC_KV_RESIZE %s", json.dumps(after)
            )
        return result

    cls._update_states = update
