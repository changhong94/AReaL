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


def set_tp_protocol(pool, requested):
    """Negotiate once before serving, or at an acknowledged idle session reset."""
    import torch.distributed as dist
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    modes = [requested]
    if group.world_size > 1:
        modes = [None] * group.world_size
        dist.all_gather_object(modes, requested, group=group.cpu_group)
    if len(set(modes)) != 1 or requested not in {"object", "packed", "fused", "voted"}:
        pool.broken = True
        raise RuntimeError(f"Incompatible TP KV protocols: {modes}")
    pool.tp_protocol = requested if group.world_size > 1 else "object"
    import logging

    logging.getLogger("vllm.areal_kv_slab").info(
        "TP_KV_CONSENSUS protocol=%s members=%s", pool.tp_protocol, group.world_size
    )


def _packed_consensus(group, phase, error, digest=None, mapping_needed=False):
    """One fixed-size CPU collective on the healthy path; collect details on failure."""
    import torch
    import torch.distributed as dist

    payload = torch.tensor(
        [phase, int(error is not None), int(mapping_needed), *(digest or bytes(32))],
        device="cpu",
        dtype=torch.uint8,
    )
    states = [torch.empty_like(payload) for _ in range(group.world_size)]
    dist.all_gather(states, payload, group=group.cpu_group)
    rows = [bytes(state.tolist()) for state in states]
    any_mapping = any(row[2] for row in rows)
    if (
        not any(row[1] for row in rows)
        and len({row[:2] + row[3:] for row in rows}) == 1
    ):
        return None, any_mapping
    errors = [None] * group.world_size
    # All ranks see the same gathered values and take this branch together.
    dist.all_gather_object(errors, error, group=group.cpu_group)
    return [
        (message, row[0], row[3:].hex()) for message, row in zip(errors, rows)
    ], any_mapping


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
    if getattr(pool, "tp_protocol", "object") != "object":
        errors, _ = _packed_consensus(group, 0, error)
    else:
        states = [None] * group.world_size
        dist.all_gather_object(states, error, group=group.cpu_group)
        errors = states if any(states) else None
    if errors:
        pool.broken = True
        raise RuntimeError(f"TP KV state update failed: {errors}")
    return result


def coordinated_reconcile(pool, block_ids, metadata_only=False):
    """Fence TP peers before mapping and before any following model collective.

    Physical residency is local: each member may map a different number of pages.
    Logical block IDs and execution readiness must agree across the instance.
    """
    import torch.distributed as dist
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    if group.world_size == 1:
        local_ids = block_ids() if callable(block_ids) else block_ids
        if metadata_only:
            if (
                pool.broken
                or not pool.paused
                or not (set(local_ids) | {0}).issubset(pool.live)
            ):
                raise RuntimeError("Invalid paused KV metadata update")
            return
        return pool.reconcile(local_ids)
    ids = set()
    mapping_needed = True
    digest = bytes(32)
    error = None
    try:
        ids = set(block_ids() if callable(block_ids) else block_ids) | {0}
        if metadata_only and (not pool.paused or not ids.issubset(pool.live)):
            raise RuntimeError("Invalid paused KV metadata update")
        if pool.broken or (pool.swapped and not metadata_only):
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
        mapping_needed = (
            bool(target - pool.mapped)
            or len(pool.mapped - target) > pool.keep_idle_pages
        )
        growth = len(target - pool.mapped) * len(pool.tensors) * pool.page
        reclaimable = (
            max(0, len(pool.mapped - target) - pool.keep_idle_pages)
            * len(pool.tensors)
            * pool.page
        )
        if (
            not metadata_only
            and growth
            and pool.backend.free_bytes() + reclaimable < growth + pool.reserve_bytes
        ):
            raise MemoryError(
                "TP member cannot fit the next KV mapping within its budget"
            )
        digest = hashlib.sha256(str((metadata_only, sorted(ids))).encode()).digest()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    packed = getattr(pool, "tp_protocol", "object") != "object"
    if packed:
        states, any_mapping = _packed_consensus(group, 1, error, digest, mapping_needed)
    else:
        states = [None] * group.world_size
        dist.all_gather_object(states, (error, digest.hex()), group=group.cpu_group)
        if (
            not any(item[0] for item in states)
            and len({item[1] for item in states}) == 1
        ):
            states = None
    if states:
        pool.broken = True
        raise RuntimeError(f"TP KV preflight rejected execution: {states}")
    if metadata_only:
        return
    if getattr(pool, "tp_protocol", "object") == "voted" and not any_mapping:
        # Every TP rank validated the same logical IDs and its own residency.
        # reconcile() would only assign live; there is no mapping mutation to acknowledge.
        pool.live = ids
        return
    error = None
    try:
        pool.reconcile(ids)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if packed:
        states, _ = _packed_consensus(group, 2, error)
    else:
        states = [None] * group.world_size
        dist.all_gather_object(states, error, group=group.cpu_group)
        states = states if any(states) else None
    if states:
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
    set_tp_protocol(pool, os.getenv("AREAL_TP_KV_PROTOCOL", "object"))
    runner._areal_dynamic_kv = pool
    # Keep allocator and MemPool alive as long as all tensor views.
    runner._areal_dynamic_kv_backend = backend
    runner._areal_reset_kv = lambda: pool.reconcile({0}, trim=True)
    logging.getLogger("vllm.areal_kv_slab").info(
        "DYNAMIC_KV_INIT %s", json.dumps(pool.snapshot())
    )
    return raw


def paused_completion_only(runner, output, pool):
    """Allow paused request bookkeeping, never scheduling or KV data movement."""
    if not pool.paused or pool.broken or getattr(runner, "speculative_config", None):
        return False
    parallel = getattr(runner, "parallel_config", None)
    if getattr(parallel, "distributed_executor_backend", None) == "external_launcher":
        return False
    if getattr(output, "total_num_scheduled_tokens", None) != 0:
        return False
    for name in (
        "scheduled_new_reqs",
        "num_scheduled_tokens",
        "scheduled_spec_decode_tokens",
        "scheduled_encoder_inputs",
        "kv_cache_block_copies",
        "new_block_ids_to_zero",
        "kv_connector_metadata",
        "ec_connector_metadata",
        "preempted_req_ids",
        "free_encoder_mm_hashes",
        "has_structured_output_requests",
        "num_spec_tokens_to_schedule",
    ):
        if getattr(output, name, None):
            return False
    cached = getattr(output, "scheduled_cached_reqs", None)
    if cached is None:
        return False
    return not any(
        getattr(cached, name, None)
        for name in (
            "req_ids",
            "resumed_req_ids",
            "new_token_ids",
            "all_token_ids",
            "new_block_ids",
            "num_computed_tokens",
            "num_output_tokens",
        )
    )


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
        metadata_only = paused_completion_only(self, output, pool)
        if getattr(pool, "tp_protocol", "object") in {"fused", "voted"}:
            pins = getattr(self, "_areal_kv_import_pins", {})

            def before_update():
                imported = set().union(*pins.values()) if pins else set()
                return pending_blocks(self.requests, output) | imported

            # No collective lies between local ID preparation and its preflight.
            # Fencing the preparation result and hash together preserves the same
            # all-member check before any mapping or native state update.
            coordinated_reconcile(pool, before_update, metadata_only=metadata_only)
            result = None

            def after_update():
                nonlocal result
                result = original(self, output)
                live = set().union(
                    *(
                        flatten_blocks(state.block_ids)
                        for state in self.requests.values()
                    )
                )
                for rid in list(pins):
                    if rid in self.requests or rid in output.finished_req_ids:
                        del pins[rid]
                return live | (set().union(*pins.values()) if pins else set())

            coordinated_reconcile(pool, after_update, metadata_only=metadata_only)
        else:
            pending = coordinated_state_call(
                pool, lambda: pending_blocks(self.requests, output)
            )
            pins = getattr(self, "_areal_kv_import_pins", {})
            imported = set().union(*pins.values()) if pins else set()
            coordinated_reconcile(pool, pending | imported, metadata_only=metadata_only)
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
            coordinated_reconcile(pool, live | imported, metadata_only=metadata_only)
        after = pool.snapshot()
        if before["mapped_bytes"] != after["mapped_bytes"]:
            logging.getLogger("vllm.areal_kv_slab").info(
                "DYNAMIC_KV_RESIZE %s", json.dumps(after)
            )
        return result

    cls._update_states = update
