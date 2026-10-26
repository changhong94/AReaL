# SPDX-License-Identifier: Apache-2.0

"""Versioned raw-allocation DRAM copies for immutable residency intervals."""

import time
from concurrent.futures import Future
from threading import Thread

from areal.utils import request_trace


class RetainedBacking:
    """Writers and residency transitions must join before touching these blocks.

    Workspace and KV allocations are deliberately excluded: their contents can
    change without a weight/optimizer mutation boundary. Allocation identity is
    checked as well as its address, so allocator reuse cannot revive stale data.
    """

    def __init__(self, memory, tags):
        self.memory = memory
        self.tags = frozenset(tags)
        self.generations = dict.fromkeys(tags, 0)
        self.records = {}
        self.future = None

    def wait(self):
        future = self.future
        if future is None:
            return
        started = time.monotonic()
        try:
            result = future.result()
        except BaseException:
            self.memory.broken = True
            raise
        if self.future is future:
            self.future = None
        request_trace.emit(
            "retained_backup_join",
            seconds=time.monotonic() - started,
            copied_bytes=result["copied_bytes"],
        )

    def dirty(self, tags):
        self.wait()
        for tag in self.tags.intersection(tags):
            self.generations[tag] += 1
        # Release stale host storage; it must never authorize a copy-free unmap.
        self.records = {
            addr: row for addr, row in self.records.items() if row[0].tag not in tags
        }

    def get(self, allocation):
        row = self.records.get(allocation.handle[2])
        if (
            row is not None
            and row[0] is allocation
            and row[1] == tuple(allocation.handle)
            and row[2] == self.generations.get(allocation.tag)
        ):
            return row[3]
        return None

    def remember(self, allocation, backup):
        if allocation.tag in self.tags and backup is not None:
            self.records = {
                **self.records,
                allocation.handle[2]: (
                    allocation,
                    tuple(allocation.handle),
                    self.generations[allocation.tag],
                    backup,
                ),
            }

    def candidates(self):
        inventory = self.memory.inventory()
        self.records = {
            addr: row
            for addr, row in self.records.items()
            if inventory.get(addr) is row[0]
        }
        return [
            a
            for addr, a in inventory.items()
            if a.tag in self.tags
            and addr not in self.memory._missing
            and self.get(a) is None
        ]

    def start(self):
        if self.memory.broken:
            raise RuntimeError("Cannot back up a broken residency pool")
        if self.future is not None:
            if not self.future.done():
                return
            self.wait()
        allocations = self.candidates()
        if not allocations:
            return
        backend = self.memory.backend
        # Fence producer kernels before the background reader starts. Subsequent
        # inference/publication can read weights; writers must call dirty().
        backend.synchronize()
        device = (
            backend.torch.npu.current_device() if hasattr(backend, "torch") else None
        )
        self.future = Future()
        future = self.future
        generations = dict(self.generations)

        def run():
            started = time.monotonic()
            try:
                if device is not None:
                    backend.torch.npu.set_device(device)
                copies = [(a, backend.backup(a)) for a in allocations]
                if self.generations != generations:
                    raise RuntimeError("Residency generation changed during backup")
                for allocation, backup in copies:
                    self.remember(allocation, backup)
                result = dict(
                    copied_bytes=sum(a.handle[1] for a in allocations),
                    seconds=time.monotonic() - started,
                )
                request_trace.emit("retained_backup_complete", **result)
                future.set_result(result)
            except BaseException as exc:
                future.set_exception(exc)

        Thread(target=run, name="retained-dram-backup").start()

    def host_bytes(self):
        # Only count mapped retained copies here; missing copies are accounted
        # by TaggedMemory.cpu_backup_bytes already.
        return sum(
            a.handle[1]
            for addr, (a, _, _, _) in self.records.items()
            if addr not in self.memory._missing
        )
