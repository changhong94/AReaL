# SPDX-License-Identifier: Apache-2.0
"""Fail-closed lifecycle for a pool containing only disposable KV memory."""

from collections.abc import Callable
from typing import Any


class KVSlabMemory:
    def __init__(
        self,
        allocator: Any,
        synchronize: Callable[[], None],
        reset_cache: Callable[[], None],
        free_bytes: Callable[[], int],
    ):
        self.allocator = allocator
        self.synchronize = synchronize
        self.reset_cache = reset_cache
        self.free_bytes = free_bytes
        self.state = "mapped"

    def _validate_inventory(self) -> int:
        allocations = list(self.allocator.pointer_to_data.values())
        if not allocations or any(a.tag != "kv_cache" for a in allocations):
            raise RuntimeError("KV slab pool must contain only KV allocations")
        return sum(a.handle[1] for a in allocations)

    def transition(self, mapped: bool) -> dict[str, Any]:
        target = "mapped" if mapped else "released"
        if self.state == "broken":
            raise RuntimeError("KV slab memory transition previously failed")
        managed_bytes = self._validate_inventory()
        if self.state == target:
            return {"state": target, "managed_bytes": managed_bytes, "changed": False}
        self.synchronize()
        before = self.free_bytes()
        # Any exception leaves this instance unusable. Never retry partial remaps.
        self.state = "broken"
        if mapped:
            self.allocator.wake_up(tags=["kv_cache"])
            self.reset_cache()
        else:
            self.allocator.sleep(offload_tags=())
        self.synchronize()
        after = self.free_bytes()
        self.state = target
        return {
            "state": target,
            "managed_bytes": managed_bytes,
            "changed": True,
            "free_bytes_before": before,
            "free_bytes_after": after,
        }
