# SPDX-License-Identifier: BSD-3-Clause
# Adapted from FineServe KVSlab / KVSlabTable (SAMSUNG SDS, 2025).
# See licenses/FineServe.txt for license and source provenance.
"""Single-format slab allocation for the opt-in Ascend vLLM adapter.

A block ID indexes the existing native per-layer KV tensors. Slabs group
contiguous block IDs; no CUDA IPC or attention-layout conversion is involved.
"""

from collections.abc import Iterable
from typing import Any


class KVSlabTable:
    """FineServe's partial/full/free slab policy with transactional checks.

    Block zero is vLLM's permanent null block. A final partial slab is retained
    rather than rounding capacity up beyond the actual KV tensor allocation.
    """

    def __init__(self, num_blocks: int, blocks_per_slab: int):
        if num_blocks < 2 or blocks_per_slab < 1:
            raise ValueError(
                "KV slabs require at least two blocks and positive slab size"
            )
        self.num_blocks = num_blocks
        self.blocks_per_slab = blocks_per_slab
        self.free: dict[int, list[int]] = {}
        for block_id in range(1, num_blocks):
            self.free.setdefault(block_id // blocks_per_slab, []).append(block_id)
        self.capacities = {sid: len(ids) for sid, ids in self.free.items()}
        self.empty = dict.fromkeys(self.free)
        self.partial: dict[int, None] = {}
        self.allocated: set[int] = set()
        self.allocation_count = 0
        self.free_count = 0
        self.high_watermark = 0

    @property
    def num_free_blocks(self) -> int:
        return self.num_blocks - 1 - len(self.allocated)

    def allocate(self, count: int) -> list[int]:
        if count < 0 or count > self.num_free_blocks:
            raise ValueError("Insufficient KV slab blocks")
        result = []
        while len(result) < count:
            # Slab ID 0 is valid: never test its truth value.
            sid = next(iter(self.partial), None)
            if sid is None:
                sid = next(iter(self.empty))
                del self.empty[sid]
            slots = self.free[sid]
            take = min(count - len(result), len(slots))
            for _ in range(take):
                result.append(slots.pop())
            if slots:
                self.partial[sid] = None
            else:
                self.partial.pop(sid, None)
        self.allocated.update(result)
        self.allocation_count += count
        self.high_watermark = max(self.high_watermark, len(self.allocated))
        return result

    def release(self, block_ids: Iterable[int]) -> None:
        ids = list(block_ids)
        if len(ids) != len(set(ids)) or not set(ids).issubset(self.allocated):
            raise ValueError("Unknown, null, or duplicate KV slab block release")
        for bid in ids:
            sid = bid // self.blocks_per_slab
            self.allocated.remove(bid)
            self.free[sid].append(bid)
            if len(self.free[sid]) == self.capacities[sid]:
                self.partial.pop(sid, None)
                self.empty[sid] = None
            else:
                self.partial[sid] = None
        self.free_count += len(ids)

    def snapshot(self) -> dict[str, int]:
        return {
            "blocks_per_slab": self.blocks_per_slab,
            "slabs": len(self.free),
            "empty_slabs": len(self.empty),
            "partial_slabs": len(self.partial),
            "used_blocks": len(self.allocated),
            "free_blocks": self.num_free_blocks,
            "allocation_count": self.allocation_count,
            "free_count": self.free_count,
            "high_watermark": self.high_watermark,
        }


class SlabFreeBlockQueue:
    """vLLM free-queue adapter; prefix caching/LRU eviction are unsupported."""

    def __init__(self, blocks: list[Any], blocks_per_slab: int):
        self.blocks = blocks
        self.table = KVSlabTable(len(blocks), blocks_per_slab)

    @property
    def num_free_blocks(self) -> int:
        return self.table.num_free_blocks

    def set_capacity(self, num_blocks: int) -> None:
        """Resize free capacity at a scheduler boundary without changing live IDs."""
        if type(num_blocks) is not int or not 2 <= num_blocks <= len(self.blocks):
            raise ValueError("Invalid KV capacity")
        old = self.table
        if any(block >= num_blocks for block in old.allocated):
            raise RuntimeError("KV capacity would exclude live blocks")
        table = KVSlabTable(num_blocks, old.blocks_per_slab)
        table.allocated = set(old.allocated)
        for sid, slots in table.free.items():
            remaining = [block for block in slots if block not in table.allocated]
            table.free[sid] = remaining
            if len(remaining) != table.capacities[sid]:
                table.empty.pop(sid, None)
                if remaining:
                    table.partial[sid] = None
        table.allocation_count = old.allocation_count
        table.free_count = old.free_count
        table.high_watermark = old.high_watermark
        self.table = table

    def popleft_n(self, count: int) -> list[Any]:
        return [self.blocks[i] for i in self.table.allocate(count)]

    def popleft(self) -> Any:
        return self.popleft_n(1)[0]

    def append_n(self, blocks: Iterable[Any]) -> None:
        self.table.release(b.block_id for b in blocks)

    def prepend_n(self, blocks: Iterable[Any]) -> None:
        self.append_n(blocks)

    def append(self, block: Any) -> None:
        self.append_n([block])

    def remove(self, block: Any) -> None:
        raise RuntimeError("Prefix-cache touching is unsupported by KV slabs")

    def get_all_free_blocks(self) -> list[Any]:
        return [self.blocks[i] for ids in self.table.free.values() for i in ids]


def set_scheduler_capacity(
    block_pool: Any, num_blocks: int, watermark_blocks: int = 0
) -> dict:
    """Update free capacity and native accounting while retaining live block IDs."""
    if watermark_blocks:
        raise ValueError("Training priority requires zero KV watermark")
    queue = block_pool.free_block_queue
    if not isinstance(queue, SlabFreeBlockQueue):
        raise RuntimeError("Training priority requires the slab scheduler")
    queue.set_capacity(num_blocks)
    block_pool.num_gpu_blocks = num_blocks
    return queue.table.snapshot()
