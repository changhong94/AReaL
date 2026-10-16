# SPDX-License-Identifier: Apache-2.0

import asyncio
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from areal.engine.vllm_ext.generation_gate import update_weights_while_paused
from areal.engine.vllm_ext.kv_slab import KVSlabTable, SlabFreeBlockQueue
from areal.engine.vllm_ext.kv_slab_adapter import transition_slab_cache
from areal.engine.vllm_ext.kv_slab_memory import KVSlabMemory


def test_slab_zero_tail_and_capacity_conservation():
    table = KVSlabTable(19, 8)
    # Slab zero remains a valid partial slab despite reserving null block zero.
    first = table.allocate(1)
    assert first == [7]
    assert table.allocate(6) == [6, 5, 4, 3, 2, 1]
    rest = table.allocate(11)
    assert set(first + [6, 5, 4, 3, 2, 1] + rest) == set(range(1, 19))
    assert table.num_free_blocks == 0
    before = table.snapshot()
    with pytest.raises(ValueError):
        table.allocate(1)
    assert table.snapshot() == before
    table.release(range(1, 19))
    assert table.num_free_blocks == 18
    assert table.snapshot()["empty_slabs"] == 3


def test_invalid_free_is_atomic():
    table = KVSlabTable(20, 4)
    ids = table.allocate(3)
    for invalid in ([ids[0], ids[0]], [ids[0], 0], [ids[0], 100]):
        before = table.snapshot()
        with pytest.raises(ValueError):
            table.release(invalid)
        assert table.snapshot() == before


def test_random_allocation_and_free_do_not_alias():
    rng = random.Random(91)
    table = KVSlabTable(137, 16)
    live = set()
    for _ in range(1000):
        if live and rng.random() < 0.5:
            ids = rng.sample(sorted(live), rng.randint(1, len(live)))
            table.release(ids)
            live.difference_update(ids)
        else:
            count = rng.randint(0, table.num_free_blocks)
            ids = table.allocate(count)
            assert not live.intersection(ids)
            assert len(ids) == len(set(ids))
            live.update(ids)
        assert table.num_free_blocks + len(live) == 136
        assert table.allocated == live
    table.release(live)
    assert table.num_free_blocks == 136


def test_queue_recycles_real_vllm_block_objects():
    blocks = [SimpleNamespace(block_id=i) for i in range(18)]
    queue = SlabFreeBlockQueue(blocks, 8)
    allocated = queue.popleft_n(17)
    assert all(block is blocks[block.block_id] for block in allocated)
    assert all(block.block_id != 0 for block in allocated)
    queue.prepend_n(allocated)
    assert queue.num_free_blocks == 17


def make_memory():
    allocator = SimpleNamespace(
        pointer_to_data={1: SimpleNamespace(tag="kv_cache", handle=(0, 4096, 1, 2))},
        sleep=Mock(),
        wake_up=Mock(),
    )
    sync, reset = Mock(), Mock()
    memory = KVSlabMemory(allocator, sync, reset, lambda: 8192)
    return memory, allocator, reset


def test_memory_repeated_cycles_and_idempotency():
    memory, allocator, reset = make_memory()
    for _ in range(3):
        assert memory.transition(False)["changed"]
        assert not memory.transition(False)["changed"]
        assert memory.transition(True)["changed"]
        assert not memory.transition(True)["changed"]
    assert allocator.sleep.call_count == allocator.wake_up.call_count == 3
    allocator.sleep.assert_called_with(offload_tags=())
    allocator.wake_up.assert_called_with(tags=["kv_cache"])
    assert reset.call_count == 3


def test_memory_never_discards_weight_pool():
    memory, allocator, _ = make_memory()
    allocator.pointer_to_data[2] = SimpleNamespace(tag="weights", handle=(0, 8, 2, 3))
    with pytest.raises(RuntimeError, match="only KV"):
        memory.transition(False)
    allocator.sleep.assert_not_called()


@pytest.mark.parametrize("failure", ["sleep", "wake", "reset"])
def test_memory_partial_failure_cannot_be_retried(failure):
    memory, allocator, reset = make_memory()
    if failure == "sleep":
        allocator.sleep.side_effect = RuntimeError("fail")
    else:
        memory.transition(False)
        if failure == "wake":
            allocator.wake_up.side_effect = RuntimeError("fail")
        else:
            reset.side_effect = RuntimeError("fail")
    with pytest.raises(RuntimeError, match="fail"):
        memory.transition(failure != "sleep")
    assert memory.state == "broken"
    with pytest.raises(RuntimeError, match="previously failed"):
        memory.transition(True)


@pytest.mark.asyncio
async def test_live_blocks_prevent_physical_reclaim(monkeypatch):
    monkeypatch.setenv("AREAL_KV_SLAB_BLOCKS", "16")
    engine = AsyncMock()
    engine.engine_core.call_utility_async.return_value = {"used_blocks": 1}
    with pytest.raises(RuntimeError, match="live blocks"):
        await transition_slab_cache(engine, False)
    engine.collective_rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_remap_leaves_weight_gate_closed():
    engine = AsyncMock()
    event = asyncio.Event()
    event.set()
    restore = AsyncMock(side_effect=RuntimeError("remap failed"))
    with pytest.raises(RuntimeError, match="remap failed"):
        await update_weights_while_paused(engine, event, "/model", before_load=restore)
    assert not event.is_set()
    engine.collective_rpc.assert_not_awaited()
    engine.resume_generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_slab_does_not_change_existing_path(monkeypatch):
    monkeypatch.delenv("AREAL_KV_SLAB_BLOCKS", raising=False)
    engine = AsyncMock()
    await transition_slab_cache(engine, False)
    engine.collective_rpc.assert_not_awaited()
    engine.engine_core.call_utility_async.assert_not_awaited()


def test_live_capacity_changes_preserve_allocator_conservation():
    rng = random.Random(37)
    queue = SlabFreeBlockQueue([SimpleNamespace(block_id=i) for i in range(65)], 8)
    live = {}
    for _ in range(300):
        if live and rng.random() < 0.4:
            ids = rng.sample(sorted(live), rng.randint(1, len(live)))
            queue.append_n([live.pop(block) for block in ids])
        elif rng.random() < 0.5:
            count = rng.randint(0, queue.num_free_blocks)
            blocks = queue.popleft_n(count)
            assert not live.keys() & {block.block_id for block in blocks}
            live.update((block.block_id, block) for block in blocks)
        else:
            capacity = rng.randint(max(2, max(live, default=0) + 1), 65)
            queue.set_capacity(capacity)
        free = {block.block_id for block in queue.get_all_free_blocks()}
        assert not free & live.keys()
        assert free | live.keys() == set(range(1, queue.table.num_blocks))
        assert queue.num_free_blocks == len(free)
        assert queue.table.allocated == live.keys()
