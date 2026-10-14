# SPDX-License-Identifier: Apache-2.0
"""Preserved request lifetime and partial live KV backing invariants."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.engine.vllm_ext.dynamic_kv import SparseKVPages
from areal.infra.borrowed_rollout import BorrowedRolloutPool
from areal.infra.controller.rollout_controller import RolloutController
from areal.infra.request_router import RequestLeasePool


class PageBackend:
    page = 16

    def __init__(self):
        self.data = {}
        self.capacity = 1000

    def synchronize(self):
        pass

    def free_bytes(self):
        return self.capacity - len(self.data) * self.page

    def map(self, address):
        self.data[address] = bytes(self.page)

    def unmap(self, address):
        del self.data[address]

    def backup(self, address):
        return self.data[address]

    def restore(self, address, backup):
        self.data[address] = backup


def pages():
    backend = PageBackend()
    tensors = [
        SimpleNamespace(data_ptr=lambda: 1024),
        SimpleNamespace(data_ptr=lambda: 2048),
    ]
    pool = SparseKVPages(backend, tensors, 4, 32, 0)
    pool.reconcile({4, 8, 12})
    for address in backend.data:
        backend.data[address] = bytes([address // 16 % 251] * 16)
    return backend, pool


def test_preserve_only_shortfall_and_restore_exact_bytes_at_same_addresses():
    backend, pool = pages()
    expected = backend.data.copy()
    pool.pause_preserved()
    pool.reclaim_preserved(backend.free_bytes() + 1)
    assert len(pool.swapped) == 1
    assert len(pool.mapped) == 3
    assert pool.snapshot()["cpu_backup_bytes"] == 32
    assert pool.live == {0, 4, 8, 12}
    with pytest.raises(RuntimeError, match="Restore retained"):
        pool.reconcile({0})
    pool.resume_preserved()
    assert backend.data == expected
    assert not pool.swapped and not pool.paused


def test_preserve_resident_when_budget_sufficient_and_second_reclaim_is_incremental():
    backend, pool = pages()
    pool.pause_preserved()
    pool.reclaim_preserved(backend.free_bytes())
    assert not pool.swapped
    pool.reclaim_preserved(backend.free_bytes() + 1)
    pool.reclaim_preserved(backend.free_bytes() + 1)
    assert len(pool.swapped) == 2 and len(pool.mapped) == 2
    pool.resume_preserved()
    assert len(pool.mapped) == 4


def test_live_reclaim_without_pause_is_rejected():
    backend, pool = pages()
    with pytest.raises(RuntimeError, match="acknowledged pause"):
        pool.reclaim_preserved(backend.free_bytes() + 1)


def test_failed_copy_blocks_resume():
    backend, pool = pages()
    pool.pause_preserved()
    backend.backup = Mock(side_effect=RuntimeError("copy failed"))
    with pytest.raises(RuntimeError, match="copy failed"):
        pool.reclaim_preserved(backend.free_bytes() + 1)
    assert pool.broken
    with pytest.raises(RuntimeError, match="broken"):
        pool.resume_preserved()


@pytest.mark.parametrize("batch", [1, 2, 16])
def test_preserve_threshold_waits_for_full_batch(batch):
    pool = BorrowedRolloutPool(batch, preserve_requests=True)
    assert pool.training_prefetch_target() == batch
    pool.choose(batch - 1)
    assert pool.state == "inference"
    assert pool.choose(batch) == (1, True)
    pool.batch_ready(batch)
    assert pool.training_prefetch_target() == 2 * batch


def test_borrow_retains_original_lease_without_waiting_for_it_to_finish():
    obj = object.__new__(RolloutController)
    obj._borrowed_pool = BorrowedRolloutPool(16, preserve_requests=True)
    obj._preserve_borrowed_requests = True
    obj._request_pool = RequestLeasePool(["shared", "fixed"], 10)
    lease = obj._request_pool.acquire("t", "r")
    assert lease["rank"] == 0
    obj._worker_rpc = Mock()
    obj._pool_event = Mock()
    obj.get_version = Mock(return_value=0)
    obj.borrow_for_training()
    assert obj._request_pool.active["t"] == lease
    assert obj._request_pool.ready == [False, True]
    obj._worker_rpc.assert_called_once_with(
        0, "pause_generation_preserve", http_timeout=60.0
    )


def test_retained_drain_precedes_new_snapshot_load(tmp_path):
    obj = object.__new__(RolloutController)
    obj._preserve_borrowed_requests = True
    obj._request_pool = RequestLeasePool(["shared", "fixed"], 10)
    lease = obj._request_pool.acquire("t", "r")
    obj._request_pool.close(0)
    obj._borrowed_pool = BorrowedRolloutPool(16, preserve_requests=True)
    obj.config = SimpleNamespace(request_timeout=1)
    obj._pool_event = Mock()
    obj.get_version = Mock(return_value=1)
    obj._dispatcher = Mock()
    obj._staleness_manager = Mock()
    obj._staleness_manager.get_stats.return_value.accepted = 16
    from threading import Lock

    obj._shared_publication_lock = Lock()
    calls = []

    def rpc(rank, method, **kwargs):
        calls.append(method)
        if method == "continue_generation" and len(calls) == 1:
            assert obj._request_pool.active["t"] == lease
            assert not obj._request_pool.ready[0]
            obj._request_pool.release("t", lease["lease_id"])
        if method == "update_weights_from_disk":
            assert not obj._request_pool.active

    obj._worker_rpc = rpc
    obj._load_shared_snapshot(
        SimpleNamespace(version=1, path=str(tmp_path / "snapshot"))
    )
    assert calls[:3] == [
        "continue_generation",
        "pause_generation",
        "update_weights_from_disk",
    ]
    assert obj._request_pool.versions[0] == 1 and obj._request_pool.ready[0]


@pytest.mark.asyncio
async def test_late_native_admission_keeps_lease_and_waits_without_holding_lock(
    monkeypatch,
):
    import asyncio
    from unittest.mock import AsyncMock

    from vllm.v1.engine.async_llm import AsyncLLM

    from areal.engine.vllm_ext.request_admission import (
        install_admission_guard,
        routed_request,
    )

    original = AsyncMock(return_value="admitted")
    original._areal_admission_guard = False
    monkeypatch.setattr(AsyncLLM, "_add_request", original)
    lock, running = asyncio.Lock(), asyncio.Event()
    install_admission_guard(lock, running)
    obj = SimpleNamespace(_areal_preserve_pause=True)
    token = routed_request.set(True)
    try:
        task = asyncio.create_task(AsyncLLM._add_request(obj))
        await asyncio.sleep(0)
        assert not task.done() and not lock.locked()
        original.assert_not_awaited()
        async with lock:
            running.set()
        assert await asyncio.wait_for(task, 1) == "admitted"
        original.assert_awaited_once()
    finally:
        routed_request.reset(token)


@pytest.mark.asyncio
async def test_new_weights_rejected_while_retained_requests_are_paused():
    import asyncio
    from unittest.mock import AsyncMock

    from areal.engine.vllm_ext.generation_gate import run_weight_update_while_paused

    engine = SimpleNamespace(_areal_preserve_pause=True)
    load = AsyncMock()
    with pytest.raises(RuntimeError, match="Drain retained"):
        await run_weight_update_while_paused(engine, asyncio.Event(), load)
    load.assert_not_awaited()
