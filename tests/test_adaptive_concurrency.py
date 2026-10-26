# SPDX-License-Identifier: Apache-2.0
"""Admission changes must preserve staleness, routing epochs and existing requests."""

from types import SimpleNamespace

from areal.infra.adaptive_concurrency import CapacityConcurrency
from areal.infra.request_router import RequestLeasePool
from areal.infra.staleness_manager import StalenessManager


def test_capacity_growth_is_rate_limited_and_withdrawal_is_immediate(monkeypatch):
    state = dict(ready=[True, True], waiting=0, borrowed_available=True)
    clock = [0.0]
    monkeypatch.setattr(
        "areal.infra.adaptive_concurrency.time.monotonic", lambda: clock[0]
    )
    policy = CapacityConcurrency(16, 1, lambda: state)
    assert policy() == 20
    assert policy() == 20
    for second, expected in [(1, 24), (2, 28), (3, 32)]:
        clock[0] = second
        assert policy() == expected
    state["borrowed_available"] = False
    assert policy() == 16
    state["ready"] = [False, False]
    assert policy() == 0


def test_request_backlog_blocks_growth_and_replica_count_not_tp_width(monkeypatch):
    state = dict(ready=[True] * 4, waiting=1, borrowed_available=True)
    policy = CapacityConcurrency(16, 2, lambda: state)
    assert policy() == 16
    state["waiting"] = 0
    assert policy() == 20
    state["ready"] = [False, False, True, True]
    assert policy() == 16


def test_dynamic_target_does_not_bypass_staleness_or_cancel_running():
    target = [32]
    manager = StalenessManager(
        SimpleNamespace(get_version=lambda: 0),
        32,
        16,
        0,
        concurrency_target=lambda: target[0],
    )
    assert manager.get_capacity() == 16
    for _ in range(16):
        manager.on_rollout_enqueued()
        manager.on_rollout_submitted()
    target[0] = 4
    assert manager.get_capacity() == -12
    assert manager.get_stats().running == 16
    target[0] = 1000
    assert manager.get_capacity() == 0


def test_external_snapshot_runs_outside_statistics_lock():
    manager = StalenessManager(SimpleNamespace(get_version=lambda: 0), 32, 16, 2)

    def snapshot():
        assert manager.lock.acquire(blocking=False)
        manager.lock.release()
        return 20

    manager.concurrency_target = snapshot
    assert manager.get_capacity() == 20


def test_admission_snapshot_excludes_closing_and_old_versions():
    pool = RequestLeasePool(["a", "b", "c", "d"], 64)
    pool.versions = [2, 1, 2, 2]
    pool.closing.add(2)
    pool.ready[3] = False
    assert pool.admission_snapshot(2)["ready"] == [True, False, False, False]


def test_provider_is_clipped_to_hard_limit_with_available_staleness():
    manager = StalenessManager(
        SimpleNamespace(get_version=lambda: 3),
        32,
        16,
        2,
        concurrency_target=lambda: 1000,
    )
    assert manager.get_capacity() == 32


def test_dispatcher_rechecks_dynamic_capacity_without_completion():
    import threading
    from collections import deque

    from areal.infra.workflow_executor import BatchTaskDispatcher

    dispatcher = object.__new__(BatchTaskDispatcher)
    dispatcher._input_cv = threading.Condition()
    dispatcher._shutdown_event = threading.Event()
    dispatcher._check_thread_exception = lambda: None
    dispatcher.runner = SimpleNamespace(paused=threading.Event())
    calls = []

    def capacity():
        calls.append(True)
        return 1 if len(calls) > 1 else 0

    dispatcher.staleness_manager = SimpleNamespace(
        get_capacity=capacity, concurrency_target=lambda: 1
    )
    dispatcher._pending_inputs = deque(["pending"])
    assert dispatcher._get_next_task_for_submission() == "pending"
    assert len(calls) == 2


def test_capacity_scales_beyond_twofold_and_preserves_staleness(monkeypatch):
    state = dict(ready=[True] * 3, waiting=0, borrowed_available=True)
    clock = [0.0]
    monkeypatch.setattr(
        "areal.infra.adaptive_concurrency.time.monotonic", lambda: clock[0]
    )
    policy = CapacityConcurrency(16, 2, lambda: state)
    for replicas, expected in [(3, 24), (4, 32), (6, 48), (8, 64), (10, 80)]:
        state["ready"] = [True] * replicas
        for _ in range(24):
            clock[0] += 1
            target = policy()
        assert target == expected
        manager = StalenessManager(
            SimpleNamespace(get_version=lambda: 0),
            policy.capacity_for(10),
            16,
            2,
            concurrency_target=policy,
        )
        assert manager.get_capacity() == min(expected, 48)
    state["ready"] = [True] * 3
    assert policy() == 24
    state["ready"] = [False] * 10
    assert policy() == 0


def test_fractional_replica_ratio_rounds_up():
    policy = CapacityConcurrency(16, 3, lambda: {})
    assert policy.capacity_for(4) == 22
    assert policy.capacity_for(12) == 64
