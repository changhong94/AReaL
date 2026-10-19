# SPDX-License-Identifier: Apache-2.0
import time
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from areal.infra.request_router import RequestLeasePool


def pool_with_request():
    pool = RequestLeasePool(["one", "two"], 8, task_affinity=False)
    lease = pool.acquire("a", "request-a")
    assert lease["rank"] == 0
    return pool, lease


def test_closed_admission_preserves_existing_lease_and_routes_new_to_other_card():
    pool, old = pool_with_request()
    pool.close(0)
    assert pool.acquire("a", "request-a") == old
    assert pool.acquire("b", "request-b")["rank"] == 1
    pool.release("a", old["lease_id"])
    assert pool.drain_until(0, time.monotonic(), lambda: True) == "empty"


@pytest.mark.parametrize(
    "ready,reason", [(False, "timeout"), (True, "next_batch_ready")]
)
def test_bounded_drain_retains_unfinished_leases_for_explicit_native_abort(
    ready, reason
):
    pool, _ = pool_with_request()
    pool.close(0)
    assert pool.drain_until(0, time.monotonic(), lambda: ready) == reason
    assert pool.active_requests(0) == ["request-a"]


def test_drain_fails_closed_on_unknown_execution_state():
    pool, lease = pool_with_request()
    pool.close(0)
    pool.release("a", lease["lease_id"], uncertain=True)
    with pytest.raises(RuntimeError, match="unknown execution state"):
        pool.drain_until(0, time.monotonic(), lambda: False)


def test_drain_rejects_open_admission():
    pool, _ = pool_with_request()
    with pytest.raises(RuntimeError, match="closed admission"):
        pool.drain_until(0, time.monotonic(), lambda: False)


@pytest.mark.parametrize("natural", [True, False])
def test_controller_native_pause_and_lease_release_precede_drain_completion(natural):
    from areal.api.alloc_mode import ModelAllocation
    from areal.infra.controller.rollout_controller import RolloutController

    controller = object.__new__(RolloutController)
    controller.rollout_alloc = ModelAllocation.from_str("vllm:d2p1t1")
    pool, lease = pool_with_request()
    controller._request_pool = pool
    controller._borrowed_pool = Mock()
    controller.training_prefetch_ready = Mock(return_value=False)
    calls = []

    def rpc(rank, method, **kwargs):
        calls.append((rank, method, pool.active_requests(0)))
        pool.release("a", lease["lease_id"])

    controller._worker_rpc = rpc
    deadline = controller.begin_shared_drain(0)
    assert not calls and pool.active_requests(0) == ["request-a"]
    if natural:
        pool.release("a", lease["lease_id"])
    controller.finish_shared_drain(deadline, 32)
    assert calls == [(0, "pause_generation", [] if natural else ["request-a"])]
    assert pool.active_requests(0) == []
    assert not pool.ready[0]


def test_trainer_keeps_budget_until_drain_ack_and_uses_absolute_batch_target():
    from areal.trainer.rl_trainer import PPOTrainer

    trainer = object.__new__(PPOTrainer)
    trainer.rollout = Mock()
    trainer.config = SimpleNamespace(train_dataset=SimpleNamespace(batch_size=16))
    trainer._shared_drain_seconds = 5
    trainer._training_inference_active = True
    trainer.rollout.begin_shared_drain.return_value = 123
    trainer._prepare_training_publication(True)
    assert trainer.rollout.mock_calls == [call.begin_shared_drain(5)]
    trainer._finish_training_drain(2)
    assert trainer.rollout.mock_calls == [
        call.begin_shared_drain(5),
        call.finish_shared_drain(123, 48),
        call.colocation_memory("kv_budget_clear"),
    ]
    assert not trainer._training_inference_active
    assert trainer._shared_drain_deadline is None


@pytest.mark.parametrize(
    "enabled,allow,active", [(0, True, True), (5, False, True), (5, True, False)]
)
def test_disabled_terminal_or_denied_admission_preserves_existing_stop_path(
    enabled, allow, active
):
    from areal.trainer.rl_trainer import PPOTrainer

    trainer = object.__new__(PPOTrainer)
    trainer.rollout = Mock()
    trainer._shared_drain_seconds = enabled
    trainer._training_inference_active = active
    trainer._prepare_training_publication(allow)
    trainer.rollout.begin_shared_drain.assert_not_called()
    if active:
        assert trainer.rollout.mock_calls == [
            call.borrow_for_training(),
            call.colocation_memory("kv_budget_clear"),
        ]


def test_drain_failure_never_lifts_budget_or_marks_inference_stopped():
    from areal.trainer.rl_trainer import PPOTrainer

    trainer = object.__new__(PPOTrainer)
    trainer.rollout = Mock()
    trainer.config = SimpleNamespace(train_dataset=SimpleNamespace(batch_size=16))
    trainer._shared_drain_deadline = 123
    trainer._training_inference_active = True
    trainer.rollout.finish_shared_drain.side_effect = RuntimeError(
        "native pause failed"
    )
    with pytest.raises(RuntimeError, match="native pause failed"):
        trainer._finish_training_drain(1)
    trainer.rollout.colocation_memory.assert_not_called()
    assert trainer._training_inference_active
