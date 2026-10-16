# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.test_resident_colocation import trainer as resident_trainer

from areal.api import Job, Worker
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import InferenceEngineConfig, SchedulingSpec, SchedulingStrategy
from areal.infra.borrowed_rollout import BorrowedRolloutPool
from areal.infra.controller.rollout_controller import RolloutController


@pytest.fixture
def trainer():
    return resident_trainer.__wrapped__()


def test_pool_withdraws_before_batch_without_stopping_fixed_replica():
    pool = BorrowedRolloutPool(8)
    assert [pool.choose(0)[0] for _ in range(4)] == [0, 1, 0, 1]
    assert pool.choose(6) == (0, False)
    assert pool.choose(7) == (1, True)
    assert pool.choose(7) == (1, False)
    pool.batch_ready(8)
    assert pool.choose(8)[0] == 1
    pool.release(8)
    assert {pool.choose(8)[0] for _ in range(2)} == {0, 1}


def test_pool_prefetched_batch_keeps_borrowed_replica_withdrawn():
    pool = BorrowedRolloutPool(8)
    pool.batch_ready(8)
    pool.release(16)
    assert pool.choose(16)[0] == 1
    pool.batch_ready(8)
    pool.release(16)
    assert {pool.choose(16)[0] for _ in range(2)} == {0, 1}


def test_pool_single_item_batch_can_use_borrowed_replica():
    pool = BorrowedRolloutPool(1)
    assert pool.choose(0)[0] == 0
    with pytest.raises(ValueError, match="fixed-size"):
        pool.batch_ready(2)


def test_borrow_pauses_only_rank_zero_in_order():
    controller = object.__new__(RolloutController)
    controller.rollout_alloc = ModelAllocation.from_str("vllm:d2p1t1")
    controller._borrowed_pool = BorrowedRolloutPool(8)
    controller._worker_rpc = Mock()
    controller._pool_event = Mock()
    controller.get_version = Mock(return_value=1)
    controller.borrow_for_training()
    assert [(c.args[0], c.args[1]) for c in controller._worker_rpc.call_args_list] == [
        (0, "pause"),
        (0, "pause_generation"),
    ]


def test_pool_trainer_borrows_without_global_pause(trainer):
    trainer.rollout_alloc = ModelAllocation.from_str("vllm:d2p1t1")
    trainer._offload_rollout()
    trainer.rollout.borrow_for_training.assert_called_once_with()
    trainer.rollout.pause.assert_not_called()
    trainer.rollout.pause_generation.assert_not_called()
    trainer._memory_policy.train.assert_called_once_with()


def test_pool_publication_pauses_both_before_loading(trainer):
    trainer.rollout_alloc = ModelAllocation.from_str("vllm:d2p1t1")
    calls = Mock()
    calls.attach_mock(trainer.actor, "actor")
    calls.attach_mock(trainer.rollout, "rollout")
    trainer._publish_weights(2)
    names = [c[0] for c in calls.mock_calls]
    loads = [i for i, n in enumerate(names) if n == "actor.update_weights"]
    assert loads[0] < names.index("rollout.pause_generation") < loads[1]
    assert (
        loads[1]
        < names.index("rollout.set_version")
        < names.index("rollout.continue_generation")
    )


def test_pool_failed_pause_never_loads_or_resumes(trainer):
    trainer.rollout_alloc = ModelAllocation.from_str("vllm:d2p1t1")
    trainer.rollout.pause_generation.side_effect = RuntimeError("pause failed")
    with pytest.raises(RuntimeError, match="pause failed"):
        trainer._publish_weights(2)
    assert trainer.actor.update_weights.call_count == 1
    trainer.rollout.set_version.assert_not_called()
    trainer.rollout.continue_generation.assert_not_called()


def test_pool_configuration_requires_prefetch(trainer):
    trainer.rollout_alloc = ModelAllocation.from_str("vllm:d2p1t1")
    with pytest.raises(ValueError, match="batch admission"):
        trainer._validate_npu_share()
    trainer.config.rollout.max_head_offpolicyness = 2
    trainer._validate_npu_share()


@pytest.mark.asyncio
async def test_mixed_jobs_preserve_engine_ranks_and_explicit_addresses():
    config = InferenceEngineConfig(backend="vllm:d2p1t1")
    scheduler = Mock()
    scheduler.create_workers.side_effect = lambda job: [f"{job.role}/0"]
    scheduler.get_workers.side_effect = lambda role: [
        Worker(id=f"{role}/0", ip="localhost")
    ]
    scheduler.create_engine = AsyncMock()
    scheduler.async_call_engine = AsyncMock()
    controller = RolloutController(Mock, config, scheduler)
    controller._worker_role = "rollout"
    controller.enable_npu_share_pool()
    job = Job(
        role="rollout",
        replicas=2,
        tasks=[SchedulingSpec(), SchedulingSpec()],
        scheduling_strategy=SchedulingStrategy(
            type="colocation", target="actor", fork=True
        ),
    )
    infos = [
        SimpleNamespace(host="localhost", port=8000),
        SimpleNamespace(host="localhost", port=8001),
    ]
    await controller._async_initialize(job, {}, server_infos=infos)
    jobs = [c.kwargs["job"] for c in scheduler.create_workers.call_args_list]
    assert [(j.role, j.replicas, str(j.scheduling_strategy.type)) for j in jobs] == [
        ("rollout", 1, "colocation"),
        ("rollout-fixed", 1, "separation"),
    ]
    calls = scheduler.async_call_engine.call_args_list
    assert [
        (c.kwargs["worker_id"], c.kwargs["engine_name"], c.kwargs["addr"])
        for c in calls
    ] == [
        ("rollout/0", "rollout/0", "localhost:8000"),
        ("rollout-fixed/0", "rollout/1", "localhost:8001"),
    ]


def test_memory_rpc_targets_only_shared_device():
    config = InferenceEngineConfig(backend="vllm:d2p1t1")
    scheduler = Mock()
    scheduler.async_call_engine = AsyncMock(return_value={"free_bytes": 123})
    controller = RolloutController(Mock, config, scheduler)
    controller._worker_role = "rollout"
    controller.workers = [
        Worker(id="rollout/0", ip="localhost"),
        Worker(id="rollout-fixed/0", ip="localhost"),
    ]
    controller.enable_npu_share_pool()
    assert controller.colocation_memory("status") == {"free_bytes": 123}
    scheduler.async_call_engine.assert_called_once_with(
        worker_id="rollout/0",
        engine_name="rollout/0",
        method="colocation_memory",
        action="status",
    )


def test_cleanup_attempts_all_roles_when_one_deletion_fails():
    controller = RolloutController(
        Mock, InferenceEngineConfig(backend="vllm:d2p1t1"), Mock()
    )
    controller._worker_roles = ["rollout", "rollout-fixed"]
    controller.scheduler.delete_workers.side_effect = [
        RuntimeError("delete failed"),
        None,
    ]
    controller._delete_worker_roles()
    assert [
        c.kwargs["role"] for c in controller.scheduler.delete_workers.call_args_list
    ] == ["rollout-fixed", "rollout"]
    assert controller._worker_roles == ["rollout-fixed"]


def test_pool_failed_load_keeps_versions_and_admission_closed(trainer):
    trainer.rollout_alloc = ModelAllocation.from_str("vllm:d2p1t1")
    trainer.actor.update_weights.side_effect = [None, RuntimeError("load failed")]
    with pytest.raises(RuntimeError, match="load failed"):
        trainer._publish_weights(2)
    trainer.rollout.set_version.assert_not_called()
    trainer.rollout.continue_generation.assert_not_called()


@pytest.mark.parametrize("failure", ["create", "initialize"])
def test_initialize_failure_cleans_both_scheduler_roles(failure):
    config = InferenceEngineConfig(
        backend="vllm:d2p1t1", scheduling_spec=[SchedulingSpec()]
    )
    scheduler = Mock()

    def create(job):
        if failure == "create" and job.role == "rollout-fixed":
            raise RuntimeError("injected initialization failure")
        return [f"{job.role}/0"]

    scheduler.create_workers.side_effect = create
    scheduler.get_workers.side_effect = lambda role: [
        Worker(id=f"{role}/0", ip="localhost")
    ]
    scheduler.create_engine = AsyncMock()
    scheduler.async_call_engine = AsyncMock(
        side_effect=RuntimeError("injected initialization failure")
    )
    controller = RolloutController(Mock, config, scheduler)
    controller.enable_npu_share_pool()
    infos = [
        SimpleNamespace(host="localhost", port=8000),
        SimpleNamespace(host="localhost", port=8001),
    ]
    with pytest.raises(RuntimeError, match="injected initialization failure"):
        controller.initialize("rollout", server_infos=infos)
    assert [c.kwargs["role"] for c in scheduler.delete_workers.call_args_list] == [
        "rollout-fixed",
        "rollout",
    ]
    assert controller.workers == []


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
def test_withdraw_and_restore_share_one_remaining_threshold(batch_size):
    pool = BorrowedRolloutPool(batch_size)
    for consumed in (0, batch_size, 2 * batch_size):
        target = consumed + max(1, batch_size - 1)
        assert pool.training_prefetch_target() == target
        pool.release(target - 1)
        assert {pool.choose(target - 1)[0] for _ in range(2)} == {0, 1}
        assert pool.choose(target) == (1, True)
        assert pool.choose(target) == (1, False)
        pool.batch_ready(batch_size)
