# SPDX-License-Identifier: Apache-2.0

from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.test_resident_colocation import trainer as resident_trainer

from areal.api import WeightUpdateMeta, Worker
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import InferenceEngineConfig
from areal.infra.controller.rollout_controller import RolloutController


@pytest.fixture
def controller():
    scheduler = Mock()
    scheduler.async_call_engine = AsyncMock()
    obj = RolloutController(
        Mock, InferenceEngineConfig(backend="vllm:d2p1t1"), scheduler
    )
    obj._worker_role = "rollout"
    obj.workers = [
        Worker(id="shared", ip="localhost"),
        Worker(id="fixed", ip="localhost"),
    ]
    obj.enable_npu_share_pool()
    obj._dispatcher = Mock()
    obj._staleness_manager = Mock()
    obj._staleness_manager.get_stats.return_value.accepted = 0
    obj._worker_rpc = Mock()
    obj._pool_event = Mock()
    return obj


def fixed_meta():
    meta = WeightUpdateMeta.from_fsdp_xccl(ModelAllocation.from_str("vllm:d1p1t1"))
    meta.npu_share_fixed = True
    return meta


@pytest.mark.asyncio
async def test_fixed_group_excludes_shared_and_remaps_logical_rank(controller):
    meta = fixed_meta()
    await controller.init_weights_update_group(meta)
    args = controller.scheduler.async_call_engine.call_args.kwargs
    assert (args["worker_id"], args["engine_name"], args["xccl_group_ranks"]) == (
        "fixed",
        "rollout/1",
        [0],
    )
    controller.scheduler.async_call_engine.reset_mock()
    await controller.update_weights_from_distributed(meta, [])
    controller.scheduler.async_call_engine.assert_awaited_once()
    assert (
        controller.scheduler.async_call_engine.call_args.kwargs["worker_id"] == "fixed"
    )


def test_fixed_ready_does_not_mark_or_resume_shared(controller):
    controller.prepare_fixed_publication()
    controller.publish_fixed_version(3)
    assert controller.get_version() == 3
    assert all(c.args[0] == 1 for c in controller._worker_rpc.call_args_list)
    assert controller._borrowed_pool.choose(0)[0] == 1
    methods = [c.args[1] for c in controller._worker_rpc.call_args_list]
    assert methods == [
        "pause_generation",
        "set_version",
        "continue_generation",
        "resume",
    ]


def test_shared_pending_is_excluded_and_snapshot_survives_until_ack(
    controller, tmp_path
):
    started, complete = Event(), Event()
    snapshot = tmp_path / "v3"
    snapshot.mkdir()
    (snapshot / "weights").write_bytes(b"owned")
    controller.prepare_fixed_publication()
    controller.publish_fixed_version(3)
    controller._worker_rpc.reset_mock()

    def rpc(rank, method, **kwargs):
        if method == "update_weights_from_disk":
            started.set()
            assert complete.wait(5)

    controller._worker_rpc.side_effect = rpc
    meta = WeightUpdateMeta(
        type="disk", path=str(snapshot), version=3, npu_share_snapshot=True
    )
    controller.load_shared_snapshot_async(meta)
    try:
        assert started.wait(5)
        controller.resume()
        assert snapshot.exists()
        assert controller._borrowed_pool.choose(0)[0] == 1
        assert not any(
            c.args == (0, "resume") for c in controller._worker_rpc.call_args_list
        )
    finally:
        complete.set()
        controller.wait_shared_publication()
    assert not snapshot.exists()
    assert {controller._borrowed_pool.choose(0)[0] for _ in range(2)} == {0, 1}
    methods = [
        c.args[1] for c in controller._worker_rpc.call_args_list if c.args[0] == 0
    ]
    assert methods == [
        "update_weights_from_disk",
        "set_version",
        "continue_generation",
        "resume",
    ]


def test_shared_failure_never_rejoins_or_allows_next_train(controller, tmp_path):
    snapshot = tmp_path / "v2"
    snapshot.mkdir()
    controller.prepare_fixed_publication()
    controller.publish_fixed_version(2)
    controller._worker_rpc.reset_mock()
    controller._worker_rpc.side_effect = RuntimeError("load failed")
    controller.load_shared_snapshot_async(
        WeightUpdateMeta(
            type="disk", path=str(snapshot), version=2, npu_share_snapshot=True
        )
    )
    with pytest.raises(RuntimeError, match="load failed"):
        controller.wait_shared_publication()
    with pytest.raises(RuntimeError, match="load failed"):
        controller.borrow_for_training()
    with pytest.raises(RuntimeError, match="load failed"):
        controller.resume()
    assert controller._borrowed_pool.choose(0)[0] == 1
    assert snapshot.exists()
    controller.dispatcher.pause.assert_called()
    controller.dispatcher.fail_fast.assert_called_once()
    assert len(controller._worker_rpc.call_args_list) == 1


def test_mismatched_snapshot_is_rejected_before_loading(controller):
    controller.publish_fixed_version(2)
    with pytest.raises(ValueError, match="match"):
        controller.load_shared_snapshot_async(
            WeightUpdateMeta(type="disk", version=1, npu_share_snapshot=True)
        )
    assert controller._shared_publication is None


def test_next_training_joins_pending_publication_before_borrow(controller):
    order = []
    controller._shared_publication = Mock()
    controller._shared_publication.result.side_effect = lambda: order.append("join")
    controller._worker_rpc.side_effect = lambda rank, method, **kw: order.append(method)
    controller.borrow_for_training()
    assert order == ["join", "pause", "pause_generation"]


def test_trainer_fuses_snapshot_with_fixed_update_before_shared_memory_switch():
    trainer = resident_trainer.__wrapped__()
    trainer._fixed_weight_update_meta = fixed_meta()
    trainer.eval_rollout = Mock()
    calls = Mock()
    calls.attach_mock(trainer.actor, "actor")
    calls.attach_mock(trainer.rollout, "rollout")
    calls.attach_mock(trainer._memory_policy, "memory")
    trainer._publish_weights(2)
    names = [c[0] for c in calls.mock_calls]
    updates = [i for i, name in enumerate(names) if name == "actor.update_weights"]
    assert names.index("rollout.prepare_fixed_publication") < updates[0]
    assert len(updates) == 1
    meta = trainer.actor.update_weights.call_args.args[0]
    assert meta.npu_share_snapshot and meta.npu_share_fixed
    assert meta.path == trainer.weight_update_meta.with_version(2).path
    assert (
        updates[0]
        < names.index("rollout.publish_fixed_version")
        < names.index("memory.inference")
        < names.index("rollout.load_shared_snapshot_async")
    )
    trainer.rollout.set_version.assert_not_called()
    trainer.rollout.continue_generation.assert_not_called()


def test_fixed_update_failure_never_resumes_or_starts_snapshot():
    trainer = resident_trainer.__wrapped__()
    trainer._fixed_weight_update_meta = fixed_meta()
    trainer.actor.update_weights.side_effect = RuntimeError("HCCL failed")
    with pytest.raises(RuntimeError, match="HCCL failed"):
        trainer._publish_weights(2)
    trainer.rollout.publish_fixed_version.assert_not_called()
    trainer.rollout.load_shared_snapshot_async.assert_not_called()
    trainer._memory_policy.inference.assert_not_called()


@pytest.mark.parametrize("snapshot", [True, False])
def test_snapshot_load_skips_legacy_actor_stage_one_notification(monkeypatch, snapshot):
    """Atomic snapshots must load without the actor's obsolete name-resolve signal."""
    from areal.infra import remote_inf_engine

    rendezvous = Mock(return_value="1.0")
    if snapshot:
        rendezvous.side_effect = AssertionError("Unexpected legacy rendezvous")
    monkeypatch.setattr(remote_inf_engine.name_resolve, "wait", rendezvous)
    backend = Mock()
    backend.build_disk_weight_update_requests.return_value = SimpleNamespace(
        requests=[]
    )
    meta = WeightUpdateMeta(type="disk", version=1, npu_share_snapshot=snapshot)
    remote_inf_engine._update_weights_from_disk(
        backend, "experiment", "trial", 0, [], meta, 1, 10.0
    )
    assert rendezvous.call_count == int(not snapshot)
    backend.build_disk_weight_update_requests.assert_called_once_with(meta)


def test_snapshot_wait_occurs_after_fixed_resume_before_memory_switch():
    trainer = resident_trainer.__wrapped__()
    trainer._fixed_weight_update_meta = fixed_meta()
    calls = Mock()
    calls.attach_mock(trainer.actor, "actor")
    calls.attach_mock(trainer.rollout, "rollout")
    calls.attach_mock(trainer._memory_policy, "memory")
    trainer._publish_weights_asymmetric(3)
    names = [c[0] for c in calls.mock_calls]
    assert names.index("rollout.publish_fixed_version") < names.index(
        "actor.wait_shared_snapshot"
    )
    assert names.index("actor.wait_shared_snapshot") < names.index("memory.inference")
