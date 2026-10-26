"""Direct training-world selection and incremental entry cost contracts."""

from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.trainer.idle_entry import EntryBudget
from areal.trainer.idle_training import IdleTrainingPool


def test_entry_charges_only_extra_exposed_preparation():
    budget = EntryBudget()
    for world, compute, prepare, returning in [(1, 60, 10, 0), (2, 40, 12, 6)]:
        for phase, seconds in [
            ("compute", compute),
            ("prepare", prepare),
            ("return", returning),
        ]:
            budget.observe(1, world, phase, seconds)
    decision = budget.decide(2, 0)
    assert decision["world_size"] == 2
    assert decision["incremental_prepare_seconds"] == 2
    assert decision["net_seconds"] == 12


def test_entry_does_not_take_busy_inference_or_unprofitable_world():
    budget = EntryBudget()
    assert budget.decide(1, 1)["world_size"] == 1
    budget.observe(1, 1, "compute", 40)
    budget.observe(1, 2, "compute", 39)
    budget.observe(1, 2, "return", 6)
    assert budget.decide(2, 0)["world_size"] == 1


def test_entry_expires_old_workload_observations():
    budget = EntryBudget()
    budget.observe(1, 1, "compute", 60)
    assert budget.decide(5, 0)["reason"] == "idle_cold_start"


@pytest.mark.parametrize("dual", [True, False])
def test_entry_prepares_only_selected_world(dual):
    pool = object.__new__(IdleTrainingPool)
    pool.primary, pool.multi = object(), object()
    pool._entry_budget = EntryBudget()
    pool.active = False
    pool._restore_multi_from_dram = Mock()
    pool._restore_single_from_dram = Mock()
    rollout = Mock()
    rollout.dispatcher.generation_work_snapshot.return_value = {
        "generating": 0 if dual else 1
    }
    rollout.dispatcher.pause_if_generation_idle.return_value = True
    pool.trainer = SimpleNamespace(
        rollout=rollout,
        _memory_policy=Mock(),
        config=SimpleNamespace(total_train_steps=6),
    )
    pool._prepare_entry(1)
    if dual:
        pool.trainer._memory_policy.train.assert_not_called()
        rollout.borrow_for_training.assert_not_called()
        pool._restore_multi_from_dram.assert_called_once()
        assert pool._entry_actor is pool.multi
    else:
        pool._restore_multi_from_dram.assert_not_called()
        rollout.dispatcher.pause_if_generation_idle.assert_not_called()
        pool._restore_single_from_dram.assert_called_once()
        assert pool._entry_actor is pool.primary


def test_direct_entry_never_inspects_or_restores_primary_hbm(tmp_path):
    pool = object.__new__(IdleTrainingPool)
    pool.path = tmp_path
    pool.primary, pool.multi = Mock(), Mock()
    pool.snapshot_actor = pool.primary
    pool.snapshot_generation = 4
    pool._device_keys = {"single": ["a"], "multi": ["a", "b"]}
    pool._standby_destination = pool.multi
    pool._standby = Future()
    pool._standby.set_result({"scheduler": {"last_epoch": 3}})
    progress = dict(
        version=4, scheduler_epoch=3, snapshot_generation=4, is_offload=True
    )
    pool.primary.idle_training_state.side_effect = [
        progress,
        {"scheduler": {"last_epoch": 3}},
    ]
    pool.multi.idle_training_state.return_value = {**progress, "is_offload": False}
    with ThreadPoolExecutor(max_workers=2) as jobs:
        pool._jobs = jobs
        pool._restore_multi_from_dram()
    assert [c.args[0] for c in pool.primary.idle_training_state.call_args_list] == [
        "dram_progress",
        "dram_wait",
    ]
    pool.primary.onload.assert_not_called()
    pool.primary.idle_handoff.assert_called_once()
    assert pool._standby is None


def test_entry_rejects_stale_snapshot_before_device_operations(tmp_path):
    pool = object.__new__(IdleTrainingPool)
    pool.path = tmp_path
    pool.primary, pool.multi = Mock(), Mock()
    pool._standby_destination = pool.multi
    pool._standby = Future()
    pool._standby.set_result({"scheduler": {"last_epoch": 2}})
    pool.snapshot_actor = pool.primary
    pool.snapshot_generation = 4
    pool.primary.idle_training_state.side_effect = [
        dict(scheduler_epoch=3, snapshot_generation=4),
        {"scheduler": {"last_epoch": 2}},
    ]
    with pytest.raises(RuntimeError, match="latest optimizer"):
        pool._restore_multi_from_dram()
    pool.primary.idle_handoff.assert_not_called()
    pool.multi.idle_handoff.assert_not_called()


def test_authoritative_multi_reuses_own_backing_without_primary_restore():
    pool = object.__new__(IdleTrainingPool)
    pool.primary, pool.multi = Mock(), Mock()
    pool._entry_snapshot = Mock(return_value=(dict(version=3, is_offload=True), False))
    pool._restore_multi_from_dram()
    pool.multi.onload.assert_called_once()
    pool.multi.idle_training_state.assert_called_once_with("dram_verify_own")
    pool.primary.onload.assert_not_called()
    pool.primary.idle_handoff.assert_not_called()


def test_cpu_publication_preserves_weights_without_optimizer_leak(tmp_path):
    import torch

    from areal.engine.idle_state import manifest, model_snapshot_parameters
    from areal.engine.weight_snapshot import open_snapshot, write_snapshot

    values = {
        "weight/model.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "optimizer/model.weight/exp_avg": torch.ones(3, 4),
    }
    layout = manifest(values)
    data = torch.empty(layout["bytes"], dtype=torch.uint8)
    for name, value in values.items():
        entry = layout["entries"][name]
        data.narrow(0, entry["offset"], entry["size"]).copy_(
            value.reshape(-1).view(torch.uint8)
        )
    parameters = model_snapshot_parameters(data, layout)
    assert [name for name, _ in parameters] == ["model.weight"]
    path = str(tmp_path / "publication")
    write_snapshot(
        path, 3, parameters, torch.bfloat16, lambda value: value.to(torch.bfloat16)
    )
    with open_snapshot(path, 3) as iterator:
        loaded = dict(iterator)
        torch.testing.assert_close(
            loaded["model.weight"],
            values["weight/model.weight"].to(torch.bfloat16),
            rtol=0,
            atol=0,
        )


def test_cpu_publication_write_failure_reaches_every_rank(monkeypatch):
    import torch
    import torch.distributed as dist

    from areal.engine import weight_snapshot
    from areal.engine.fsdp_engine import FSDPEngine
    from areal.engine.idle_state import manifest

    values = {"weight/model.weight": torch.ones(2)}
    layout = manifest(values)
    snapshot = dict(generation=1, scheduler={"last_epoch": 2})
    group = object()
    engine = SimpleNamespace(
        _idle_dram=SimpleNamespace(
            wait=lambda: snapshot,
            data=torch.empty(layout["bytes"], dtype=torch.uint8),
            layout=layout,
        ),
        cpu_group=group,
        rank=0,
        lr_scheduler=SimpleNamespace(last_epoch=2),
        is_vision_model=False,
        config=SimpleNamespace(use_lora=False),
        _compute_dtype=lambda: torch.bfloat16,
    )
    monkeypatch.setattr(dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(dist, "get_global_rank", lambda group, rank: 0)
    errors = []
    monkeypatch.setattr(
        dist, "broadcast_object_list", lambda value, **kwargs: errors.extend(value)
    )

    def fail(*args, **kwargs):
        raise OSError("injected snapshot failure")

    monkeypatch.setattr(weight_snapshot, "write_snapshot", fail)
    with pytest.raises(RuntimeError, match="injected snapshot failure"):
        FSDPEngine.idle_publication_snapshot(engine, "/unused", 3)
    assert errors == ["OSError: injected snapshot failure"]


def test_entry_keeps_cost_fallback_but_skips_terminal_publication():
    budget = EntryBudget()
    budget.observe(1, 2, "return", 12)
    budget.observe(4, 1, "return", 2)
    budget.observe(5, 1, "compute", 30)
    budget.observe(5, 2, "compute", 25)
    regular = budget.decide(5, 0)
    assert regular["world_size"] == 1
    assert regular["incremental_return_seconds"] == 10
    assert regular["publication_source"] == "last_known_fallback"
    terminal = budget.decide(5, 0, terminal=True)
    assert terminal["world_size"] == 2
    assert terminal["incremental_return_seconds"] == 0


@pytest.mark.parametrize("asynchronous", [True, False])
def test_multi_publication_restores_only_required_group_before_background(
    asynchronous, monkeypatch
):
    from dataclasses import dataclass, replace

    @dataclass
    class Meta:
        type: str = "disk"
        path: str = "/unused"
        version: int = 0
        npu_share_snapshot: bool = False
        clear_checkpoint_after_load: bool = True

        def with_version(self, version):
            return replace(self, version=version)

    monkeypatch.setenv("AREAL_IDLE_ASYNC_TAIL", "1" if asynchronous else "0")
    pool = object.__new__(IdleTrainingPool)
    rollout = Mock()
    rollout.fixed_ranks = [1]
    pool.trainer = SimpleNamespace(rollout=rollout, weight_update_meta=Meta())
    pool.current = Mock()
    pool._entry_budget = Mock()
    pool.active = True
    pool.publish_multi(3)
    if asynchronous:
        rollout._collective_rpc.assert_not_called()
        assert rollout._group_rpc.call_args_list[0].args == ([1], "colocation_memory")
        assert rollout.load_shared_snapshot_async.call_args.kwargs == {
            "restore_memory": True
        }
    else:
        rollout._collective_rpc.assert_called_once_with(
            "colocation_memory", action="restore"
        )
        assert not rollout.load_shared_snapshot_async.call_args.kwargs
    # The only wait is for a previous publication, before this generation starts.
    rollout.wait_shared_publication.assert_called_once()
    assert not pool.active
