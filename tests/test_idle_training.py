"""Admission fencing and state handoff contracts for idle training."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.infra.workflow_executor import BatchTaskDispatcher
from areal.trainer.idle_budget import SwitchBudget
from areal.trainer.idle_training import IdleTrainingPool


def dispatcher(
    *, running=0, scoring=0, pending=False, capacity=0, queued=0, dispatching=False
):
    paused = threading.Event()
    runner = SimpleNamespace(
        paused=paused, get_input_queue_size=lambda: queued, pause=paused.set
    )
    manager = SimpleNamespace(
        lock=threading.Lock(),
        rollout_stat=SimpleNamespace(running=running),
        _generation_finished=set(range(scoring)),
        get_capacity=lambda: capacity,
    )
    return SimpleNamespace(
        _input_cv=threading.Condition(),
        _check_thread_exception=lambda: None,
        runner=runner,
        staleness_manager=manager,
        _pending_inputs=[1] if pending else [],
        _dispatching_input=dispatching,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(running=1),
        dict(queued=1),
        dict(dispatching=True),
        dict(pending=True, capacity=1),
        dict(running=3, scoring=2),
    ],
)
def test_idle_fence_rejects_generation(kwargs):
    value = dispatcher(**kwargs)
    assert not BatchTaskDispatcher.pause_if_generation_idle(value)
    assert not value.runner.paused.is_set()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(running=3, scoring=3),
        dict(pending=True, capacity=0, running=2, scoring=2),
    ],
)
def test_idle_fence_allows_reward_only_work(kwargs):
    value = dispatcher(**kwargs)
    assert BatchTaskDispatcher.pause_if_generation_idle(value)
    assert value.runner.paused.is_set()


def test_idle_fence_does_not_take_existing_pause():
    value = dispatcher()
    value.runner.paused.set()
    assert not BatchTaskDispatcher.pause_if_generation_idle(value)


@pytest.mark.parametrize("direction", ["single_to_multi", "multi_to_single"])
def test_handoff_overlaps_restore_with_source_release_and_rejects_corruption(
    tmp_path, direction
):
    from concurrent.futures import Future, ThreadPoolExecutor

    pool = object.__new__(IdleTrainingPool)
    pool._switch_budget = SwitchBudget()
    from areal.trainer.idle_entry import EntryBudget

    pool._entry_budget = EntryBudget()
    pool.helper = None
    pool.path = tmp_path
    before = dict(
        version=2,
        scheduler_epoch=2,
        optimizer_step_min=2,
        optimizer_step_max=2,
        sampled_state_sha256="before",
    )
    source, destination = Mock(), Mock()
    pool.primary = source if direction == "single_to_multi" else destination
    pool.multi = destination if direction == "single_to_multi" else source
    pool._device_keys = {"single": ["a"], "multi": ["a", "b"]}
    pool.snapshot_generation = 4
    pool._standby_destination = destination
    pool._standby = Future()
    pool._standby.set_result({"scheduler": {"last_epoch": 2}})
    pool._jobs = ThreadPoolExecutor(max_workers=2)
    source.idle_training_state.return_value = before
    destination.idle_training_state.return_value = {
        **before,
        "sampled_state_sha256": "bad",
    }
    entered, released = threading.Event(), threading.Event()

    def restore(*args):
        entered.set()
        assert released.wait(5)

    def release(*args):
        assert entered.wait(5)
        released.set()

    destination.idle_handoff.side_effect = restore
    source.idle_handoff.side_effect = release
    try:
        with pytest.raises(RuntimeError, match="sampled_state_sha256"):
            pool._transfer(source, destination, direction)
        source.idle_handoff.assert_called_once()
        destination.idle_handoff.assert_called_once()
    finally:
        pool._jobs.shutdown(wait=True)


@pytest.mark.parametrize("generation", [4, 5])
def test_standby_polls_without_blocking_source_and_rejects_stale_generation(
    tmp_path, generation
):
    from concurrent.futures import ThreadPoolExecutor

    pool = object.__new__(IdleTrainingPool)
    pool._switch_budget = SwitchBudget()
    from areal.trainer.idle_entry import EntryBudget

    pool._entry_budget = EntryBudget()
    pool.helper = None
    pool.path = tmp_path
    pool.snapshot_generation = 4
    pool._standby = None
    pool._jobs = ThreadPoolExecutor(max_workers=1)
    snapshot = {"version": 2, "scheduler": {"last_epoch": 2}, "copy_seconds": 0.1}
    pool.primary = Mock()
    pool.snapshot_actor = pool.primary
    pool.primary.idle_training_state.side_effect = [
        {"generation": generation, "done": False, "snapshot": None},
        {"generation": generation, "done": True, "snapshot": snapshot},
        snapshot,
    ]
    pool.multi = Mock()
    try:
        pool._prepare_standby()
        if generation == 4:
            assert pool._standby.result(timeout=5) == snapshot
            assert pool.multi.idle_training_state.call_count == 2
        else:
            with pytest.raises(RuntimeError, match="generation changed"):
                pool._standby.result(timeout=5)
            pool.multi.idle_training_state.assert_not_called()
        assert all(
            c.args in (("dram_poll",), ("dram_wait",))
            for c in pool.primary.idle_training_state.call_args_list
        )
    finally:
        pool._jobs.shutdown(wait=True)


def test_mid_batch_segments_preserve_packed_plan(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from areal.trainer.idle_training import IdleTrainingPool

    monkeypatch.setenv("AREAL_IDLE_MID_BATCH", "1")
    monkeypatch.setenv("AREAL_IDLE_CHUNK_ANSWERS", "32")
    calls = []

    def prepare_plan(method, data, path, token, **kwargs):
        Path(path).mkdir(parents=True)
        return [
            dict(denominator=256, answer_counts=[4] * 64, work_tokens=[400] * 64)
        ] * 32

    actor = SimpleNamespace(
        _custom_function_call=prepare_plan,
        ppo_update=lambda data, **kw: calls.append((len(data), kw["accumulation"])),
    )
    pool = object.__new__(IdleTrainingPool)
    pool._switch_budget = SwitchBudget()
    from areal.trainer.idle_entry import EntryBudget

    pool._entry_budget = EntryBudget()
    pool.helper = None
    pool.multi = object()
    from pathlib import Path

    pool.path = tmp_path
    pool.trainer = SimpleNamespace(
        config=SimpleNamespace(
            actor=SimpleNamespace(ppo_n_minibatches=1), total_train_steps=6
        ),
        rollout=SimpleNamespace(
            dispatcher=SimpleNamespace(
                pause_if_generation_idle=Mock(
                    side_effect=AssertionError(
                        "Budget rejection must not fence inference"
                    )
                )
            )
        ),
    )
    data = [{"input_ids": SimpleNamespace(shape=(8, 100))} for _ in range(32)]
    assert pool.ppo_update(actor, data, 2) is actor
    assert [n for n, _ in calls] == [32] * 8
    assert [(x["start_mb"], x["stop_mb"]) for _, x in calls] == [
        (i, i + 8) for i in range(0, 64, 8)
    ]
    assert sum(x["first"] for _, x in calls) == 1
    assert sum(x["last"] for _, x in calls) == 1
    assert {x["denominator"] for _, x in calls} == {256}


@pytest.mark.parametrize("microbatches", [64, 65])
def test_expanded_suffix_runs_once_without_repacking(
    monkeypatch, tmp_path, microbatches
):
    monkeypatch.setenv("AREAL_IDLE_SWITCH_BUDGET", "0")
    from pathlib import Path

    monkeypatch.setenv("AREAL_IDLE_MID_BATCH", "1")
    monkeypatch.setenv("AREAL_IDLE_CHUNK_ANSWERS", "32")
    counts = [4] * microbatches
    primary, multi = Mock(), Mock()

    def custom(method, *args, **kwargs):
        if method == "idle_prepare_plan":
            Path(args[1]).mkdir()
            return [
                dict(
                    denominator=sum(counts),
                    answer_counts=counts,
                    work_tokens=[400] * len(counts),
                )
            ]
        if method == "idle_export_gradients":
            Path(args[0]).mkdir()
        return {}

    primary._custom_function_call.side_effect = custom
    pool = object.__new__(IdleTrainingPool)
    pool._switch_budget = SwitchBudget()
    from areal.trainer.idle_entry import EntryBudget

    pool._entry_budget = EntryBudget()
    pool.helper = None
    pool.path, pool.primary, pool.multi = tmp_path, primary, multi
    pool._activate_multi = Mock()
    pool.trainer = SimpleNamespace(
        config=SimpleNamespace(
            actor=SimpleNamespace(ppo_n_minibatches=1), total_train_steps=6
        ),
        rollout=SimpleNamespace(
            dispatcher=SimpleNamespace(pause_if_generation_idle=lambda: True)
        ),
    )
    assert pool.ppo_update(primary, [{}] * 32, 2) is multi
    primary.ppo_update.assert_called_once()
    multi.ppo_update.assert_called_once()
    prefix = primary.ppo_update.call_args.kwargs["accumulation"]
    suffix = multi.ppo_update.call_args.kwargs["accumulation"]
    assert (prefix["start_mb"], prefix["stop_mb"]) == (0, 8)
    assert (suffix["start_mb"], suffix["stop_mb"]) == (8, microbatches)
    assert prefix["first"] and not prefix["last"]
    assert suffix["first"] and suffix["last"]
    assert suffix["denominator"] == prefix["denominator"] == sum(counts)
    assert suffix["plan_path"] == prefix["plan_path"]
    assert suffix["prefix_path"] is not None
    pool._activate_multi.assert_called_once_with(2)


def test_helper_keeps_exclusive_intervals_and_accepts_scalar_metric(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from unittest.mock import Mock

    from areal.trainer.idle_budget import SwitchBudget
    from areal.trainer.idle_training import IdleTrainingPool

    monkeypatch.setenv("AREAL_IDLE_HELPER_VALIDATE", "0")
    pool = object.__new__(IdleTrainingPool)
    pool.path = tmp_path
    pool.helper = Mock()

    def helper_rpc(name, *args):
        if name == "idle_export_gradients":
            from pathlib import Path

            Path(args[0]).mkdir()

    pool.helper._custom_function_call.side_effect = helper_rpc
    pool.primary = Mock()
    pool.primary.idle_training_state.return_value = {"version": 1}
    pool.primary._custom_function_call.return_value = 0.25
    pool.snapshot_actor = pool.primary
    pool.snapshot_generation = 2
    pool._helper_costs = []
    pool._switch_budget = SwitchBudget()
    pool._switch_budget.observe(1, 1, 100, 10)
    rollout = Mock()
    rollout.dispatcher.pause_if_generation_idle.return_value = True
    rollout.fixed_ranks = [1]
    pool.trainer = SimpleNamespace(rollout=rollout)
    pool._jobs = ThreadPoolExecutor(max_workers=2)
    try:
        assert pool._try_helper(
            pool.primary, [], 2, 2, [100] * 10, [1] * 10, "plan", "token", 100
        )
        helper_args = pool.helper.ppo_update.call_args.kwargs["accumulation"]
        primary_args = [
            c.kwargs["accumulation"] for c in pool.primary.ppo_update.call_args_list
        ]
        assigned = list(range(helper_args["start_mb"], helper_args["stop_mb"]))
        for a in primary_args:
            assigned.extend(range(a["start_mb"], a["stop_mb"]))
        assert sorted(assigned) == list(range(2, 10)) and len(assigned) == len(
            set(assigned)
        )
        assert not helper_args["last"]
        assert sum(a["last"] for a in primary_args) == 1
        assert pool._helper_costs and pool._helper_used
        rollout._group_rpc.assert_called_once_with(
            [1], "colocation_memory", action="restore"
        )
    finally:
        pool._jobs.shutdown(wait=True)
