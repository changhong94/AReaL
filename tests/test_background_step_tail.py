# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest

from tests.test_asymmetric_publication import fixed_meta
from tests.test_resident_colocation import trainer as resident_trainer


@pytest.fixture
def trainer():
    obj = resident_trainer.__wrapped__()
    obj._fixed_weight_update_meta = fixed_meta()
    obj.recover_info = None
    obj.train_dataloader = [[], []]
    obj.valid_dataloader = None
    obj.config.total_train_epochs = 1
    obj.ref = None
    obj.teachers = []
    obj.data_controller = None
    obj.saver = Mock()
    obj._requires_proxy_workflow = Mock(return_value=False)
    obj._save_hf = Mock()
    obj._save_recover_checkpoint = Mock()
    obj._save_perf_tracer = Mock()
    obj._export_and_commit_stats = Mock()
    obj.actor.prepare_batch.return_value = []
    obj.actor.compute_logp.return_value = []
    return obj


@pytest.mark.parametrize("data_service", [False, True])
def test_next_prepare_overlaps_memory_switch_but_training_waits_for_tail(
    trainer, data_service
):
    """Main thread enters next batch while the previous memory switch owns the actor."""
    snapshot_started, next_prepare, release, second_update = (
        Event(),
        Event(),
        Event(),
        Event(),
    )
    prepare_count = 0
    update_count = 0
    if data_service:
        trainer.data_controller = Mock()

    def prepare(*args, **kwargs):
        nonlocal prepare_count
        prepare_count += 1
        if prepare_count == 2:
            if data_service:
                trainer.data_controller.clear_batches.assert_called_once()
            assert snapshot_started.wait(5)
            next_prepare.set()
        return []

    def switch():
        if not snapshot_started.is_set():
            snapshot_started.set()
            assert release.wait(5)

    def update(*args):
        nonlocal update_count
        update_count += 1
        if update_count == 2:
            second_update.set()

    trainer.actor.prepare_batch.side_effect = prepare
    trainer._memory_policy.inference.side_effect = switch
    trainer.actor.ppo_update.side_effect = update
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(trainer.train, workflow="test")
        try:
            assert next_prepare.wait(5)
            assert not second_update.wait(0.05)
            assert trainer._export_and_commit_stats.call_count == 0
        finally:
            release.set()
        result.result(timeout=5)
    assert second_update.is_set()
    assert trainer._export_and_commit_stats.call_count == 2
    assert trainer._step_tail is None


def test_background_switch_failure_wakes_waiter_and_prevents_second_update(trainer):
    """A snapshot failure must interrupt prepare_batch instead of hanging it."""
    waiting, failed = Event(), Event()
    count = 0

    def prepare(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            waiting.set()
            assert failed.wait(5)
            raise RuntimeError("snapshot failed")
        return []

    def switch():
        assert waiting.wait(5)
        raise RuntimeError("snapshot failed")

    trainer.actor.prepare_batch.side_effect = prepare
    trainer._memory_policy.inference.side_effect = switch
    trainer.rollout.dispatcher.fail_fast.side_effect = lambda exc: failed.set()
    with pytest.raises(RuntimeError, match="snapshot failed"):
        trainer.train(workflow="test")
    assert trainer.actor.ppo_update.call_count == 1
    trainer.rollout.dispatcher.pause.assert_called_once()
    trainer.rollout.load_shared_snapshot_async.assert_not_called()


def test_evaluation_step_is_synchronous_and_frequency_is_checked_once(trainer):
    """Evaluation holds the step barrier and does not consume its schedule twice."""
    trainer.valid_dataloader = []
    trainer.evaluator = Mock()
    trainer.evaluator.evaluate.side_effect = lambda fn, *args: fn()
    trainer._evaluate_fn = Mock()
    trainer.actor.prepare_batch.side_effect = [[], []]
    trainer.train(workflow="test", eval_workflow="eval")
    assert trainer.evaluator.evaluate.call_count == 2
    assert trainer._evaluate_fn.call_count == 2
    assert trainer._step_tail is None


def test_final_step_skips_publication_but_preserves_saving_and_stats(trainer):
    """No inference consumer remains, but saving and cleanup must still run."""
    trainer.train_dataloader = [[]]
    trainer.train(workflow="test")
    trainer.rollout.load_shared_snapshot_async.assert_not_called()
    trainer.actor.update_weights.assert_not_called()
    trainer._memory_policy.inference.assert_not_called()
    trainer.rollout.resume.assert_not_called()
    trainer._save_hf.assert_called_once()
    trainer._save_recover_checkpoint.assert_called_once()
    trainer.actor.clear_batches.assert_called_once()
    trainer.actor.set_version.assert_called_once_with(1)
    trainer._export_and_commit_stats.assert_called_once()
    assert trainer._step_tail is None


def test_main_prepare_failure_still_drains_background(trainer):
    """An input failure cannot leave a background RPC racing worker shutdown."""
    trainer.actor.prepare_batch.side_effect = [[], ValueError("input failed")]
    with pytest.raises(ValueError, match="input failed"):
        trainer.train(workflow="test")
    trainer._export_and_commit_stats.assert_called_once()
    assert trainer._step_tail is None


def test_data_service_cleanup_precedes_fixed_publication(trainer):
    """Sweep inputs before opening next reads, and never repeat it in the tail."""
    trainer.data_controller = Mock()
    prepared = 0

    def prepare(*args, **kwargs):
        nonlocal prepared
        assert trainer.data_controller.clear_batches.call_count == prepared
        prepared += 1
        return []

    trainer.actor.prepare_batch.side_effect = prepare

    def fixed_ready(version):
        assert trainer.data_controller.clear_batches.call_count == version

    trainer.rollout.publish_fixed_version.side_effect = fixed_ready
    trainer.train(workflow="test")
    assert prepared == 2
    assert trainer.data_controller.clear_batches.call_count == 2


@pytest.mark.parametrize("failure", ["memory", "load", "stats"])
def test_background_tail_stage_failure_stops_next_training(trainer, failure):
    """Any deferred stage fails the run without allowing another actor update."""
    target = {
        "memory": trainer._memory_policy.inference,
        "load": trainer.rollout.load_shared_snapshot_async,
        "stats": trainer._export_and_commit_stats,
    }[failure]
    target.side_effect = RuntimeError(failure)
    with pytest.raises(RuntimeError, match=failure):
        trainer.train(workflow="test")
    assert trainer.actor.ppo_update.call_count == 1
    trainer.rollout.dispatcher.fail_fast.assert_called_once()


def test_next_rollout_timing_is_not_committed_to_previous_step(trainer, monkeypatch):
    """A fast next prepare may complete before old stats, but records after join."""
    from areal.utils import stats_tracker

    records, commits = [], []
    monkeypatch.setattr(stats_tracker, "scalar", lambda **kw: records.append(kw))
    trainer._export_and_commit_stats.side_effect = lambda **kw: commits.append(
        len(records)
    )
    trainer.train(workflow="test")
    assert commits == [1, 2]
    assert all("timeperf/rollout" in record for record in records)


@pytest.mark.parametrize(
    "epochs,limit,expected_steps", [(2, None, 4), (3, 3, 3), (1, 9, 2)]
)
def test_only_actual_final_step_skips_publication(
    trainer, epochs, limit, expected_steps
):
    trainer.config.total_train_epochs = epochs
    trainer.config.total_train_steps = limit
    trainer.train(workflow="test")
    assert trainer.actor.ppo_update.call_count == expected_steps
    assert trainer.actor.update_weights.call_count == expected_steps - 1
    assert trainer.rollout.load_shared_snapshot_async.call_count == expected_steps - 1
    assert trainer.rollout.resume.call_count == expected_steps - 1
    assert trainer._export_and_commit_stats.call_count == expected_steps
    assert trainer._step_tail is None


def test_final_evaluation_still_publishes_and_loads_shared_weights(trainer):
    trainer.train_dataloader = [[]]
    trainer.valid_dataloader = []
    trainer.evaluator = Mock()
    trainer.evaluator.evaluate.side_effect = lambda fn, *args: fn()
    trainer._evaluate_fn = Mock()
    trainer.train(workflow="test", eval_workflow="eval")
    trainer.actor.update_weights.assert_called_once()
    trainer.rollout.load_shared_snapshot_async.assert_called_once()
    trainer._memory_policy.inference.assert_called_once()
    trainer._evaluate_fn.assert_called_once()
