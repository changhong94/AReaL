# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest

from tests.test_background_step_tail import trainer as background_trainer

from areal.api.alloc_mode import ModelAllocation
from areal.infra.borrowed_rollout import BorrowedRolloutPool
from areal.trainer.training_prefetch import TrainingRestorePrefetch


@pytest.fixture
def trainer():
    return background_trainer.__wrapped__()


def test_target_matches_existing_withdraw_window():
    pool = BorrowedRolloutPool(16)
    assert pool.training_prefetch_target() == 15
    pool.batch_ready(16)
    assert pool.training_prefetch_target() == 31


def test_restore_finishes_before_answers_are_ready(trainer, monkeypatch, tmp_path):
    monkeypatch.setenv("AREAL_REQUEST_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("AREAL_PARK_TRAIN_CACHE", "1")
    monkeypatch.setenv("AREAL_OVERLAP_TRAIN_RESTORE", "1")
    trainer.rollout_alloc = ModelAllocation.from_str("vllm:d2p1t1")
    trainer.train_dataloader = [[]]
    window, restored = Event(), Event()
    trainer.rollout.training_prefetch_target.return_value = 15
    trainer.rollout.training_prefetch_ready.side_effect = lambda target: window.is_set()
    trainer._memory_policy.train.side_effect = restored.set

    def answers(*args, **kwargs):
        window.set()
        assert restored.wait(5), "restoration must overlap answer preparation"
        trainer.actor.ppo_update.assert_not_called()
        # Controller receives remote tensor handles, not materialized tensors.
        return [{"attention_mask": object(), "loss_mask": object()}]

    trainer.actor.prepare_batch.side_effect = answers
    trainer.train(workflow="test")
    trainer.rollout.borrow_for_training.assert_called_once()
    trainer._memory_policy.train.assert_called_once()
    trainer.actor.ppo_update.assert_called_once()
    assert trainer._train_prefetch is None


def test_input_failure_cancels_waiting_prefetch(trainer, monkeypatch):
    monkeypatch.setenv("AREAL_PARK_TRAIN_CACHE", "1")
    monkeypatch.setenv("AREAL_OVERLAP_TRAIN_RESTORE", "1")
    trainer.rollout_alloc = ModelAllocation.from_str("vllm:d2p1t1")
    trainer.rollout.training_prefetch_ready.return_value = False
    trainer.actor.prepare_batch.side_effect = ValueError("bad input")
    with pytest.raises(ValueError, match="bad input"):
        trainer.train(workflow="test")
    trainer.rollout.borrow_for_training.assert_not_called()
    assert trainer._train_prefetch is None


def test_prefetch_queues_after_previous_actor_owner():
    previous_started, release, prepared = Event(), Event(), Event()

    def previous():
        previous_started.set()
        assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        prior = executor.submit(previous)
        assert previous_started.wait(2)
        job = TrainingRestorePrefetch(executor, lambda: True, prepared.set, Mock(), 2)
        try:
            assert not prepared.wait(0.02)
        finally:
            release.set()
        prior.result(timeout=5)
        assert job.join(force=True) and prepared.is_set()


def test_failed_restore_wakes_dispatcher_and_propagates():
    fail = Mock()
    with ThreadPoolExecutor(max_workers=1) as executor:
        job = TrainingRestorePrefetch(
            executor,
            lambda: True,
            Mock(side_effect=RuntimeError("restore failed")),
            fail,
            1,
        )
        with pytest.raises(RuntimeError, match="restore failed"):
            job.join()
    fail.assert_called_once()


def test_cancel_before_ready_never_borrows():
    prepare = Mock()
    with ThreadPoolExecutor(max_workers=1) as executor:
        job = TrainingRestorePrefetch(executor, lambda: False, prepare, Mock(), 1)
        job.cancel()
        assert not job.join()
    prepare.assert_not_called()


def test_completed_batch_forces_preparation_once():
    prepare = Mock()
    with ThreadPoolExecutor(max_workers=1) as executor:
        job = TrainingRestorePrefetch(executor, lambda: False, prepare, Mock(), 1)
        assert job.join(force=True)
        assert job.join()
    prepare.assert_called_once()
