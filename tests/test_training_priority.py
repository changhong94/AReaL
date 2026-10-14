# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.test_dynamic_kv import make

from areal.engine.vllm_ext.kv_slab import SlabFreeBlockQueue
from areal.trainer.colocation_memory import GIB, ColocationMemoryPolicy


def test_scheduler_capacity_bounds_ids_and_restores_only_when_idle():
    q = SlabFreeBlockQueue([SimpleNamespace(block_id=i) for i in range(40)], 8)
    q.set_capacity(12)
    blocks = q.popleft_n(11)
    assert {b.block_id for b in blocks} == set(range(1, 12))
    assert q.num_free_blocks == 0
    with pytest.raises(ValueError):
        q.popleft()
    with pytest.raises(RuntimeError):
        q.set_capacity(40)
    q.append_n(blocks)
    q.set_capacity(40)
    assert q.num_free_blocks == 39


def test_physical_budget_covers_null_padding_and_rejects_out_of_range():
    backend, pool = make()
    assert pool.set_budget(64) == 8
    pool.reconcile(set(range(8)))
    assert pool.snapshot()["mapped_bytes"] == 64
    with pytest.raises(ValueError):
        pool.reconcile({8})
    with pytest.raises(RuntimeError):
        pool.set_budget(None)
    pool.reconcile({0}, trim=True)
    assert pool.set_budget(None) == 20
    pool.reconcile({19})
    assert not pool.broken


def test_insufficient_kv_budget_does_not_mutate_mapping():
    _, pool = make()
    before = pool.snapshot()
    assert pool.set_budget(0) == 0
    assert pool.snapshot() == before


def policy(admitted=True, restore=0, cap=None):
    cfg = SimpleNamespace(
        train_workspace_gib=8,
        inference_workspace_gib=2,
        safety_margin_gib=2,
        device_budget_gib=cap,
    )
    actor = Mock(
        return_value=dict(
            free_bytes=30 * GIB,
            restore_bytes=restore,
            future_optimizer_bytes=2 * GIB,
            gradient_bytes=3 * GIB,
        )
    )
    status = dict(
        free_bytes=30 * GIB,
        total_bytes=64 * GIB,
        restore_bytes=0,
        dynamic_kv={"mapped_bytes": GIB},
    )
    rollout = Mock(
        side_effect=lambda action, **kw: status
        if action == "status"
        else dict(
            free_bytes=30 * GIB, concurrent_admitted=admitted, capacity_blocks=100
        )
    )
    return ColocationMemoryPolicy(cfg, actor, rollout, Mock()), actor, rollout


def test_budget_reserves_training_optimizer_gradients_inference_and_margin():
    p, actor, rollout = policy()
    assert p.concurrent_train()
    rollout.assert_any_call("kv_budget", kv_budget_bytes=14 * GIB)
    actor.assert_called_once_with("status")  # Never reclaim actor during PPO.


def test_device_cap_limits_concurrent_budget():
    p, _, rollout = policy(cap=60)
    assert p.concurrent_train()
    rollout.assert_any_call("kv_budget", kv_budget_bytes=10 * GIB)


def test_insufficient_capacity_restores_scheduler_limit_for_migration():
    p, _, rollout = policy(admitted=False)
    assert not p.concurrent_train()
    rollout.assert_any_call("kv_budget_clear")


def test_partial_model_residency_never_resumes_inference():
    p, _, rollout = policy(restore=GIB)
    assert not p.concurrent_train()
    rollout.assert_called_once_with("status")


def test_overlap_admission_requires_successful_memory_budget(monkeypatch):
    from areal.trainer.rl_trainer import PPOTrainer

    monkeypatch.setenv("AREAL_TRAIN_PRIORITY_INFERENCE", "1")
    monkeypatch.setenv("AREAL_NPU_SHARE_PAUSE_KEEP", "0")
    trainer = object.__new__(PPOTrainer)
    trainer._memory_policy = Mock()
    trainer.rollout = Mock()
    trainer._memory_policy.concurrent_train.return_value = False
    trainer._resume_training_inference()
    trainer.rollout.resume_training_inference.assert_not_called()
    trainer._memory_policy.concurrent_train.return_value = True
    trainer._resume_training_inference()
    trainer.rollout.resume_training_inference.assert_called_once()
    assert trainer._training_inference_active


def test_publication_stops_inference_before_lifting_kv_cap():
    from unittest.mock import call

    from areal.trainer.rl_trainer import PPOTrainer

    trainer = object.__new__(PPOTrainer)
    trainer.rollout = Mock()
    trainer._training_inference_active = True
    trainer._stop_training_inference()
    assert trainer.rollout.mock_calls == [
        call.borrow_for_training(),
        call.colocation_memory("kv_budget_clear"),
    ]
    assert not trainer._training_inference_active
    trainer._stop_training_inference()
    assert len(trainer.rollout.mock_calls) == 2


def test_native_vllm_reset_and_usage_follow_reduced_capacity():
    pytest.importorskip("vllm")
    from vllm.v1.core.block_pool import BlockPool

    from areal.engine.vllm_ext.kv_slab import set_scheduler_capacity

    pool = BlockPool(num_gpu_blocks=40, enable_caching=False, hash_block_size=16)
    pool.free_block_queue = SlabFreeBlockQueue(pool.blocks, 8)
    with pytest.raises(ValueError):
        set_scheduler_capacity(pool, 12, watermark_blocks=1)
    assert pool.num_gpu_blocks == 40
    set_scheduler_capacity(pool, 12)
    assert pool.get_usage() == 0
    blocks = pool.get_new_blocks(5)
    assert not pool.reset_prefix_cache()
    with pytest.raises(RuntimeError):
        set_scheduler_capacity(pool, 40)
    assert pool.num_gpu_blocks == 12
    pool.free_blocks(blocks)
    assert pool.reset_prefix_cache()
    assert pool.get_usage() == 0
    set_scheduler_capacity(pool, 40)
    assert pool.reset_prefix_cache()
    assert pool.get_num_free_blocks() == 39
