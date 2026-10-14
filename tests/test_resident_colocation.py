# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import pytest

from areal.api import WeightUpdateMeta
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GRPOConfig, SchedulingStrategy, load_expr_config
from areal.trainer.rl_trainer import PPOTrainer


@pytest.fixture
def trainer():
    """Use real trainer handoff methods with RPC boundaries replaced by mocks."""
    trainer = object.__new__(PPOTrainer)
    config = GRPOConfig()
    config.npu_share = True
    config.scheduler.type = "local"
    config.actor.weight_update_mode = "disk"
    config.actor._version = config.rollout._version = "v1"
    config.rollout.max_head_offpolicyness = 0
    config.rollout.scheduling_strategy = SchedulingStrategy(
        type="colocation", target="actor", fork=True
    )
    config.vllm.enforce_eager = True
    config.recover.mode = "disabled"
    trainer.config = config
    trainer.actor_alloc = ModelAllocation.from_str("fsdp:d1p1t1")
    trainer.rollout_alloc = ModelAllocation.from_str("vllm:d1p1t1")
    trainer.actor = Mock()
    trainer._memory_policy = Mock()
    trainer.rollout = Mock()
    trainer.eval_rollout = Mock()
    trainer.critic = None
    trainer._should_offload_actor = False
    trainer._should_offload_rollout = True
    trainer.weight_update_meta = WeightUpdateMeta.from_disk(
        experiment_name="test",
        trial_name="resident",
        file_root="/tmp/test",
        name="default",
        colocate_mode=True,
    )
    return trainer


def test_resident_configuration_accepts_supported_path(trainer):
    """One actor and a forked colocated rollout need no memory saver."""
    trainer._validate_npu_share()


def test_single_npu_example_configuration_is_valid(trainer):
    """The checked-in recipe parses into the actual public config schema."""
    path = Path(__file__).parents[1] / "examples/math/gsm8k_grpo_single_npu.yaml"
    config, _ = load_expr_config(["--config", str(path)], GRPOConfig)
    trainer.config = config
    trainer._validate_npu_share()
    assert config.cluster.n_gpus_per_node == 1
    assert config.total_train_steps == 2


@pytest.mark.parametrize(
    "path,value",
    [
        ("enable_offload", True),
        ("actor.weight_update_mode", "xccl"),
        ("rollout.max_head_offpolicyness", 2),
        ("rollout.scheduling_strategy.fork", False),
        ("vllm.enforce_eager", False),
        ("recover.mode", "auto"),
    ],
)
def test_resident_configuration_rejects_unsupported_path(trainer, path, value):
    """Unsafe combinations fail before any worker launches."""
    obj = trainer.config
    fields = path.split(".")
    for field in fields[:-1]:
        obj = getattr(obj, field)
    setattr(obj, fields[-1], value)
    with pytest.raises(ValueError, match="npu_share requires"):
        trainer._validate_npu_share()


def test_resident_pause_stops_generation_without_releasing_memory(trainer):
    """Admission and generation must stop before training begins."""
    trainer._offload_rollout()
    trainer._memory_policy.train.assert_called_once_with()
    assert [c[0] for c in trainer.rollout.mock_calls] == ["pause", "pause_generation"]


def test_resident_publish_loads_then_commits_versions_then_allows_generation(trainer):
    """A new generation can never precede successful weight publication."""
    calls = Mock()
    calls.attach_mock(trainer.actor, "actor")
    calls.attach_mock(trainer.rollout, "rollout")
    calls.attach_mock(trainer.eval_rollout, "eval")
    calls.attach_mock(trainer._memory_policy, "memory")
    trainer._publish_weights(3)
    assert [c[0] for c in calls.mock_calls] == [
        "actor.update_weights",
        "actor.export_stats",
        "memory.inference",
        "actor.update_weights",
        "actor.set_version",
        "rollout.set_version",
        "eval.set_version",
        "rollout.continue_generation",
    ]
    metas = [c.args[0] for c in trainer.actor.update_weights.call_args_list]
    assert [m.colocate_stage for m in metas] == [0, 1]
    trainer.actor.set_version.assert_called_once_with(3)
    trainer.rollout.resume.assert_not_called()
    trainer.actor.offload.assert_not_called()
    trainer.actor.onload.assert_not_called()


@pytest.mark.parametrize("stage", [0, 1])
def test_failed_publish_keeps_generation_paused(trainer, stage):
    """A failed save/synchronization/load must not release inference."""

    def update(meta):
        if meta.colocate_stage == stage:
            raise RuntimeError("weight handoff failed")

    trainer.actor.update_weights.side_effect = update
    with pytest.raises(RuntimeError, match="handoff failed"):
        trainer._publish_weights(1)
    trainer.rollout.continue_generation.assert_not_called()
    trainer.rollout.resume.assert_not_called()
    trainer.rollout.set_version.assert_not_called()


def test_failed_memory_wake_keeps_dispatcher_paused(trainer):
    """Existing offload mode also fails closed instead of resuming on error."""
    trainer.config.npu_share = False
    trainer.rollout.onload.side_effect = RuntimeError("wake failed")
    with pytest.raises(RuntimeError, match="wake failed"):
        trainer._onload_rollout()
    trainer.rollout.resume.assert_not_called()
    trainer.rollout.continue_generation.assert_not_called()
