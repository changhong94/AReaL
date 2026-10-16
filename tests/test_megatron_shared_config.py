# SPDX-License-Identifier: Apache-2.0
"""Validate shared Megatron configuration before launching any NPU workers."""

from pathlib import Path

import pytest

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.trainer.rl_trainer import PPOTrainer


@pytest.fixture
def trainer():
    obj = object.__new__(PPOTrainer)
    recipe = Path(__file__).parents[1] / "examples/math/gsm8k_grpo_single_npu.yaml"
    obj.config, _ = load_expr_config(["--config", str(recipe)], GRPOConfig)
    obj.actor_alloc = ModelAllocation.from_str("megatron:d1p1t1")
    obj.rollout_alloc = ModelAllocation.from_str("vllm:d1p1t1")
    for role in ("actor", "rollout", "critic", "ref", "teacher"):
        setattr(obj, f"_should_offload_{role}", False)
    config = obj.config.actor.megatron
    config.bridge_type = "mbridge"
    config.fp8_config = None
    config.use_precision_aware_optimizer = False
    config.async_save = False
    config.use_torch_fsdp2 = False
    config.use_custom_fsdp = False
    config.wrap_with_ddp = True
    return obj


def test_supported_megatron_skips_fsdp_specific_loading_requirement(trainer):
    trainer.config.actor.fsdp.memory_efficient_load = False
    trainer._validate_cfg()


@pytest.mark.parametrize("backend", ["megatron", "fsdp"])
def test_missing_optimizer_is_configuration_error(trainer, backend):
    trainer.actor_alloc = ModelAllocation.from_str(f"{backend}:d1p1t1")
    trainer.config.actor.optimizer = None
    with pytest.raises(ValueError, match="ordinary AdamW"):
        trainer._validate_cfg()


@pytest.mark.parametrize(
    "field,value",
    [
        ("bridge_type", "megatron-bridge"),
        ("fp8_config", object()),
        ("use_precision_aware_optimizer", True),
        ("async_save", True),
        ("use_torch_fsdp2", True),
        ("use_custom_fsdp", True),
        ("wrap_with_ddp", False),
    ],
)
def test_unsupported_megatron_features_rejected_before_launch(trainer, field, value):
    setattr(trainer.config.actor.megatron, field, value)
    with pytest.raises(ValueError, match="Shared Megatron requires"):
        trainer._validate_cfg()


def test_merged_lora_rejected_before_launch(trainer):
    trainer.config.actor.use_merged_lora = True
    with pytest.raises(ValueError, match="Shared Megatron requires"):
        trainer._validate_cfg()


def test_fixed_megatron_does_not_inherit_shared_feature_restrictions(trainer):
    trainer.config.npu_share = False
    trainer.config.actor.megatron.bridge_type = "megatron-bridge"
    trainer._validate_cfg()


@pytest.mark.parametrize("backend", ["fsdp", "megatron"])
@pytest.mark.parametrize("devices,tp", [(2, 1), (4, 1), (4, 2), (8, 1), (8, 2), (8, 4)])
def test_shared_multi_npu_topologies_validate(trainer, backend, devices, tp):
    trainer.config.rollout.max_head_offpolicyness = 2
    half = devices // 2
    trainer.actor_alloc = ModelAllocation.from_str(f"{backend}:d{half}p1t1")
    trainer.rollout_alloc = ModelAllocation.from_str(f"vllm:d{devices // tp}p1t{tp}")
    trainer.config.vllm.distributed_executor_backend = "uni" if tp == 1 else "mp"
    trainer._validate_cfg()
