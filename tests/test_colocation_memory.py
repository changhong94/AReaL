# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.engine.colocation_memory import TaggedMemory
from areal.engine.vllm_ext.kv_slab_adapter import transition_slab_cache
from areal.infra.controller.rollout_controller import RolloutController
from areal.trainer.colocation_memory import GIB, ColocationMemoryPolicy, single_status
from areal.trainer.rl_trainer import PPOTrainer


@pytest.mark.parametrize("suffix", ["", "_kv_slab", "_auto_memory"])
def test_adaptive_recipe_parses_into_public_configuration(suffix):
    path = (
        Path(__file__).parents[1] / f"examples/math/gsm8k_grpo_single_npu{suffix}.yaml"
    )
    config, _ = load_expr_config(["--config", str(path)], GRPOConfig)
    assert config.npu_share
    assert config.actor.fsdp.memory_efficient_load
    assert config.vllm.distributed_executor_backend == "uni"


def test_npu_share_defaults_off_and_cli_can_disable_recipe():
    assert GRPOConfig().npu_share is False
    path = Path(__file__).parents[1] / "examples/math/gsm8k_grpo_single_npu.yaml"
    config, _ = load_expr_config(["--config", str(path), "npu_share=false"], GRPOConfig)
    assert config.npu_share is False
    assert not hasattr(config, "resident_colocation")
    assert not hasattr(config.colocation_memory, "enabled")


def test_memory_rpc_reaches_scheduler_with_unambiguous_arguments():
    async def dispatch(worker_id, method, engine_name, action):
        assert worker_id == "worker" and method == "colocation_memory"
        assert engine_name == "engine-0" and action == "status"
        return {"free_bytes": 123}

    controller = object.__new__(RolloutController)
    controller._borrowed_pool = None
    controller.workers = [SimpleNamespace(id="worker")]
    controller.scheduler = SimpleNamespace(async_call_engine=dispatch)
    controller._engine_name = lambda rank: f"engine-{rank}"
    assert controller.colocation_memory("status") == [{"free_bytes": 123}]


def test_adaptive_configuration_rejects_unmanaged_loading_before_launch():
    trainer = object.__new__(PPOTrainer)
    trainer.config = GRPOConfig()
    trainer.config.npu_share = True
    trainer.config.vllm.distributed_executor_backend = "uni"
    trainer.config.actor.fsdp.memory_efficient_load = False
    trainer.actor_alloc = SimpleNamespace(backend="fsdp")
    trainer.rollout_alloc = SimpleNamespace(
        backend="vllm", parallel=SimpleNamespace(tp_size=1)
    )
    with pytest.raises(ValueError, match="memory_efficient_load"):
        trainer._validate_cfg()


class Backend:
    def __init__(self):
        self.data = {
            1: SimpleNamespace(
                tag="weights", handle=(0, 4, 1, 0), cpu_backup_tensor=None
            ),
            2: SimpleNamespace(tag="kv", handle=(0, 4, 2, 0), cpu_backup_tensor=None),
        }
        self.contents = {1: bytearray(b"abcd"), 2: bytearray(b"1234")}
        self.calls = []

    def inventory(self):
        return self.data

    def free_memory(self):
        return 100, 200

    def synchronize(self):
        pass

    def backup(self, a):
        return self.contents[a.handle[2]][:]

    def unmap(self, a):
        self.calls.append(("unmap", a.tag))
        del self.contents[a.handle[2]]

    def map(self, a):
        self.calls.append(("map", a.tag))
        self.contents[a.handle[2]] = bytearray(4)

    def restore(self, a):
        self.contents[a.handle[2]][:] = a.cpu_backup_tensor


def test_selective_kv_reclaim_never_unmaps_weights():
    backend = Backend()
    memory = TaggedMemory(backend, ["weights", "kv"])
    memory.transition(["kv"], mapped=False, preserve=False)
    assert backend.contents[1] == b"abcd"
    assert backend.data[2].cpu_backup_tensor is None
    assert backend.calls == [("unmap", "kv")]
    memory.transition(["kv"], mapped=False, preserve=False)
    assert len(backend.calls) == 1


def test_cpu_backup_roundtrip_preserves_contents_and_handles():
    backend = Backend()
    memory = TaggedMemory(backend, ["weights", "kv"])
    original = backend.data[1]
    for _ in range(3):
        memory.transition(["weights"], mapped=False)
        assert memory.snapshot()["cpu_backup_bytes"] == 4
        memory.transition(["weights"], mapped=True)
        assert backend.contents[1] == b"abcd"
        assert backend.data[1] is original
        assert memory.snapshot()["cpu_backup_bytes"] == 0


@pytest.mark.parametrize("operation", ["backup", "unmap", "map", "restore"])
def test_partial_failure_is_terminal(operation):
    backend = Backend()
    memory = TaggedMemory(backend, ["weights", "kv"])
    if operation in {"map", "restore"}:
        memory.transition(["weights"], mapped=False)
    setattr(backend, operation, Mock(side_effect=RuntimeError("failed")))
    with pytest.raises(RuntimeError, match="failed"):
        memory.transition(["weights"], mapped=operation in {"map", "restore"})
    with pytest.raises(RuntimeError, match="previously failed"):
        memory.transition(["weights"], mapped=True)


class World:
    def __init__(self, free):
        self.free = free * GIB
        self.cache = 0
        self.kv = True
        self.weights = True
        self.actor_mapped = True
        self.calls = []

    def status(self, restore=0):
        return {
            "free_bytes": self.free,
            "total_bytes": 64 * GIB,
            "restore_bytes": restore,
        }

    def actor(self, action):
        self.calls.append(("actor", action))
        if action == "release_cache":
            self.free += self.cache
            self.cache = 0
        if action == "offload" and self.actor_mapped:
            self.free += 16 * GIB
            self.actor_mapped = False
        if action == "restore" and not self.actor_mapped:
            self.free -= 16 * GIB
            self.actor_mapped = True
        return {
            **self.status(0 if self.actor_mapped else 16 * GIB),
            "future_optimizer_bytes": 4 * GIB,
            "gradient_bytes": 4 * GIB,
        }

    def rollout(self, action):
        self.calls.append(("rollout", action))
        if action in {"release_kv", "offload"} and self.kv:
            self.free += 8 * GIB
            self.kv = False
        if action == "offload" and self.weights:
            self.free += 4 * GIB
            self.weights = False
        missing = (0 if self.kv else 8 * GIB) + (0 if self.weights else 4 * GIB)
        if action == "restore":
            self.free -= missing
            self.kv = self.weights = True
            missing = 0
        return [[[self.status(missing)]]]

    def policy(self):
        config = SimpleNamespace(
            train_workspace_gib=4,
            inference_workspace_gib=2,
            safety_margin_gib=2,
            device_budget_gib=None,
        )
        return ColocationMemoryPolicy(config, self.actor, self.rollout, Mock())


@pytest.mark.parametrize(
    "free,expected", [(20, []), (10, ["release_kv"]), (4, ["release_kv", "offload"])]
)
def test_budget_selects_least_expensive_safe_train_path(free, expected):
    world = World(free)
    world.policy().train()
    assert [
        action
        for role, action in world.calls
        if role == "rollout" and action != "status"
    ] == expected
    assert world.calls[-1] == ("actor", "restore")


def test_insufficient_budget_never_restores_actor():
    world = World(0)
    with pytest.raises(RuntimeError, match="Insufficient memory"):
        world.policy().train()
    assert ("actor", "restore") not in world.calls


def test_inference_offloads_actor_before_restoring_kv_and_weights():
    world = World(3)
    world.kv = world.weights = False
    world.policy().inference()
    assert world.calls.index(("actor", "offload")) < world.calls.index(
        ("rollout", "restore")
    )


def test_initial_load_uses_host_memory_when_resident_load_does_not_fit():
    world = World(8)
    world.policy().initial_inference(0.3)
    assert not world.actor_mapped


def test_budget_cap_and_single_device_result_validation():
    world = World(20)
    policy = world.policy()
    policy.config.device_budget_gib = 48
    assert policy.free(world.status()) == 4 * GIB
    assert (
        single_status([world.status(), world.status()])["free_bytes"]
        == world.status()["free_bytes"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result", [[], [{"states": {"weights": "cpu", "kv_cache": "mapped"}}]]
)
async def test_restore_requires_positive_worker_ack(monkeypatch, result):
    monkeypatch.setenv("AREAL_AUTO_MEMORY", "1")
    monkeypatch.setenv("AREAL_KV_SLAB_BLOCKS", "16")
    engine = AsyncMock()
    engine.collective_rpc.return_value = result
    with pytest.raises(RuntimeError, match="acknowledge"):
        await transition_slab_cache(engine, mapped=True)


@pytest.mark.parametrize("phase", ["train", "inference"])
def test_sufficient_budget_retains_actor_cache(phase):
    world = World(20)
    world.cache = 8 * GIB
    getattr(world.policy(), phase)()
    assert ("actor", "release_cache") not in world.calls
    assert world.cache == 8 * GIB


@pytest.mark.parametrize("phase", ["train", "inference"])
def test_cache_reclaim_avoids_weight_offload(phase):
    world = World(3)
    world.cache = 8 * GIB
    getattr(world.policy(), phase)()
    assert ("actor", "release_cache") in world.calls
    assert not any(action == "offload" for _, action in world.calls)
    assert world.cache == 0


def test_failed_cache_reclaim_never_resumes_inference():
    world = World(0)
    original = world.actor

    def actor(action):
        if action == "release_cache":
            raise RuntimeError("cache reclaim failed")
        return original(action)

    policy = world.policy()
    policy.actor = actor
    with pytest.raises(RuntimeError, match="cache reclaim failed"):
        policy.inference()
    assert ("rollout", "restore") not in world.calls


def test_cap_still_requires_cache_reclaim():
    world = World(20)
    world.kv = False
    world.cache = 10 * GIB
    policy = world.policy()
    policy.config.device_budget_gib = 48
    policy.inference()
    assert ("actor", "release_cache") in world.calls
    assert world.actor_mapped
