# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.test_colocation_memory import World

from areal.api import WeightUpdateMeta, Worker
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import InferenceEngineConfig
from areal.infra.borrowed_rollout import BorrowedRolloutPool
from areal.infra.controller.rollout_controller import RolloutController
from areal.infra.request_router import RequestLeasePool
from areal.trainer.colocation_memory import GroupColocationMemoryPolicy, single_status


@pytest.mark.parametrize("replicas", [1, 2, 4])
def test_withdraw_routes_only_to_dedicated_replicas(replicas):
    pool = BorrowedRolloutPool(16, replicas_per_group=replicas)
    assert {pool.choose(0)[0] for _ in range(2 * replicas)} == set(range(2 * replicas))
    pool.withdraw()
    assert {pool.choose(0)[0] for _ in range(2 * replicas)} == set(
        range(replicas, 2 * replicas)
    )


def test_memory_pressure_reclaims_only_affected_device():
    worlds = [World(10), World(30)]
    policy = GroupColocationMemoryPolicy(
        worlds[0].policy().config,
        lambda action, device_rank, **kw: worlds[device_rank].actor(action),
        lambda action, device_rank, **kw: worlds[device_rank].rollout(action),
        Mock(),
        2,
    )
    policy.train()
    assert ("rollout", "release_kv") in worlds[0].calls
    assert ("rollout", "release_kv") not in worlds[1].calls
    assert worlds[1].kv


def test_memory_summary_never_sums_free_capacity():
    status = single_status(
        [
            dict(free_bytes=2, total_bytes=64, restore_bytes=5),
            dict(free_bytes=30, total_bytes=64, restore_bytes=0),
        ]
    )
    assert status["free_bytes"] == 2
    assert status["restore_bytes"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "devices,tp,training",
    [
        (2, 1, 1),
        (4, 1, 2),
        (4, 2, 2),
        (8, 1, 4),
        (8, 2, 4),
        (8, 4, 4),
        (3, 1, 1),
        (4, 1, 1),
        (6, 2, 2),
        (8, 2, 2),
    ],
)
async def test_publication_joins_every_dedicated_instance_once(devices, tp, training):
    replicas = devices // tp
    config = InferenceEngineConfig(backend=f"vllm:d{replicas}p1t{tp}")
    scheduler = SimpleNamespace(async_call_engine=AsyncMock())
    ctrl = RolloutController(Mock, config, scheduler)
    ctrl._worker_role = "rollout"
    ctrl.enable_npu_share_pool(borrowed_replicas=training // tp)
    ctrl.workers = [Worker(id=f"rollout/{i}", ip="localhost") for i in range(replicas)]
    meta = WeightUpdateMeta.from_fsdp_xccl(
        gen_allocation=ModelAllocation.from_str(
            f"vllm:d{(devices - training) // tp}p1t{tp}"
        )
    )
    meta.npu_share_fixed = True
    await ctrl.init_weights_update_group(meta)
    calls = scheduler.async_call_engine.call_args_list
    assert [c.kwargs["worker_id"] for c in calls] == [
        f"rollout/{i}" for i in ctrl.fixed_ranks
    ]
    assert [c.kwargs["xccl_group_ranks"] for c in calls] == [
        [i] for i in range((devices - training) // tp)
    ]
    assert meta.gen_allocation.world_size == devices - training


def test_rebalance_reserves_actual_target_without_duplicate_execution():
    pool = RequestLeasePool(["a", "b", "c", "d"], 3, task_affinity=False)
    for rank in range(3):
        pool.close(rank)
    lease = pool.acquire("ticket", "request")
    assert lease["rank"] == 3
    for rank in range(3):
        pool.open(rank, 0)
    selected = pool.reserve_rebalance({"request": lease["lease_id"]}, 0, 3, 1)
    assert selected == {"request": lease["lease_id"]}
    assert pool.active["ticket"]["rank"] == 3
    assert pool.rebalance_reservations["request"]["rank"] == 1
    assert pool.reserve_rebalance({"request": lease["lease_id"]}, 0, 3, 0) == {}


@pytest.mark.parametrize("peer_error,growth", [(None, False), ("out of memory", True)])
def test_tp_mapping_requires_all_members_and_allows_no_growth(
    monkeypatch, peer_error, growth
):
    import torch.distributed as dist

    from areal.engine.vllm_ext.dynamic_kv import coordinated_reconcile

    group = SimpleNamespace(world_size=2, cpu_group=object())
    monkeypatch.setitem(
        sys.modules, "vllm.distributed", SimpleNamespace(get_tp_group=lambda: group)
    )

    def gather(output, value, group):
        output[:] = [
            value,
            (peer_error, value[1]) if isinstance(value, tuple) else peer_error,
        ]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    pool = SimpleNamespace(
        broken=False,
        swapped={},
        capacity_blocks=8,
        blocks_per_page=1,
        num_blocks=8,
        bytes_per_block=1,
        page=1,
        tensors=[object()],
        mapped={0},
        keep_idle_pages=0,
        reserve_bytes=100,
        backend=SimpleNamespace(free_bytes=lambda: 0),
        reconcile=Mock(),
    )
    if peer_error:
        with pytest.raises(RuntimeError, match="preflight"):
            coordinated_reconcile(pool, {0, 1})
        pool.reconcile.assert_not_called()
        assert pool.broken
    else:
        coordinated_reconcile(pool, {0})
        pool.reconcile.assert_called_once_with({0})


@pytest.mark.parametrize("replicas", [4, 8])
def test_global_drain_accounts_for_answers_across_all_instances(replicas):
    import copy

    from tests.test_rolling_drain import frame, requests

    from areal.infra.rolling_drain import RollingBatchCalibration

    calibration = RollingBatchCalibration(samples=16, replicas=replicas)
    engines, route = frame(lengths=[("calibration", 10)])
    engines = [copy.deepcopy(engines[0]) for _ in range(replicas)]
    calibration.freeze(engines, route, [dict(tail=1.0, restore=1.0)])
    route, _ = requests(count=12, groups=2)
    for index, request in enumerate(route["active"]):
        request["rank"] = index % replicas
    result = calibration.estimate(route, [dict(active=[]) for _ in range(replicas)], 2)
    assert result["known_questions"] == 2
    assert result["migrating"] == sum(
        r["rank"] < replicas // 2 for r in route["active"]
    )
    assert result["stay_data_ready"] > 0
    assert result["move_data_ready"] > 0


def test_grouped_fork_partitions_parent_devices_without_reserving_twice(monkeypatch):
    from areal.api import Job
    from areal.api.cli_args import SchedulingSpec, SchedulingStrategy
    from areal.infra.scheduler import local

    scheduler = object.__new__(local.LocalScheduler)
    parents = [
        local.WorkerInfo(
            worker=Worker(id=f"actor/{i}", ip="localhost"),
            process=None,
            role="actor",
            gpu_devices=[i],
            created_at=0,
            log_file="unused",
        )
        for i in range(4)
    ]
    scheduler._workers = {"actor": parents}
    scheduler._colocated_roles = {}
    specs = [SchedulingSpec(gpu=2), SchedulingSpec(gpu=2)]
    scheduler._prepare_worker_specs = Mock(return_value=specs)
    call = Mock(return_value=["rollout/0", "rollout/1"])
    monkeypatch.setattr(local, "run_async_task", call)
    job = Job(
        role="rollout",
        replicas=2,
        tasks=specs,
        scheduling_strategy=SchedulingStrategy(
            type="colocation", target="actor", fork=True
        ),
    )
    scheduler.fork_workers(job)
    grouped = call.call_args.args[3]
    assert [w.gpu_devices for w in grouped] == [[0, 1], [2, 3]]
    assert [w.worker.id for w in grouped] == ["actor/0", "actor/2"]
    assert [w.gpu_devices for w in parents] == [[0], [1], [2], [3]]


@pytest.mark.asyncio
async def test_grouped_fork_cleanup_uses_actual_parent():
    from areal.infra.scheduler import local

    scheduler = object.__new__(local.LocalScheduler)
    parents = [
        local.WorkerInfo(
            worker=Worker(id=f"actor/{i}", ip="localhost"),
            process=None,
            role="actor",
            gpu_devices=[i],
            created_at=0,
            log_file="unused",
        )
        for i in range(4)
    ]
    scheduler._workers = {"actor": parents}
    scheduler._kill_forked_worker = AsyncMock()
    child = local.WorkerInfo(
        worker=Worker(id="rollout/1", ip="localhost"),
        process=None,
        role="rollout",
        gpu_devices=[2, 3],
        created_at=0,
        log_file="unused",
        fork_parent_index=2,
    )
    await scheduler._cleanup_forked_workers_async("rollout", "actor", [child])
    assert scheduler._kill_forked_worker.call_args.args[3] is parents[2]


def test_retained_lease_reopens_only_same_weight_version():
    pool = RequestLeasePool(["one"], 2, task_affinity=False)
    pool.acquire("ticket", "request")
    pool.close(0)
    pool.reopen(0, 0)
    assert pool.active["ticket"]["version"] == 0
    with pytest.raises(RuntimeError, match="another version"):
        pool.reopen(0, 1)


def test_scheduler_queue_trace_installs_without_device_hooks(monkeypatch):
    from areal.utils import request_trace

    class Scheduler:
        def add_request(self, request):
            return request.request_id

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.core.sched.scheduler",
        SimpleNamespace(Scheduler=Scheduler),
    )
    monkeypatch.setattr(request_trace, "enabled", lambda: True)
    emit = Mock()
    monkeypatch.setattr(request_trace, "emit", emit)
    request_trace.install_scheduler_trace()
    request_trace.install_scheduler_trace()
    assert Scheduler().add_request(SimpleNamespace(request_id="request")) == "request"
    emit.assert_called_once_with("native_queue", internal_id="request")


def test_create_workers_dispatches_grouped_fork_before_replica_equality_check():
    from areal.infra.scheduler.local import LocalScheduler

    scheduler = object.__new__(LocalScheduler)
    scheduler._workers = {"actor": [object() for _ in range(4)]}
    scheduler._prepare_worker_specs = Mock(return_value=[])
    scheduler.fork_workers = Mock(return_value=["rollout/0", "rollout/1"])
    job = SimpleNamespace(
        role="rollout",
        replicas=2,
        tasks=[],
        scheduling_strategy=SimpleNamespace(
            type="colocation", target="actor", fork=True
        ),
    )
    assert scheduler.create_workers(job) == ["rollout/0", "rollout/1"]
    scheduler.fork_workers.assert_called_once_with(job)


def test_shared_vllm_avoids_training_hccl_port_and_preserves_explicit_config():
    from areal.engine.vllm_remote import VLLMBackend

    env = {"AREAL_AUTO_MEMORY": "1", "ASCEND_RT_VISIBLE_DEVICES": "0,1"}
    assert VLLMBackend.build_server_env(env)["HCCL_NPU_SOCKET_PORT_RANGE"] == "auto"
    assert "HCCL_NPU_SOCKET_PORT_RANGE" not in env
    env["HCCL_NPU_SOCKET_PORT_RANGE"] = "61000-61050"
    assert (
        VLLMBackend.build_server_env(env)["HCCL_NPU_SOCKET_PORT_RANGE"] == "61000-61050"
    )
    assert "HCCL_NPU_SOCKET_PORT_RANGE" not in VLLMBackend.build_server_env({})
