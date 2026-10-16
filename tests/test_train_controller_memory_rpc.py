# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock

import pytest

from areal.api.alloc_mode import ModelAllocation
from areal.infra.controller.train_controller import TrainController


@pytest.fixture
def controller():
    value = object.__new__(TrainController)
    value.train_alloc = ModelAllocation.from_str("fsdp:d2p1t1")
    value._call_workers = AsyncMock(
        return_value=[{"free_bytes": 4096}, {"free_bytes": 2048}]
    )
    return value


@pytest.mark.parametrize("target", [0, 4096])
def test_reclaim_forwards_free_memory_target_to_every_worker(controller, target):
    result = controller.colocation_memory("reclaim", target_free_bytes=target)
    assert result["free_bytes"] == 2048
    assert len(result["devices"]) == 2
    controller._call_workers.assert_awaited_once_with(
        "colocation_memory",
        [["reclaim", "reclaim"]],
        {"target_free_bytes": [target, target]},
    )


@pytest.mark.parametrize("action", ["status", "restore", "offload", "finish_train"])
def test_memory_rpc_without_target_preserves_worker_arguments(controller, action):
    controller.colocation_memory(action)
    controller._call_workers.assert_awaited_once_with(
        "colocation_memory", [[action, action]], {}
    )
