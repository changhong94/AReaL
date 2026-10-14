# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areal.engine.vllm_ext import ascend_kernel_warmup as warmup


@pytest.mark.parametrize(
    "capacity,expected", [(1, (1,)), (8, (1, 2)), (4096, (1, 2, 16))]
)
def test_sizes_cover_singleton_and_alignment_within_budget(capacity, expected):
    assert warmup.request_warmup_sizes(capacity) == expected


@pytest.fixture
def runner(monkeypatch):
    config = ModuleType("vllm.config")
    config.CUDAGraphMode = SimpleNamespace(NONE="none")
    logger = ModuleType("vllm.logger")
    logger.init_logger = Mock(return_value=Mock())
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=Mock()), raising=False
    )
    bt = SimpleNamespace(
        is_mamba_group=False,
        block_table=SimpleNamespace(gpu=torch.full((2, 4), 7, dtype=torch.int32)),
        slot_mapping=SimpleNamespace(gpu=torch.full((32,), 9, dtype=torch.int64)),
    )
    table = SimpleNamespace(block_tables=[bt], compute_slot_mapping=Mock())
    return SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=0, block_table=table),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        device="cpu",
        _dummy_run=Mock(),
    )


@pytest.mark.parametrize("fail", [False, True])
def test_slot_buffers_restored_and_failures_propagated(runner, fail):
    table = runner.input_batch.block_table
    bt = table.block_tables[0]

    def mapping(num_reqs, starts, positions):
        assert num_reqs == 1
        assert starts.tolist() == [0, len(positions)]
        assert torch.count_nonzero(bt.block_table.gpu[0]) == 0
        bt.slot_mapping.gpu.fill_(3)
        if fail:
            raise RuntimeError("compile failed")

    table.compute_slot_mapping.side_effect = mapping
    if fail:
        with pytest.raises(RuntimeError, match="compile failed"):
            warmup.warmup_request_kernels(SimpleNamespace(model_runner=runner))
    else:
        warmup.warmup_request_kernels(SimpleNamespace(model_runner=runner))
        assert table.compute_slot_mapping.call_count == 3
    assert [c.args[0] for c in runner._dummy_run.call_args_list] == [1, 2, 16]
    torch.testing.assert_close(
        bt.block_table.gpu, torch.full((2, 4), 7, dtype=torch.int32)
    )
    torch.testing.assert_close(
        bt.slot_mapping.gpu, torch.full((32,), 9, dtype=torch.int64)
    )


def test_live_requests_reject_warmup(runner):
    runner.input_batch.num_reqs = 1
    with pytest.raises(RuntimeError, match="before admitting"):
        warmup.warmup_request_kernels(SimpleNamespace(model_runner=runner))
    runner._dummy_run.assert_not_called()


@pytest.mark.parametrize("device,enabled", [("cuda", "1"), ("npu", "0"), ("npu", "1")])
def test_install_is_gated_idempotent_and_preserves_vendor_order(
    monkeypatch, device, enabled
):
    platform = ModuleType("areal.infra.platforms")
    platform.current_platform = SimpleNamespace(device_type=device)
    monkeypatch.setitem(sys.modules, "areal.infra.platforms", platform)
    parent = ModuleType("vllm_ascend.model_executor.warmup")
    order = []
    original = Mock(side_effect=lambda w: order.append("vendor"))
    original._areal_request_warmup = False
    module = ModuleType("vllm_ascend.model_executor.warmup.kernel_warmup")
    module.kernel_warmup = original
    parent.kernel_warmup = original  # Vendor package re-exports the function.
    monkeypatch.setitem(sys.modules, "vllm_ascend.model_executor.warmup", parent)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("AREAL_ASCEND_KERNEL_WARMUP", enabled)
    monkeypatch.setattr(
        warmup, "warmup_request_kernels", lambda w: order.append("request")
    )
    warmup.install_ascend_kernel_warmup()
    warmup.install_ascend_kernel_warmup()
    module.kernel_warmup(object())
    expected = (
        ["vendor", "request"] if device == "npu" and enabled == "1" else ["vendor"]
    )
    assert order == expected
