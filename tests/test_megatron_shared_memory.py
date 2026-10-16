# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areal.engine.megatron_utils.shared_memory import (
    MegatronSharedMemory,
    future_optimizer_bytes,
    persistent_tensors,
)


def make_engine():
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    param.main_grad = torch.zeros(4, dtype=torch.float32)
    master = torch.nn.Parameter(param.float().detach().clone())
    inner = torch.optim.AdamW([master])
    wrapper = SimpleNamespace(optimizer=inner, fp32_from_float16_groups=[[master]])
    buffer = SimpleNamespace(param_data=param.detach(), grad_data=param.main_grad)
    model = SimpleNamespace(
        parameters=lambda: iter([param]),
        buffers=[buffer],
        expert_parallel_buffers=[],
        module=SimpleNamespace(buffers=lambda: []),
    )
    engine = SimpleNamespace(
        model=[model],
        optimizer=wrapper,
        is_offload=False,
        optimizer_zero_grad=Mock(),
        logger=Mock(),
    )
    backend = SimpleNamespace(
        workspace=None,
        assert_covered=Mock(),
        synchronize=Mock(),
        clear_cache=Mock(),
        free_memory=Mock(return_value=(0, 8192)),
    )
    memory = Mock()
    memory.snapshot.side_effect = lambda: {"restore_bytes": 0}
    memory.reclaim.return_value = {"restore_bytes": 64}
    return engine, backend, memory, param, master


def test_persistent_inventory_covers_flat_gradients_master_and_adam_state():
    """Storage coverage must include tensors absent from module.parameters()."""
    engine, _, _, param, master = make_engine()
    moment = torch.zeros_like(master)
    engine.optimizer.optimizer.state[master]["exp_avg"] = moment
    tensors = persistent_tensors(engine)
    ids = {id(t) for t in tensors}
    assert {id(param), id(param.main_grad), id(master), id(moment)} <= ids
    assert len(ids) == len(tensors)


def test_future_moments_reserve_fp32_and_skip_existing_state():
    """Two lazy FP32 moments need reservation; existing ones do not."""
    engine, _, _, _, master = make_engine()
    assert (
        future_optimizer_bytes(engine.optimizer) == int((32 + 8) * 1.05) + 4 * 1024**2
    )
    engine.optimizer.optimizer.state[master].update(
        exp_avg=torch.zeros_like(master), exp_avg_sq=torch.zeros_like(master), step=0
    )
    assert future_optimizer_bytes(engine.optimizer) == 0


def test_chained_optimizer_shared_parameter_is_reserved_once():
    """Chained wrappers must not double count aliased optimizer parameters."""
    engine, _, _, _, _ = make_engine()
    chain = SimpleNamespace(chained_optimizers=[engine.optimizer, engine.optimizer])
    assert future_optimizer_bytes(chain) == future_optimizer_bytes(engine.optimizer)


def test_detached_master_parameters_still_reserve_future_moments():
    """Megatron assigns gradients to detached masters without autograd."""
    engine, _, _, _, master = make_engine()
    expected = future_optimizer_bytes(engine.optimizer)
    master.requires_grad_(False)
    assert future_optimizer_bytes(engine.optimizer) == expected


def test_inventory_covers_optimizer_overflow_scalars_and_master_gradients():
    """Fused wrappers can retain scalar control tensors outside optimizer.state."""
    engine, _, _, _, master = make_engine()
    engine.optimizer.found_inf = torch.zeros(1)
    master.grad = torch.ones_like(master)
    ids = {id(t) for t in persistent_tensors(engine)}
    assert id(engine.optimizer.found_inf) in ids
    assert id(master.grad) in ids


def test_status_keeps_cached_memory_and_does_not_double_count_ddp_gradients():
    """A read-only query must not zero buffers, synchronize or evict caches."""
    engine, backend, memory, _, _ = make_engine()
    status = MegatronSharedMemory(engine, backend, memory).action()
    assert status["gradient_bytes"] == 0
    engine.optimizer_zero_grad.assert_not_called()
    backend.clear_cache.assert_not_called()
    backend.synchronize.assert_not_called()
    backend.assert_covered.assert_called_once()


def test_reclaim_preserves_ddp_buffers_and_never_writes_after_unmapping():
    """Repeated reclaim is legal without touching unmapped gradient views."""
    engine, backend, memory, param, _ = make_engine()
    helper = MegatronSharedMemory(engine, backend, memory)
    pointer = param.main_grad.data_ptr()
    helper.action("reclaim", 4096)
    assert engine.is_offload
    memory.reclaim.assert_called_with(
        ["actor_optimizer", "actor_weights"], 4096, preserve=True
    )
    helper.action("reclaim", 8192)
    engine.optimizer_zero_grad.assert_called_once()
    assert param.main_grad.data_ptr() == pointer


def test_offload_repeat_does_not_zero_unmapped_ddp_storage():
    """Backing up allocations must preserve view addresses and master values."""
    engine, backend, memory, _, master = make_engine()
    helper = MegatronSharedMemory(engine, backend, memory)
    helper.action("offload")
    memory.snapshot.side_effect = lambda: {"restore_bytes": 64}
    helper.action("offload")
    engine.optimizer_zero_grad.assert_called_once()
    memory.transition.assert_called_with(
        ["actor_weights", "actor_optimizer"], mapped=False, preserve=True
    )
    torch.testing.assert_close(master, torch.ones(4), rtol=0, atol=0)


def test_restore_precedes_coverage_checks():
    """Coverage inspection occurs after restoring allocations."""
    engine, backend, memory, _, _ = make_engine()
    engine.is_offload = True
    MegatronSharedMemory(engine, backend, memory).action("restore")
    assert not engine.is_offload
    memory.transition.assert_called_once_with(
        ["actor_weights", "actor_optimizer"], mapped=True
    )
    backend.assert_covered.assert_called_once()


def test_finish_train_refuses_offloaded_buffers():
    """Never call zero_grad_buffer against an unmapped allocation."""
    engine, backend, memory, _, _ = make_engine()
    engine.is_offload = True
    with pytest.raises(RuntimeError, match="offloaded"):
        MegatronSharedMemory(engine, backend, memory).action("finish_train")
    engine.optimizer_zero_grad.assert_not_called()


@pytest.mark.parametrize("target", [-1, None, True, 1.5])
def test_reclaim_rejects_invalid_target_before_memory_mutation(target):
    """Reject invalid budgets without changing allocation state."""
    engine, backend, memory, _, _ = make_engine()
    with pytest.raises(ValueError, match="nonnegative"):
        MegatronSharedMemory(engine, backend, memory).action("reclaim", target)
    engine.optimizer_zero_grad.assert_not_called()


def test_finish_train_synchronizes_without_freeing_flat_gradients():
    """Persistent gradient storage stays valid for the next DDP step."""
    engine, backend, memory, param, _ = make_engine()
    before = param.main_grad.untyped_storage().nbytes()
    MegatronSharedMemory(engine, backend, memory).action("finish_train")
    engine.optimizer_zero_grad.assert_called_once()
    backend.synchronize.assert_called_once()
    backend.clear_cache.assert_not_called()
    assert param.main_grad.untyped_storage().nbytes() == before


def test_workspace_uses_reclaimable_default_pool_and_preserves_managed_state():
    """Temporary workspace must not leak into a non-reclaimable private pool."""
    from contextlib import nullcontext

    engine, backend, memory, _, _ = make_engine()
    backend.allocation_scope = Mock(side_effect=lambda tag: nullcontext())
    helper = MegatronSharedMemory(engine, backend, memory)
    with helper.allocation_scope("actor_workspace"):
        pass
    backend.allocation_scope.assert_not_called()
    for tag in ("actor_weights", "actor_optimizer"):
        with helper.allocation_scope(tag):
            pass
    assert [call.args[0] for call in backend.allocation_scope.call_args_list] == [
        "actor_weights",
        "actor_optimizer",
    ]
    helper.action("status")
    backend.clear_cache.assert_not_called()
    helper.action("release_cache")
    backend.clear_cache.assert_called_once()


def test_workspace_refuses_missing_managed_state():
    engine, backend, memory, _, _ = make_engine()
    backend.residency = SimpleNamespace(broken=False, _missing={1})
    helper = MegatronSharedMemory(engine, backend, memory)
    with pytest.raises(RuntimeError, match="Restore pool"):
        helper.allocation_scope("actor_workspace")


def test_reclaim_releases_disposable_cache_only_under_pressure():
    engine, backend, memory, _, _ = make_engine()
    helper = MegatronSharedMemory(engine, backend, memory)
    backend.free_memory.return_value = (8192, 16384)
    helper.action("reclaim", 4096)
    backend.clear_cache.assert_not_called()
    engine.is_offload = False
    backend.free_memory.return_value = (0, 16384)
    helper.action("reclaim", 4096)
    backend.clear_cache.assert_called_once()
