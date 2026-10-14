# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.engine.colocation_memory import AscendPoolBackend, ParkedWorkspace


class Backend:
    def __init__(self):
        self.data = {
            100: SimpleNamespace(
                handle=(0, 10, 100, 1), tag="actor_workspace", cpu_backup_tensor=None
            ),
            200: SimpleNamespace(
                handle=(0, 10, 200, 2), tag="actor_weights", cpu_backup_tensor=None
            ),
        }
        self.records = [
            dict(
                address=100,
                total_size=10,
                allocated_size=0,
                active_size=0,
                blocks=[dict(state="inactive", size=10)],
            )
        ]
        self.unmap = Mock()
        self.map = Mock()
        self.synchronize = Mock()

    def inventory(self):
        return self.data

    def segments(self):
        return self.records


def test_only_inactive_workspace_is_parked_and_address_retained():
    backend = Backend()
    workspace = ParkedWorkspace(backend)
    workspace.park()
    assert workspace.restore_bytes == 10
    backend.unmap.assert_called_once_with(backend.data[100])
    workspace.park()
    backend.unmap.assert_called_once()
    workspace.restore()
    backend.map.assert_called_once_with(backend.data[100])
    assert workspace.restore_bytes == 0 and backend.data[100].handle[2] == 100
    assert backend.data[100].cpu_backup_tensor is None


@pytest.mark.parametrize("state", ["active_allocated", "active_awaiting_free"])
def test_live_or_pending_block_prevents_discard(state):
    backend = Backend()
    backend.records[0]["blocks"][0]["state"] = state
    workspace = ParkedWorkspace(backend)
    workspace.park()
    backend.unmap.assert_not_called()


@pytest.mark.parametrize("bad", ["missing", "size", "unknown_block"])
def test_unknown_segment_is_not_discarded(bad):
    backend = Backend()
    if bad == "missing":
        backend.records = []
    elif bad == "size":
        backend.records[0]["total_size"] = 20
    else:
        backend.records[0]["blocks"][0]["state"] = "unknown"
    workspace = ParkedWorkspace(backend)
    if bad != "unknown_block":
        with pytest.raises(RuntimeError, match="exact allocator evidence"):
            workspace.park()
    else:
        workspace.park()
    backend.unmap.assert_not_called()


@pytest.mark.parametrize("operation", ["map", "unmap"])
def test_partial_failure_is_terminal(operation):
    backend = Backend()
    workspace = ParkedWorkspace(backend)
    if operation == "map":
        workspace.park()
    getattr(backend, operation).side_effect = RuntimeError("failed")
    with pytest.raises(RuntimeError, match="failed"):
        getattr(workspace, "restore" if operation == "map" else "park")()
    with pytest.raises(RuntimeError, match="previously failed"):
        workspace.restore()


def test_cache_eviction_is_forbidden_while_parked():
    backend = object.__new__(AscendPoolBackend)
    backend.workspace = ParkedWorkspace(Backend())
    backend.workspace.park()
    backend.torch = Mock()
    with pytest.raises(RuntimeError, match="Cannot evict"):
        backend.clear_cache()
    backend.torch.npu.empty_cache.assert_not_called()


def test_parked_actor_rejects_stats_and_checkpoint_context():
    from areal.engine.fsdp_engine import FSDPEngine

    engine = object.__new__(FSDPEngine)
    workspace = ParkedWorkspace(Backend())
    workspace.park()
    engine._adaptive_backend = SimpleNamespace(workspace=workspace)
    engine.is_offload = False
    with pytest.raises(RuntimeError, match="Restore workspace"):
        with engine._offload_aware_context():
            pass
    with pytest.raises(RuntimeError, match="Restore workspace"):
        engine._ensure_ready()
    workspace.restore()
    with engine._offload_aware_context():
        pass


def test_backward_scope_uses_calling_thread_and_restores_autograd_mode():
    import threading
    from contextlib import contextmanager

    import torch

    from areal.engine.fsdp_engine import FSDPEngine

    caller = threading.get_ident()
    calls = []

    @contextmanager
    def scope(tag):
        assert tag == "actor_workspace"
        assert not torch.autograd.is_multithreading_enabled()
        yield

    weight = torch.tensor([[2.0, 3.0]], requires_grad=True)

    def model(**kwargs):
        logits = weight * weight
        logits.grad_fn.register_prehook(
            lambda grad: calls.append(threading.get_ident())
        )
        return SimpleNamespace(logits=logits)

    engine = object.__new__(FSDPEngine)
    engine._adaptive_backend = SimpleNamespace(
        workspace=object(), allocation_scope=scope
    )
    engine._prepare_mb_inputs = lambda mb: ({}, SimpleNamespace(to_dict=lambda: {}))
    engine.enable_tree_training = False
    engine.model = model
    previous = torch.autograd.is_multithreading_enabled()
    engine.forward_backward_batch([None], lambda logits, ctx: logits.sum())
    assert calls == [caller]
    torch.testing.assert_close(weight.grad, torch.tensor([[4.0, 6.0]]))
    assert torch.autograd.is_multithreading_enabled() == previous
