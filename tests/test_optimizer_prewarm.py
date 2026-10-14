# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager, nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areal.engine import colocation_memory as memory
from areal.engine.fsdp_engine import FSDPEngine


class Backend:
    def __init__(self, free=100 * 1024**3):
        self.free = free
        self.scopes = []
        stream = SimpleNamespace(wait_event=Mock(), synchronize=Mock())
        self.torch = SimpleNamespace(
            npu=SimpleNamespace(
                Event=lambda: SimpleNamespace(record=Mock()),
                current_stream=lambda device: None,
                set_device=Mock(),
                Stream=lambda **kw: stream,
                stream=lambda s: nullcontext(),
            )
        )

    def free_memory(self):
        return self.free, 100 * 1024**3

    @contextmanager
    def allocation_scope(self, tag):
        self.scopes.append(tag)
        yield


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("amsgrad", [False, True])
def test_prewarm_matches_first_real_adam_update_without_modifying_weights(
    fused, amsgrad
):
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    left = torch.nn.Parameter(original.clone())
    right = torch.nn.Parameter(original.clone())
    a = torch.optim.AdamW([left], lr=0.01, fused=fused, amsgrad=amsgrad)
    b = torch.optim.AdamW([right], lr=0.01, fused=fused, amsgrad=amsgrad)
    b.state.update(memory.initialize_adam_states(b))
    torch.testing.assert_close(right, original, rtol=0, atol=0)
    assert b.state[right]["step"].item() == 0 and right.grad is None
    left.grad = torch.full_like(left, 0.2)
    right.grad = left.grad.clone()
    a.step()
    b.step()
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    for key, value in a.state[left].items():
        torch.testing.assert_close(value, b.state[right][key], rtol=0, atol=0)


def test_async_job_publishes_complete_state_only_when_joined(monkeypatch):
    p = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([p])
    entered, release = Event(), Event()
    original = memory.initialize_adam_states

    def delayed(opt):
        entered.set()
        assert release.wait(5)
        return original(opt)

    monkeypatch.setattr(memory, "initialize_adam_states", delayed)
    job = memory.OptimizerPrewarm(optimizer, Backend(), 0)
    try:
        assert entered.wait(2) and not optimizer.state
    finally:
        release.set()
        job.wait()
    job.wait()
    assert set(optimizer.state[p]) == {"step", "exp_avg", "exp_avg_sq"}


@pytest.mark.parametrize("cap", [None, 50 * 1024**3])
def test_insufficient_budget_creates_only_pool(monkeypatch, cap):
    p = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([p])
    backend = Backend(free=1024)
    init = Mock(side_effect=AssertionError("must not allocate"))
    monkeypatch.setattr(memory, "initialize_adam_states", init)
    job = memory.OptimizerPrewarm(optimizer, backend, 1024, cap)
    job.wait()
    assert backend.scopes == ["actor_optimizer"] and not optimizer.state
    init.assert_not_called()


def test_async_failure_propagates_after_thread_join(monkeypatch):
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(4))])
    monkeypatch.setattr(
        memory,
        "initialize_adam_states",
        Mock(side_effect=RuntimeError("failed allocation")),
    )
    job = memory.OptimizerPrewarm(optimizer, Backend(), 0)
    with pytest.raises(RuntimeError, match="failed allocation"):
        job.wait()
    assert not job._thread.is_alive() and not optimizer.state


def test_handoffs_preserve_cache_until_explicit_release():
    engine = object.__new__(FSDPEngine)
    engine.is_offload = False
    engine.model = torch.nn.Linear(2, 2)
    engine.optimizer = torch.optim.AdamW(engine.model.parameters())
    engine._adaptive_memory = Mock()
    engine._adaptive_memory.snapshot.return_value = dict(restore_bytes=0)
    engine._adaptive_backend = Mock(assert_covered=Mock(), workspace=None)
    engine.logger = Mock()
    engine.colocation_memory("status")
    engine._adaptive_backend.clear_cache.assert_not_called()
    engine.colocation_memory("restore")
    engine.optimizer_zero_grad = Mock()
    engine.colocation_memory("finish_train")
    engine.optimizer_zero_grad.assert_called_once()
    engine._adaptive_backend.synchronize.assert_called_once()
    engine._adaptive_backend.clear_cache.assert_not_called()
    engine.colocation_memory("release_cache")
    engine._adaptive_backend.clear_cache.assert_called_once()


def test_prewarm_error_prevents_memory_transition():
    engine = object.__new__(FSDPEngine)
    engine._adaptive_memory = Mock()
    engine._optimizer_prewarm = Mock()
    engine._optimizer_prewarm.wait.side_effect = RuntimeError("prewarm failed")
    with pytest.raises(RuntimeError, match="prewarm failed"):
        engine.colocation_memory("offload")
    engine._adaptive_memory.transition.assert_not_called()


def test_camem_callbacks_keep_identity_after_scope_exit_and_gc():
    import gc
    import weakref

    class Allocator:
        def python_malloc_callback(self, handle):
            return handle

        def python_free_callback(self, pointer):
            return pointer

    allocator = Allocator()
    memory.pin_camem_callbacks(allocator)
    malloc_ref = weakref.ref(allocator.python_malloc_callback)
    free_ref = weakref.ref(allocator.python_free_callback)
    memory.pin_camem_callbacks(allocator)
    gc.collect()
    assert allocator.python_malloc_callback is malloc_ref()
    assert allocator.python_free_callback is free_ref()
    assert malloc_ref()(123) == 123 and free_ref()(456) == 456
