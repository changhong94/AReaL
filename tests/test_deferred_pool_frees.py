"""CPU tests for the private-pool deferred-free exit boundary."""

import importlib.util
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location(
    "pool_backend",
    Path(__file__).resolve().parents[1] / "areal/engine/colocation_memory.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DeferredFreeTests(unittest.TestCase):
    def backend(self, reuse=False):
        backend = object.__new__(module.AscendPoolBackend)
        backend.workspace = None
        backend.pools = {"actor_workspace": ("pool", None)} if reuse else {}
        active = []

        @contextmanager
        def scope(*args, **kwargs):
            active.append(True)
            try:
                yield
            finally:
                active.pop()

        backend.allocator = SimpleNamespace(
            use_memory_pool=scope,
            use_allocation_tag=scope,
            allocator_and_pools={
                "actor_workspace": ("pool", None),
                "actor_weights": ("weights", None),
            },
        )

        def allocate(*args, **kwargs):
            self.assertFalse(
                active, "Reclaim trigger must run outside allocation scopes"
            )
            return object()

        backend.torch = SimpleNamespace(
            empty=Mock(side_effect=allocate),
            uint8="uint8",
            npu=SimpleNamespace(
                memory=SimpleNamespace(
                    use_mem_pool=scope,
                    MemPoolContext=SimpleNamespace(active_pool=lambda: None),
                ),
                synchronize=Mock(),
                empty_cache=Mock(),
            ),
        )
        return backend

    def test_first_and_reused_workspace_process_events_outside_scope(self):
        for reuse in (False, True):
            backend = self.backend(reuse)
            with backend.allocation_scope("actor_workspace"):
                backend.torch.empty.assert_not_called()
            backend.torch.empty.assert_called_once_with(
                1024, dtype="uint8", device="npu"
            )
            backend.torch.npu.synchronize.assert_not_called()
            backend.torch.npu.empty_cache.assert_not_called()

    def test_failure_preserves_original_exception(self):
        backend = self.backend()
        with self.assertRaisesRegex(ValueError, "original"):
            with backend.allocation_scope("actor_workspace"):
                raise ValueError("original")
        backend.torch.empty.assert_not_called()

    def test_other_pool_does_not_trigger(self):
        backend = self.backend()
        with backend.allocation_scope("actor_weights"):
            pass
        backend.torch.empty.assert_not_called()

    def test_enclosing_pool_defers_trigger(self):
        backend = self.backend()
        backend.torch.npu.memory.MemPoolContext.active_pool = lambda: object()
        with backend.allocation_scope("actor_workspace"):
            pass
        backend.torch.empty.assert_not_called()


if __name__ == "__main__":
    unittest.main()
