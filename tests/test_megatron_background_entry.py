"""CPU orchestration failure tests; real NPU state is covered by short runs."""

import ast
import importlib.util
import json
import math
import os
import shutil
import tempfile
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

budget_spec = importlib.util.spec_from_file_location(
    "entry_budget", Path(__file__).parents[1] / "areal/trainer/idle_entry.py"
)
budget_module = importlib.util.module_from_spec(budget_spec)
budget_spec.loader.exec_module(budget_module)
Budget = budget_module.EntryBudget

source = Path(__file__).parents[1] / "areal/trainer/megatron_idle_training.py"
node = next(
    n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef)
)


class Parent:
    def arm(self, batch):
        pass

    def cancel_entry(self):
        pass


namespace = dict(
    IdleTrainingPool=Parent,
    time=time,
    os=os,
    shutil=shutil,
    json=json,
    math=math,
    request_trace=types.SimpleNamespace(emit=lambda *a, **k: None),
)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
Pool = namespace["MegatronIdleTrainingPool"]


class TestBackgroundEntry(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.pool = Pool.__new__(Pool)
        p = self.pool
        p._base, p._expanded = 1, 2
        p._retains = {1: True, 2: False}
        p._standby = {1: 2, 2: 1}
        p.path = Path(self.temp.name)
        p._cpu_jobs = ThreadPoolExecutor(max_workers=1)
        p._cpu_stage = None
        p._prepared = None
        p._entry_budget = Budget()
        p._cpu_capture = None
        p._maintenance_batch = None
        p._transfer_costs = []
        p._restore_costs = {1: [], 2: []}
        self.calls = []
        self.version = 0
        self.stage_error = False

        def save(action, path):
            self.calls.append("save")
            Path(path).mkdir()
            (Path(path) / "ready-0.json").write_text(
                json.dumps(dict(version=self.version, rank=0, world=1))
            )
            return {"version": self.version, "world": 1}

        def stage(action, path):
            self.calls.append("stage")
            if self.stage_error:
                raise RuntimeError("injected stage failure")
            return {
                "version": 0,
                "target_world": 2,
                "ranks": [{"version": 0}, {"version": 0}],
            }

        p.primary = types.SimpleNamespace(
            offload=lambda: self.calls.append("park"), idle_training_state=save
        )
        p.multi = types.SimpleNamespace(
            idle_training_state=stage, onload=lambda: self.calls.append("onload")
        )
        p.snapshot_actor = p.primary
        p.trainer = types.SimpleNamespace(
            rollout=types.SimpleNamespace(
                _request_pool=types.SimpleNamespace(active_requests=lambda rank: []),
                workers=[0, 1],
                wait_shared_publication=lambda: self.calls.append("publication"),
            ),
            _wait_step_tail=lambda: None,
        )

    def tearDown(self):
        self.pool._cpu_jobs.shutdown(wait=True)
        self.temp.cleanup()

    def test_staging_completes_before_restore(self):
        self.pool.arm(1)
        self.pool._join_cpu_stage()
        self.assertEqual(self.calls, ["publication", "park", "save", "stage"])
        with patch.dict(os.environ, {"AREAL_IDLE_FULL_VERIFY": "0"}):
            self.pool._restore(self.pool.multi)
        self.assertEqual(self.calls[-1], "onload")

    def test_stale_generation_rejects_before_target_write(self):
        self.version = 7
        self.pool.arm(1)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.pool._restore(self.pool.multi)
        self.assertNotIn("stage", self.calls)
        self.assertNotIn("onload", self.calls)

    def test_failed_stage_never_restores(self):
        self.stage_error = True
        self.pool.arm(1)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.pool._restore(self.pool.multi)
        self.assertNotIn("onload", self.calls)

    def test_cancel_joins_background(self):
        self.pool.arm(1)
        self.pool.cancel_entry()
        self.assertIsNone(self.pool._cpu_stage)
        self.assertNotIn("onload", self.calls)

    def test_source_restores_while_destination_stage_is_pending(self):
        from threading import Event

        gate = Event()
        original = self.pool.multi.idle_training_state

        def delayed(action, path):
            if not gate.wait(5):
                raise RuntimeError("test stage gate timeout")
            return original(action, path)

        self.pool.multi.idle_training_state = delayed
        self.pool.primary.onload = lambda: self.calls.append("source_onload")
        self.pool._entry_budget = types.SimpleNamespace(observe=lambda *a: None)
        self.pool._entry_world = 1
        self.pool._entry_compute_started = time.monotonic()
        self.pool.current = self.pool.primary
        try:
            self.pool.arm(1)
            with patch.dict(os.environ, {"AREAL_IDLE_FULL_VERIFY": "0"}):
                self.pool._restore(self.pool.primary)
            self.assertIn("source_onload", self.calls)
            self.assertFalse(self.pool._cpu_stage.done())
            self.assertTrue((self.pool.path / "generation-0").exists())
            self.pool.finish(1)
            self.assertFalse(self.pool._cpu_stage.done())
            self.assertTrue((self.pool.path / "generation-0").exists())
            self.pool.arm(2)  # Coalesce instead of waiting for obsolete standby.
            self.assertEqual(self.pool._maintenance_batch, 1)
        finally:
            gate.set()
        self.pool.cancel_entry()

    def test_historical_prior_is_not_read(self):
        self.pool._remaining_preparation = lambda batch: (1, {1: None, 2: None}, False)
        self.pool._entry_budget = Budget()
        with patch.dict(
            os.environ, {"AREAL_IDLE_PREEMPT_ESTIMATE": "not even valid JSON"}
        ):
            choice = self.pool._full_entry_prediction(1, {"generating": 0})
        self.assertFalse(choice["accepted"])
        self.assertEqual(choice["reason"], "insufficient_current_run_evidence")

    def test_first_preemption_is_deferred_until_data_ready(self):
        self.pool._prepared = {"source": self.pool.primary}
        with (
            patch.dict(os.environ, {"AREAL_IDLE_PREEMPT_FIRST": "1"}),
            patch.object(Parent, "arm") as arm,
        ):
            self.pool.arm(1)
            arm.assert_not_called()

    def test_preemption_uses_real_group_policy_and_larger_rank_budget(self):
        import importlib.util
        import sys

        name = "areal.trainer.colocation_memory"
        spec = importlib.util.spec_from_file_location(
            name, source.parent / "colocation_memory.py"
        )
        mem = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mem)
        p = self.pool
        cfg = types.SimpleNamespace(
            device_budget_gib=None, train_workspace_gib=1, safety_margin_gib=1
        )
        p.trainer._memory_policy = mem.GroupColocationMemoryPolicy(
            cfg, lambda **k: None, lambda **k: None, lambda x: None, 1
        )
        p._prepared = dict(
            source=p.primary,
            destination=p.multi,
            exported={"version": 0},
            imported={"version": 0},
        )
        p.multi.colocation_memory = lambda device_rank: dict(
            free_bytes=100 * mem.GIB,
            total_bytes=100 * mem.GIB,
            restore_bytes=(10 + device_rank) * mem.GIB,
            future_optimizer_bytes=0,
            gradient_bytes=0,
        )
        targets = []

        def group(ranks, method, **kw):
            targets.append(kw["target_free_bytes"])
            return dict(free_bytes=50 * mem.GIB, total_bytes=100 * mem.GIB)

        p.trainer.rollout = types.SimpleNamespace(
            wait_shared_publication=lambda: None,
            wait_running_rebalance=lambda: None,
            dispatcher=types.SimpleNamespace(
                generation_work_snapshot=lambda: {"generating": 3}
            ),
            _request_pool=object(),
            workers=[0, 1],
            pause=lambda: None,
            _collective_rpc=lambda *a, **kw: None,
            _group_rpc=group,
        )
        p._restore_multi_from_dram = lambda: self.calls.append("restore")
        env = {
            "AREAL_IDLE_PREEMPT_FIRST": "1",
            "AREAL_IDLE_ENTRY_VALIDATE_WORLDS": "",
            "AREAL_IDLE_PREEMPT_ESTIMATE": json.dumps(
                dict(
                    single_seconds=65,
                    dual_seconds=35,
                    handoff_seconds=5,
                    inference_delay_seconds=15,
                )
            ),
        }
        p._full_entry_prediction = lambda batch, snapshot: {"accepted": True}
        with patch.dict(sys.modules, {name: mem}), patch.dict(os.environ, env):
            self.assertTrue(p._try_initial_preemption(1))
        self.assertEqual(targets, [13 * mem.GIB, 13 * mem.GIB])
        self.assertIn("restore", self.calls)

    def test_publication_records_only_wait_as_drain(self):
        import sys
        from dataclasses import dataclass, replace
        from unittest.mock import Mock

        @dataclass
        class Meta:
            version: int = 0
            type: str = "disk"
            npu_share_snapshot: bool = False
            clear_checkpoint_after_load: bool = False

            def with_version(self, version):
                return replace(self, version=version)

        for requests in [0, 2]:
            clock = [0.0]

            def collective(method, **kw):
                if method in ("colocation_memory", "continue_generation"):
                    clock[0] += 2.5

            def wait_empty(rank, **kw):
                if requests:
                    clock[0] += 10.0

            rollout = Mock()
            rollout._collective_rpc.side_effect = collective
            rollout._group_rpc.side_effect = lambda *a, **kw: clock.__setitem__(
                0, clock[0] + 2.0
            )
            rollout._request_pool.active_requests.side_effect = lambda rank: list(
                range(requests)
            )
            rollout._request_pool.wait_empty.side_effect = wait_empty
            self.pool.trainer.rollout = rollout
            self.pool.trainer.weight_update_meta = Meta()
            self.pool.current = Mock()
            self.pool._preempted_inference = True
            self.pool._entry_budget = Budget()
            module = types.SimpleNamespace(call_maybe_async=lambda f: f())
            with (
                patch.dict(sys.modules, {"areal.infra.utils.concurrent": module}),
                patch.object(time, "monotonic", lambda: clock[0]),
            ):
                self.pool.publish_multi(1)
            self.assertEqual(self.pool._entry_budget.estimate(1, 2, "return"), 9.0)
            drain = self.pool._entry_budget.drain_samples[-1]
            self.assertEqual(drain[2], requests * 2)
            self.assertEqual(drain[3], 20.0 if requests else 0.0)

    def test_direct_multi_entry_parks_primary_before_restore(self):
        self.pool._restore = lambda actor: self.calls.append("restore_multi")
        self.pool._restore_multi_from_dram()
        self.assertEqual(self.calls, ["park", "restore_multi"])

    def test_remaining_cost_uses_one_current_run_sample(self):
        self.pool._entry_budget = Budget()
        for world, compute in [(1, 65), (2, 35)]:
            for phase, seconds in [
                ("compute", compute),
                ("return", 4),
                ("tail", 0),
                ("handoff", 0),
            ]:
                self.pool._entry_budget.observe(1, world, phase, seconds)
        self.pool._remaining_preparation = lambda batch: (2, {1: 22.0, 2: 2.0}, False)
        self.assertEqual(
            self.pool._full_entry_prediction(2, {"generating": 0})["world_size"], 2
        )
        # No matched drain sample: don't preempt runnable inference.
        self.assertEqual(
            self.pool._full_entry_prediction(2, {"generating": 16})["world_size"], 1
        )
        self.pool.trainer.rollout._request_pool.active_requests = lambda rank: list(
            range(8)
        )
        self.pool._entry_budget.observe_drain(1, 2, 16, 4)
        self.assertEqual(
            self.pool._full_entry_prediction(2, {"generating": 16})["world_size"], 2
        )
        self.pool._remaining_preparation = lambda batch: (1, {1: 2.0, 2: 80.0}, False)
        self.assertEqual(
            self.pool._full_entry_prediction(2, {"generating": 16})["world_size"], 1
        )

    def test_retaining_dual_does_not_rpc_to_busy_standby(self):
        self.pool.snapshot_actor = self.pool.multi
        self.pool.primary.offload = lambda: (_ for _ in ()).throw(
            AssertionError("blocked on standby")
        )
        self.pool._restore = lambda actor: self.calls.append("restore_multi")
        self.pool._restore_multi_from_dram()
        self.assertEqual(self.calls, ["restore_multi"])

    def test_all_preempt_arm_never_starts_device_prefetch(self):
        self.pool._join_cpu_stage = lambda: None
        self.pool._start_maintenance = lambda batch: None
        with (
            patch.dict(os.environ, {"AREAL_IDLE_PREEMPT_ALL": "1"}),
            patch.object(Parent, "arm") as arm,
        ):
            self.pool.arm(2)
            arm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
