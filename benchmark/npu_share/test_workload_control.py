"""Exercise the exact-workload dispatcher with supported benchmark batch sizes."""

import importlib.util
import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from areal.utils import request_trace


class WorkloadControlTests(unittest.TestCase):
    def load_control(self, batch_size):
        settings = dict(
            batch_size=batch_size, questions=batch_size * 6, batches=6, timeout=60
        )
        path = Path(__file__).parent / "templates/fair48_control.py"
        spec = importlib.util.spec_from_file_location(
            "workload_control_under_test", path
        )
        module = importlib.util.module_from_spec(spec)
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            if str(path) == "/work/settings.json":
                return json.dumps(settings)
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read_text):
            spec.loader.exec_module(module)
        return module, settings

    def trainer(self, batch_size, shared):
        ready = []
        dispatcher = SimpleNamespace(
            _input_cv=threading.Condition(),
            _pending_inputs=[],
            staleness_manager=SimpleNamespace(get_pending_limit=lambda: batch_size * 2),
            runner=SimpleNamespace(
                max_queue_size=batch_size * 4, get_input_queue_size=lambda: len(ready)
            ),
            submit_task_input=lambda task: ready.append(task),
        )

        def wait_results(count, timeout):
            result = ready[:count]
            del ready[:count]
            return result

        dispatcher.wait_results = wait_results
        return SimpleNamespace(
            rollout=SimpleNamespace(dispatcher=dispatcher),
            config=SimpleNamespace(npu_share=shared),
            _publish_weights=lambda version: None,
            _finish_step=lambda **kwargs: None,
        )

    def test_exact_workload_across_all_batch_sizes_and_arms(self):
        for batch_size in (16, 32, 64):
            for shared in (False, True):
                with self.subTest(batch_size=batch_size, shared=shared):
                    module, settings = self.load_control(batch_size)
                    trainer = self.trainer(batch_size, shared)
                    tasks = [
                        SimpleNamespace(task_id=i, data={"question": i})
                        for i in range(settings["questions"])
                    ]
                    generator = iter(tasks)
                    with patch.object(request_trace, "emit"):
                        submitted = module.configure_workload(trainer)
                        results = []
                        for _ in range(settings["batches"]):
                            batch = trainer.rollout.dispatcher.active_submit_and_wait(
                                generator, batch_size
                            )
                            self.assertEqual(len(batch), batch_size)
                            results.extend(batch)
                    self.assertEqual(
                        [r.task_id for r in results], list(range(settings["questions"]))
                    )
                    self.assertEqual(len(submitted), settings["questions"])
                    self.assertIsNone(next(generator, None))

    def test_mismatched_or_dynamic_batch_rejected_before_submission(self):
        module, _ = self.load_control(32)
        for size, dynamic in ((16, False), (32, True)):
            trainer = self.trainer(32, True)
            submitted = module.configure_workload(trainer)
            with self.assertRaisesRegex(ValueError, "configured static batch size 32"):
                trainer.rollout.dispatcher.active_submit_and_wait(
                    iter(()), size, dynamic
                )
            self.assertEqual(submitted, [])

    def test_single_batch_queue_admits_work_without_initial_deadlock(self):
        for shared in (False, True):
            module, settings = self.load_control(32)
            trainer = self.trainer(32, shared)
            dispatcher = trainer.rollout.dispatcher
            dispatcher.runner.max_queue_size = 32
            original_wait = dispatcher.wait_results

            def wait_results(count, timeout):
                self.assertGreater(dispatcher.runner.get_input_queue_size(), 0)
                self.assertLessEqual(dispatcher.runner.get_input_queue_size(), 32)
                return original_wait(count, timeout)

            dispatcher.wait_results = wait_results
            tasks = iter(
                SimpleNamespace(task_id=i, data={"question": i})
                for i in range(settings["questions"])
            )
            with patch.object(request_trace, "emit"):
                submitted = module.configure_workload(trainer)
                for _ in range(settings["batches"]):
                    self.assertEqual(
                        len(dispatcher.active_submit_and_wait(tasks, 32)), 32
                    )
            self.assertEqual(len(submitted), settings["questions"])
