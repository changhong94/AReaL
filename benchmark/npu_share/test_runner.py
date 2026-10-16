"""CPU-only orchestration tests. Docker and NPU calls are mocked."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_bench as run


class RunnerTests(unittest.TestCase):
    def exercise(self, stop_fails=False):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name in ["repo", "model", "train", "test"]:
                (base / name).mkdir()
            profile = dict(
                repo=str(base / "repo"),
                model=str(base / "model"),
                workspace_root=str(base / "runs"),
                devices=[1, 2],
                timezone="UTC",
                plot_python="plot-python",
                npu_smi="smi",
                driver="/driver",
                cann_env=["/cann/env.sh"],
                engines={"fsdp": dict(image="tag", python="/python/bin/python3")},
                datasets={
                    "math": dict(train=str(base / "train"), test=str(base / "test"))
                },
            )
            path = base / "profile.json"
            path.write_text(json.dumps(profile))
            opts = SimpleNamespace(
                profile=path,
                rl="math",
                engine="fsdp",
                concurrency=1,
                max_head_offpolicyness=3,
                batches=100,
                npu_count=2,
                inference_tp=1,
                workspace_root=None,
                timeout=1,
                prepare_only=False,
                render_only=None,
            )
            calls = []

            def cmd(args, **kwargs):
                calls.append(list(map(str, args)))
                return SimpleNamespace(returncode=0)

            raw = "Process id\n" + "No running processes found in NPU\n" * 8

            def cap(args):
                if args[0] == "smi":
                    return raw
                if "image" in args:
                    return "sha256:immutable-image"
                if "{{.State.ExitCode}}" in args:
                    return "0"
                return ""

            def isrunning(name):
                return stop_fails

            def snap(repo, dest):
                dest.mkdir()
                return {"head": "test", "files": {}}

            telemetry = SimpleNamespace(
                sampler=lambda *a: None,
                parse_info=lambda raw: ({str(i): {} for i in range(8)}, []),
            )
            with (
                patch.object(run, "arguments", return_value=opts),
                patch.object(run, "command", side_effect=cmd),
                patch.object(run, "capture", side_effect=cap),
                patch.object(run, "running", side_effect=isrunning),
                patch.object(run, "freeze", side_effect=snap),
                patch.object(run, "render") as render,
                patch.object(
                    run,
                    "signal",
                    SimpleNamespace(
                        SIGINT=2, SIGTERM=15, SIG_IGN=1, signal=lambda *a: None
                    ),
                ),
                patch.object(run.time, "sleep"),
                patch.object(run.time, "monotonic", side_effect=[0, 2, 0, 2]),
                patch.object(
                    run.subprocess, "run", return_value=SimpleNamespace(returncode=0)
                ),
                patch.dict(sys.modules, {"telemetry": telemetry}),
                patch.object(run.Path, "home", return_value=base),
            ):
                if stop_fails:
                    with self.assertRaises(SystemExit):
                        run.main()
                else:
                    run.main()
            workspace = next((base / "runs").iterdir())
            state = json.loads((workspace / "state.json").read_text())
            settings = json.loads((workspace / "settings.json").read_text())
            self.assertEqual(settings["questions"], 1600)
            self.assertEqual(settings["concurrency"], 1)
            self.assertEqual(settings["max_head_offpolicyness"], 3)
            self.assertTrue((workspace / "benchmark/templates/entry.py").exists())
            self.assertTrue((workspace / "benchmark/run_bench.py").exists())
            self.assertFalse(any(a[:2] == ["docker", "stop"] for a in calls))
            training = [a for a in calls if a[:3] == ["docker", "run", "-d"]]
            self.assertTrue(training)
            self.assertTrue(all("sha256:immutable-image" in a for a in training))
            restarts = [a for a in calls if a[:2] == ["docker", "start"]]
            if stop_fails:
                self.assertEqual(state["phase"], "failed")
                self.assertIn("cleanup_error", state)
                self.assertFalse(restarts)
                render.assert_not_called()
            else:
                self.assertEqual(state["phase"], "complete")
                self.assertEqual(state["completed"], ["fixed", "latest"])
                self.assertFalse(restarts)
                render.assert_called_once()

    def test_matched_run_does_not_start_external_containers(self):
        self.exercise()

    def test_failed_stop_reports_cleanup_failure(self):
        self.exercise(stop_fails=True)

    def test_progress_preserves_partial_records_and_does_not_repeat_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phases.jsonl"
            messages = []
            progress = SimpleNamespace(message=messages.append)
            path.write_text('{"event":"initialize.end"}\n{"event":"_finish_step.end"')
            offset, completed = run.report_events(path, 0, 0, 6, progress)
            self.assertEqual(completed, 0)
            with path.open("a") as stream:
                stream.write("}\n")
            offset, completed = run.report_events(path, offset, completed, 6, progress)
            self.assertEqual(completed, 1)
            offset, completed = run.report_events(path, offset, completed, 6, progress)
            self.assertEqual(len(messages), 2)
            self.assertIn("1/6", messages[-1])

    def test_npu_count_requires_equal_nonempty_groups(self):
        for value in ["0", "1", "3", "-2"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                run.npu_count(value)
        self.assertEqual(run.npu_count("4"), 4)
        self.assertEqual(
            run.allocation_counts(4),
            {
                "total": 4,
                "training": 2,
                "dedicated_inference": 2,
                "shared_inference_max": 4,
            },
        )
        run.validate_topology(2)
        for count in (2, 4, 8):
            for tp in (1, 2, 4):
                if count // 2 % tp == 0:
                    run.validate_topology(count, tp)
                else:
                    with self.assertRaises(ValueError):
                        run.validate_topology(count, tp)

    def test_offpolicy_argument(self):
        with patch.object(sys, "argv", ["run_bench.py"]):
            self.assertEqual(run.arguments().max_head_offpolicyness, 2)
        with patch.object(
            sys, "argv", ["run_bench.py", "--max-head-offpolicyness", "4"]
        ):
            self.assertEqual(run.arguments().max_head_offpolicyness, 4)
        for value in ["0", "-1"]:
            with patch.object(
                sys, "argv", ["run_bench.py", "--max-head-offpolicyness", value]
            ):
                with self.assertRaises(SystemExit):
                    run.arguments()

    def test_invalid_counts_are_rejected(self):
        for value in ["0", "-1"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                run.positive(value)
        self.assertEqual(run.positive("100"), 100)


if __name__ == "__main__":
    unittest.main()
