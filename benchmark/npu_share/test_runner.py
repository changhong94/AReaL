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
    def test_reuse_route_defaults_to_push_and_allows_poll(self):
        import run_reuse_dev_bench as reuse

        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            profile.write_text("{}")
            for flags, expected in [
                ([], ["push", "push"]),
                (["--shared-route-transports", "poll,push"], ["poll", "push"]),
            ]:
                opts = SimpleNamespace(
                    adaptive_concurrency="capacity",
                    render_only=False,
                    npu_count=2,
                    inference_tp=1,
                    profile=profile,
                )
                with (
                    patch.object(sys, "argv", ["reuse", *flags]),
                    patch.object(reuse.run_bench, "arguments", return_value=opts),
                    patch.object(reuse, "run") as execute,
                ):
                    reuse.main()
                self.assertEqual(
                    execute.call_args.args[0].shared_route_transports, expected
                )

    def test_shared_host_checks_only_selected_idle_devices(self):
        raw = "Process id\nNo running processes found in NPU 4\nNo running processes found in NPU 5\n"
        devices = {str(i): {"health": "OK"} for i in range(8)}
        run.validate_idle_devices(raw, devices, [{"device": "0"}], [4, 5])
        for selected, processes in [
            ([0, 4], []),
            ([4, 5], [{"device": "5"}]),
            ([4, 7], []),
        ]:
            with self.assertRaises(RuntimeError):
                run.validate_idle_devices(raw, devices, processes, selected)
        devices["4"]["health"] = "Warning"
        with self.assertRaises(RuntimeError):
            run.validate_idle_devices(raw, devices, [], [4, 5])
        with self.assertRaises(RuntimeError):
            run.validate_idle_devices("", devices, [], [4, 5])

    def test_code_reward_policy_is_opt_in(self):
        with patch.object(sys, "argv", ["run_bench.py"]):
            self.assertEqual(run.arguments().code_reward_scheduling, "answer")
        with patch.object(
            sys,
            "argv",
            ["run_bench.py", "--rl", "code", "--code-reward-scheduling", "question"],
        ):
            self.assertEqual(run.arguments().code_reward_scheduling, "question")

    def test_r027_defaults_only_enabled_for_shared_arm(self):
        for name in ("AREAL_PROFIT_AWARE_RETURN", "AREAL_RESIDENT_BUDGET"):
            self.assertEqual(run.environment(True)[name], "1")
            self.assertEqual(run.environment(False)[name], "0")

    def test_coverage_workload_values_and_explicit_defaults(self):
        import shlex
        import subprocess

        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            profile.write_text(
                json.dumps({"workspace_root": directory, "timezone": "UTC"})
            )
            output = subprocess.check_output(
                [
                    "bash",
                    str(Path(__file__).with_name("coverage_bench.sh")),
                    "--suite",
                    "all",
                    "--mode",
                    "plan",
                    "--profile",
                    str(profile),
                ],
                text=True,
            )
        commands = [
            shlex.split(line)
            for line in output.splitlines()
            if line.startswith("python3 ")
        ]
        self.assertEqual(len(commands), 28)

        def value(cmd, flag):
            return cmd[cmd.index(flag) + 1]

        self.assertEqual(
            {value(cmd, "--batch-size") for cmd in commands}, {"16", "32", "64"}
        )
        self.assertEqual(
            {value(cmd, "--answers") for cmd in commands}, {"4", "8", "16"}
        )
        self.assertEqual(
            sum(value(cmd, "--inference-mode") == "eager" for cmd in commands), 2
        )

    def test_workload_defaults_override_and_validation(self):
        with patch.object(sys, "argv", ["run_bench.py"]):
            opts = run.arguments()
            self.assertEqual((opts.batch_size, opts.answers), (32, 8))
        with patch.object(
            sys, "argv", ["run_bench.py", "--batch-size", "64", "--answers", "16"]
        ):
            opts = run.arguments()
            self.assertEqual((opts.batch_size, opts.answers), (64, 16))
        for option in ["--batch-size", "--answers"]:
            for value in ["0", "-1", "bad"]:
                with (
                    self.subTest(option=option, value=value),
                    patch.object(sys, "argv", ["run_bench.py", option, value]),
                ):
                    with self.assertRaises(SystemExit):
                        run.arguments()

    def test_inference_mode_default_and_override(self):
        with patch.object(sys, "argv", ["run_bench.py"]):
            self.assertEqual(run.arguments().inference_mode, "graph")
        with patch.object(sys, "argv", ["run_bench.py", "--inference-mode", "eager"]):
            self.assertEqual(run.arguments().inference_mode, "eager")

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
                code_reward_scheduling="answer",
                engine="fsdp",
                concurrency=1,
                max_head_offpolicyness=3,
                running_rebalance="shadow",
                adaptive_concurrency="off",
                inference_mode="graph",
                batches=100,
                batch_size=64,
                answers=16,
                npu_count=2,
                inference_tp=1,
                inference_training_ratio="1:1",
                workspace_root=None,
                timeout=1,
                prepare_only=False,
                render_only=None,
            )
            calls = []

            def cmd(args, **kwargs):
                calls.append(list(map(str, args)))
                return SimpleNamespace(returncode=0)

            raw = "Process id\n" + "".join(
                f"No running processes found in NPU {i}\n" for i in range(8)
            )

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
                parse_info=lambda raw: (
                    {str(i): {"health": "OK"} for i in range(8)},
                    [],
                ),
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
            self.assertEqual(settings["inference_mode"], "graph")
            self.assertEqual(settings["questions"], 6400)
            self.assertEqual(settings["batch_size"], 64)
            self.assertEqual(settings["answers"], 16)
            self.assertEqual(settings["concurrency"], 1)
            self.assertEqual(settings["max_head_offpolicyness"], 3)
            self.assertEqual(settings["running_rebalance"], "shadow")
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
        for value in ["0", "1", "-2"]:
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

    def test_unequal_topology(self):
        for count, ratio, tp, train, inference in [
            (3, "2:1", 1, 1, 2),
            (4, "3:1", 1, 1, 3),
            (6, "2:1", 2, 2, 4),
            (8, "3:1", 2, 2, 6),
        ]:
            run.validate_topology(count, tp, ratio)
            allocation = run.allocation_counts(count, ratio)
            self.assertEqual(
                (allocation["training"], allocation["dedicated_inference"]),
                (train, inference),
            )
        for count, ratio, tp in [(4, "2:1", 1), (3, "2:1", 2), (8, "3:1", 4)]:
            with self.assertRaises(ValueError):
                run.validate_topology(count, tp, ratio)

    def test_running_policy_argument(self):
        with patch.object(sys, "argv", ["run_bench.py"]):
            self.assertEqual(run.arguments().running_rebalance, "kv")
        with patch.object(
            sys, "argv", ["run_bench.py", "--running-rebalance", "active"]
        ):
            self.assertEqual(run.arguments().running_rebalance, "active")

    def test_running_policy_can_be_disabled(self):
        with patch.object(sys, "argv", ["run_bench.py", "--running-rebalance", "off"]):
            self.assertEqual(run.arguments().running_rebalance, "off")

    def test_adaptive_concurrency_defaults_to_capacity(self):
        with patch.object(sys, "argv", ["run_bench.py"]):
            self.assertEqual(run.arguments().adaptive_concurrency, "capacity")
        with patch.object(
            sys, "argv", ["run_bench.py", "--adaptive-concurrency", "off"]
        ):
            self.assertEqual(run.arguments().adaptive_concurrency, "off")

    def test_offpolicy_argument(self):
        with patch.object(sys, "argv", ["run_bench.py"]):
            self.assertEqual(run.arguments().max_head_offpolicyness, 2)
        with patch.object(
            sys, "argv", ["run_bench.py", "--max-head-offpolicyness", "4"]
        ):
            self.assertEqual(run.arguments().max_head_offpolicyness, 4)
        with patch.object(
            sys, "argv", ["run_bench.py", "--max-head-offpolicyness", "0"]
        ):
            self.assertEqual(run.arguments().max_head_offpolicyness, 0)
        for value in ["-1"]:
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

    def test_drain_policies_require_training_priority_inference(self):
        for enabled in ("0", "1"):
            with patch.dict(
                "os.environ", {"AREAL_TRAIN_PRIORITY_INFERENCE": enabled}, clear=True
            ):
                for shared in (False, True):
                    env = run.environment(shared)
                    expected = "1" if shared and enabled == "1" else "0"
                    self.assertEqual(env["AREAL_ADAPTIVE_DRAIN"], expected)
                    self.assertEqual(env["AREAL_BATCH_DRAIN"], expected)
                    if expected == "1":
                        self.assertEqual(env["AREAL_DRAIN_METRICS"], "1")
                        self.assertEqual(env["AREAL_BATCH_DRAIN_METRICS"], "1")

    def test_safe_shared_policy_defaults_and_explicit_overrides(self):
        with patch.dict("os.environ", {}, clear=True):
            env = run.environment(True)
            self.assertEqual(env["AREAL_ROUTE_TRANSPORT"], "push")
            self.assertEqual(env["AREAL_RETURN_CALIBRATION_PROBE"], "1")
            self.assertEqual(env["AREAL_TRAIN_PRIORITY_INFERENCE"], "0")
        overrides = {
            "AREAL_TRAIN_PRIORITY_INFERENCE": "1",
            "AREAL_RETURN_CALIBRATION_PROBE": "0",
            "AREAL_ROUTE_TRANSPORT": "poll",
        }
        with patch.dict("os.environ", overrides):
            self.assertEqual(
                run.environment(True)["AREAL_TRAIN_PRIORITY_INFERENCE"], "1"
            )
            self.assertEqual(
                run.environment(False)["AREAL_TRAIN_PRIORITY_INFERENCE"], "0"
            )
            self.assertEqual(
                run.environment(True)["AREAL_RETURN_CALIBRATION_PROBE"], "0"
            )
            self.assertEqual(run.environment(True)["AREAL_ROUTE_TRANSPORT"], "poll")


if __name__ == "__main__":
    unittest.main()
