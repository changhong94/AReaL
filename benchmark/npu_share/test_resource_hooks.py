"""Host hook execution does not require accelerators."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import run_bench as run


class HookTests(unittest.TestCase):
    def test_commands_run_once_and_receive_context(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            script = p / "hook.sh"
            script.write_text(
                'echo "$AREAL_BENCH_DEVICES" >> "$AREAL_BENCH_WORKSPACE/calls"\n'
            )
            h = run.ResourceHooks(
                {
                    "resource_hooks": {
                        "before": ["bash", str(script)],
                        "after": ["bash", str(script)],
                    }
                },
                p,
                [1, 2],
                Mock(),
            )
            h.finish()
            self.assertFalse((p / "calls").exists())
            h.start()
            h.finish()
            h.finish()
            self.assertEqual((p / "calls").read_text(), "1,2\n1,2\n")

    def test_failure_is_visible_and_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            h = run.ResourceHooks(
                {"resource_hooks": {"after": ["false"]}}, Path(d), [1], Mock()
            )
            h.start()
            with self.assertRaises(run.subprocess.CalledProcessError):
                h.finish()
            h.finish()

    def test_missing_hooks_are_noop(self):
        h = run.ResourceHooks({}, Path("."), [1], Mock())
        h.start()
        h.finish()
