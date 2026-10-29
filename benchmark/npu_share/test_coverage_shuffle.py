"""Coverage order tests use plan mode and never start experiments."""

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("coverage_bench.sh")


class ShuffleTests(unittest.TestCase):
    def plan(self, *options):
        with tempfile.TemporaryDirectory() as folder:
            profile = Path(folder) / "profile.json"
            profile.write_text(
                json.dumps({"workspace_root": folder, "timezone": "UTC"})
            )
            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--suite",
                    "all",
                    "--mode",
                    "plan",
                    "--profile",
                    str(profile),
                    *options,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(list(Path(folder).iterdir()), [profile])
            return result

    def ids(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return re.findall(r"^\[([A-Z][0-9]{2})\]", result.stdout, re.M)

    def test_reproducible_complete_order(self):
        default = self.ids(self.plan())
        first = self.ids(self.plan("--shuffle-seed", "42"))
        self.assertEqual(len(first), 28)
        self.assertCountEqual(first, default)
        self.assertEqual(first, self.ids(self.plan("--shuffle-seed", "42")))
        self.assertNotEqual(first, default)
        self.assertNotEqual(first, self.ids(self.plan("--shuffle-seed", "43")))

    def test_explicit_cases(self):
        result = self.plan("--cases", "T01 T02 E01", "--shuffle")
        self.assertCountEqual(self.ids(result), ["T01", "T02", "E01"])
        self.assertRegex(result.stdout, r"Coverage shuffle seed=[0-9]+;")

    def test_invalid_seed(self):
        self.assertNotEqual(self.plan("--shuffle-seed", "oops").returncode, 0)


if __name__ == "__main__":
    unittest.main()
