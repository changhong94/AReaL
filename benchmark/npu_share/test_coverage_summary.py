"""CPU-only suite tests with a fake benchmark runner; no containers or NPUs."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from coverage_summary import write_reports


class CoverageTests(unittest.TestCase):
    def test_failed_case_does_not_skip_remaining_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["coverage_bench.sh", "coverage_summary.py"]:
                shutil.copy(Path(__file__).with_name(name), root / name)
            profile = root / "profile.json"
            profile.write_text(json.dumps({"workspace_root": str(root / "runs")}))
            (root / "run_bench.py").write_text("""
import json,sys
from pathlib import Path
args=sys.argv[1:]
def arg(key):return args[args.index(key)+1]
base=Path(arg('--workspace-root'))
workspace=base/arg('--engine');workspace.mkdir()
print('WORKSPACE='+str(workspace),flush=True)
failed=arg('--engine')=='fsdp'
(workspace/'state.json').write_text(json.dumps({'phase':'failed' if failed else 'complete','error':'bad <test> | detail' if failed else ''}))
if not failed:
 (workspace/'summary.json').write_text(json.dumps({'loop_seconds':{'fixed':100,'latest':120},'token_audit':{'fixed':{'reported':500},'latest':{'reported':520}}}))
sys.exit(1 if failed else 0)
""")
            result = subprocess.run(
                [
                    "bash",
                    str(root / "coverage_bench.sh"),
                    "--profile",
                    str(profile),
                    "--cases",
                    "C01 C02",
                    "--mode",
                    "run",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            workspace = next((root / "runs").iterdir())
            report = json.loads((workspace / "summary.json").read_text())
            self.assertEqual(
                [r["status"] for r in report["cases"]], ["failed", "complete"]
            )
            self.assertEqual(report["cases"][1]["reduction_percent"], -20)
            self.assertIsNone(report["cases"][0]["reduction_percent"])
            for filename in [
                "summary.md",
                "summary.zh.md",
                "index.html",
                "index.zh.html",
            ]:
                self.assertIn("-20.00%", (workspace / filename).read_text())
            self.assertIn("bad &lt;test&gt;", (workspace / "index.html").read_text())
            self.assertIn("fsdp/state.json", (workspace / "index.html").read_text())

    def test_interruption_and_unstarted_cases_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "planned-cases.tsv").write_text("case\trl\nC01\tmath\nC02\tcode\n")
            (root / "C01.log").write_text("still initializing\n")
            self.assertEqual(
                [r["status"] for r in write_reports(root)], ["running", "not_run"]
            )
            self.assertEqual(
                [r["status"] for r in write_reports(root, final=True)],
                ["interrupted", "not_run"],
            )

    def test_prepare_is_not_a_successful_training_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cases.tsv").write_text("case\texit_code\nC01\t0\n")
            report = write_reports(root, final=True)
            self.assertEqual(report[0]["status"], "prepared")
            self.assertIsNone(report[0]["reduction_percent"])


if __name__ == "__main__":
    unittest.main()
