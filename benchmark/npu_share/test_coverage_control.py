import subprocess
import tempfile
import time
import unittest
from pathlib import Path

CONTROL = Path(__file__).with_name("coverage_control.py")


class CoverageControlTest(unittest.TestCase):
    def test_pause_drains_child_then_resume_continues(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            script = root / "coverage_bench.sh"
            script.write_text(
                'sleep 2 &\nwait\necho resumed > "' + str(root / "next") + '"\n'
            )
            launch = subprocess.Popen(["bash", str(script)])
            try:
                time.sleep(0.2)

                def control(action, *extra):
                    return subprocess.run(
                        [
                            "python3",
                            str(CONTROL),
                            action,
                            "--pid",
                            str(launch.pid),
                            *extra,
                        ],
                        capture_output=True,
                        text=True,
                    )

                self.assertIn("pause_pending", control("pause").stdout)
                self.assertNotEqual(control("rerun", "--cases", "T01").returncode, 0)
                time.sleep(2.2)
                self.assertFalse((root / "next").exists())
                self.assertIn("paused:", control("status").stdout)
                self.assertEqual(control("resume").returncode, 0)
                launch.wait(timeout=3)
                self.assertTrue((root / "next").exists())
            finally:
                if launch.poll() is None:
                    launch.kill()
                    launch.wait()


if __name__ == "__main__":
    unittest.main()
