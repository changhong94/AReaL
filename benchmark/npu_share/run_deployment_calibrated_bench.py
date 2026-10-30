"""Run isolated deployment calibration, then a fresh fixed/shared comparison."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    opts, extra = parser.parse_known_args()
    if any(
        arg.startswith(
            (
                "--return-calibration",
                "--workspace-root",
                "--prepare-only",
                "--render-only",
            )
        )
        for arg in extra
    ):
        parser.error("Calibration workflow controls cache and workspace arguments")
    root = opts.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Never overwrite or silently relaunch an existing experiment.
    state_path = root / "state.json"
    with state_path.open("x") as f:
        json.dump(dict(phase="starting"), f)
    started = time.monotonic()
    state = dict(
        phase="starting",
        restore="new_process_original_model_no_optimizer_or_requests_reused",
    )

    def save():
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(state_path)

    def run(stage, args):
        state["phase"] = stage
        save()
        before = time.monotonic()
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("run_bench.py")),
            "--profile",
            str(opts.profile),
            *extra,
            "--workspace-root",
            str(root / stage),
            *args,
        ]
        with (root / (stage + ".log")).open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        text = (root / (stage + ".log")).read_text()
        matches = re.findall(r"^WORKSPACE=(.+)$", text, re.MULTILINE)
        if not matches:
            raise RuntimeError("Missing completed workspace for " + stage)
        workspace = Path(matches[-1])
        if json.loads((workspace / "state.json").read_text())["phase"] != "complete":
            raise RuntimeError("Incomplete stage: " + stage)
        state[stage] = dict(
            workspace=str(workspace), wall_seconds=time.monotonic() - before
        )
        save()
        return workspace

    try:
        warmup = run(
            "deployment-calibration", ["--batches", "2", "--return-calibration-out"]
        )
        cache = warmup / "return-calibration.json"
        if not cache.is_file():
            raise RuntimeError(
                "Calibration did not produce a completed return/restore sample"
            )
        measured = run("measured", ["--return-calibration-in", str(cache)])
        summary = json.loads((measured / "summary.json").read_text())
        state.update(
            phase="complete",
            loop_seconds=summary["loop_seconds"],
            reduction_percent=summary["reduction_percent"],
            calibration_costs=json.loads(cache.read_text())["costs"],
            total_wall_seconds=time.monotonic() - started,
        )
        save()
    except BaseException as exc:
        state.update(
            phase="failed",
            error=str(exc),
            total_wall_seconds=time.monotonic() - started,
        )
        save()
        raise


if __name__ == "__main__":
    main()
