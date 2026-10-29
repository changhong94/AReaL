#!/usr/bin/env python3
"""Pause a live coverage launcher without interrupting its current comparison.

Only the Bash launcher receives SIGSTOP/SIGCONT, never its process group.
The active comparison continues and releases its resources normally. This is
an in-memory pause: the host and launcher must remain alive until resume.
"""

import argparse
import os
import re
import signal
import subprocess
import time
from pathlib import Path


def process(pid):
    root = Path("/proc") / str(pid)
    raw = (root / "stat").read_text()
    fields = raw[raw.rfind(")") + 2 :].split()
    return fields[0], int(fields[1]), fields[19]


def children(pid):
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                state, parent, _ = process(int(path.name))
                if parent == pid and state != "Z":
                    result.append(int(path.name))
            except (OSError, ValueError):
                pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pause", "resume", "status", "rerun"])
    parser.add_argument(
        "--pid",
        type=int,
        required=True,
        help="Bash coverage launcher PID, not a training PID",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the current comparison to finish after requesting pause",
    )
    parser.add_argument("--cases", help="Explicit case IDs to rerun in a new workspace")
    args = parser.parse_args()
    root = Path("/proc") / str(args.pid)
    cmd = (root / "cmdline").read_bytes().split(b"\0")
    if (
        len(cmd) < 2
        or Path(os.fsdecode(cmd[0])).name != "bash"
        or Path(os.fsdecode(cmd[1])).name != "coverage_bench.sh"
    ):
        parser.error("PID is not a Bash coverage_bench.sh launcher")
    if args.action == "rerun":
        if not args.cases:
            parser.error("rerun requires --cases")
        if process(args.pid)[0] != "T" or children(args.pid):
            parser.error("Wait until coverage status is paused before rerunning a case")
        original = [os.fsdecode(value) for value in cmd if value]
        command = original[:2] + ["--mode", "run", "--cases", args.cases]
        for option in ("--profile", "--inference-mode"):
            if option in original:
                command += [option, original[original.index(option) + 1]]
        log_path = Path(os.readlink(root / "fd/1"))
        text = log_path.read_text(errors="replace")
        matches = re.findall(r"^WORKSPACE=(.+)$", text, re.MULTILINE)
        if not matches:
            parser.error("Cannot locate original coverage directory from launcher log")
        output = Path(matches[-1].strip()).parent
        if not (output / "cases.tsv").is_file():
            parser.error("Original coverage cases.tsv is missing")
        command += ["--output", str(output)]
        print(
            "Manual rerun reuses original coverage directory; prior attempts are retained.",
            flush=True,
        )
        raise SystemExit(subprocess.call(command))
    _, _, identity = process(args.pid)
    # pidfd prevents PID reuse from redirecting a control signal to another process.
    fd = os.pidfd_open(args.pid)
    try:
        if process(args.pid)[2] != identity:
            raise RuntimeError("Launcher changed while opening process handle")
        if args.action == "pause":
            signal.pidfd_send_signal(fd, signal.SIGSTOP)
        elif args.action == "resume":
            signal.pidfd_send_signal(fd, signal.SIGCONT)
        while True:
            state, _, _ = process(args.pid)
            active = children(args.pid)
            status = (
                "pause_pending"
                if state == "T" and active
                else "paused"
                if state == "T"
                else "running"
            )
            print(
                f"{status}: launcher={args.pid}; active_children={active}", flush=True
            )
            if not (args.action == "pause" and args.wait and active):
                break
            time.sleep(15)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
