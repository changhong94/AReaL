"""Maintain the machine-local CSV ledger of formal benchmark workspaces."""

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
import re
import socket
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

OLD_FIELDS = "date case machine status run_tag workspace fixed_seconds colocated_seconds shared_seconds reduction_percent error".split()
PREVIOUS_FIELDS = [k for k in OLD_FIELDS if k not in {"error", "workspace"}] + [
    "workspace"
]
FIELDS = PREVIOUS_FIELDS[:-1] + ["invalidated", "workspace"]
RUN = re.compile(r"^(\d{8}-\d{6}[+-]\d{4})_(math|code)_.*_c\d+_b\d+_[0-9a-f]+$")
CASE = re.compile(r"[A-Z]\d{2}")


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def cases(parent):
    found = {}
    for path in list(parent.glob("*.log")) + list(parent.glob("attempts/*/*/*.log")):
        case = path.stem if CASE.fullmatch(path.stem) else path.parent.parent.name
        if CASE.fullmatch(case):
            for workspace in re.findall(
                r"^WORKSPACE=(.+)$", path.read_text(errors="replace"), re.M
            ):
                found[Path(workspace.strip()).name] = case
    return found


def workspaces(root):
    for folder, dirs, _ in os.walk(root):
        p = Path(folder)
        if RUN.fullmatch(p.name):
            dirs.clear()
            yield p
        else:
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".") and d not in {"attempts", "repetitions"}
            ]


def duration(folder):
    f = folder / "phases.jsonl"
    if f.is_file():
        times = {}
        for line in f.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") in ("train.begin", "train.end"):
                times[event["event"]] = event["time"]
        if "train.begin" in times and "train.end" in times:
            return times["train.end"] - times["train.begin"]


def row(w, machine, case=None):
    if (
        (w / "session-state.json").exists()
        or (w / "repetitions").exists()
        or (w / "NOT_REAL_BENCHMARK.txt").exists()
    ):
        return None
    settings = read(w / "settings.json")
    state = read(w / "state.json")
    summary = read(w / "summary.json")
    if state.get("phase") == "prepared" or not (state or summary):
        return None
    tag = settings.get("run_tag")
    date = None
    if tag:
        date = dt.datetime.strptime(tag.split("_")[0], "%Y%m%d").date()
    elif match := RUN.fullmatch(w.name):
        try:
            date = (
                dt.datetime.strptime(match[1], "%Y%m%d-%H%M%S%z")
                .astimezone(ZoneInfo("America/Creston"))
                .date()
            )
        except ValueError:
            pass
    arms = read(w / "arm-status.json")
    errors = [
        str(state[k])
        for k in ("error", "cleanup_error", "resource_hook_error")
        if state.get(k)
    ]
    errors.extend(
        f"{a}: {v['reason']}"
        for a, v in arms.items()
        if v.get("status") == "failed" and v.get("reason")
    )
    status = state.get("phase")
    if status == "complete" and any(v.get("status") == "failed" for v in arms.values()):
        status = "partial"
    times = dict(summary.get("loop_seconds") or {})
    for a in ("fixed", "colocated", "latest"):
        if times.get(a) is None and (
            a in state.get("completed", [])
            or arms.get(a, {}).get("status") == "complete"
        ):
            times[a] = duration(w / a)
    fixed, shared = times.get("fixed"), times.get("latest")
    gain = (
        100 * (fixed - shared) / fixed
        if status == "complete" and fixed and shared is not None
        else None
    )
    return dict(
        date=date,
        case=settings.get("coverage_case") or case,
        machine=machine,
        status=status,
        run_tag=tag,
        workspace=str(w),
        fixed_seconds=fixed,
        colocated_seconds=times.get("colocated"),
        shared_seconds=shared,
        reduction_percent=gain,
        error=" | ".join(errors),
    )


def update(root, paths=None):
    root = root.resolve()
    label = root / ".result-machine"
    machine = label.read_text().strip() if label.exists() else socket.gethostname()
    dst = root / "result.csv"
    with (root / ".result.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = {}
        if dst.exists():
            with dst.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames not in (FIELDS, PREVIOUS_FIELDS, OLD_FIELDS):
                    raise ValueError(f"Unexpected ledger header: {dst}")
                records = {
                    r["workspace"]: {k: r.get(k, "1") for k in FIELDS}
                    for r in reader
                    if not r.get("error") and r["status"] not in {"failed", "partial"}
                }
        mapping = {}
        for w in paths if paths is not None else workspaces(root):
            if w.parent not in mapping:
                mapping[w.parent] = cases(w.parent)
            r = row(
                w,
                machine,
                mapping[w.parent].get(w.name)
                or next((p.name for p in w.parents if CASE.fullmatch(p.name)), None),
            )
            if r is not None:
                if r.pop("error") or r["status"] in {"failed", "partial"}:
                    records.pop(str(w), None)
                else:
                    r["invalidated"] = records.get(str(w), {}).get("invalidated", "0")
                    records[str(w)] = r
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", dir=root, delete=False
        ) as tmp:
            writer = csv.DictWriter(tmp, FIELDS)
            writer.writeheader()
            writer.writerows(
                sorted(
                    records.values(),
                    key=lambda r: (
                        str(r["date"] or ""),
                        Path(r["workspace"]).name,
                        r["workspace"],
                    ),
                )
            )
        os.replace(tmp.name, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    update(parser.parse_args().root)
