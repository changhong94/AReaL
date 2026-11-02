"""Inspect, invalidate or restore a host-local R031 revision (never edits old samples)."""

import argparse
import fcntl
import json
import re
from pathlib import Path

from templates.local_cost_store import Store, adjust_layout_probe, host_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=["list", "history", "show", "invalidate", "rollback", "adjust-probe"],
    )
    p.add_argument("--kind", choices=["training-layout", "inference-return"])
    p.add_argument("--key")
    p.add_argument("--revision")
    p.add_argument("--reason")
    p.add_argument("--layout")
    p.add_argument("--field", choices=["compute_seconds", "entry", "publication"])
    p.add_argument("--seconds", type=float)
    a = p.parse_args()
    host = host_identity()
    root = Path.home() / ".cache/areal-deployment-calibration" / host
    if a.action == "list":
        for f in sorted(root.glob("*/*.json")):
            print(f.parent.name, f.stem, f.read_text())
        return
    if not a.kind or not re.fullmatch("[a-f0-9]{64}", a.key or ""):
        p.error("--kind and --key required")
    pointer = root / a.kind / (a.key + ".json")
    current = json.loads(pointer.read_text())["revision"]
    if not re.fullmatch("[a-f0-9]{32}", current):
        p.error("invalid pointer")
    ctx = json.loads(
        (pointer.with_suffix(".history") / (current + ".json")).read_text()
    )["context"]
    s = Store(ctx)
    if s.path != pointer:
        raise ValueError("Host/kind/key mismatch")
    if a.action == "history":
        while current:
            row = s.revision(current)
            print(
                current,
                row["created_utc"],
                row["source"],
                row["details"].get("reason", ""),
            )
            current = row["previous"]
    elif a.action == "show":
        print(
            json.dumps(s.revision(a.revision or current), indent=2, ensure_ascii=False)
        )
    else:
        if not a.reason:
            p.error("--reason is required")
        lock = Path.home() / ".cache/areal-benchmark.lock"
        with lock.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if a.action == "adjust-probe":
                print(adjust_layout_probe(s, a.layout, a.field, a.seconds, a.reason))
                return
            old = (
                s.revision(a.revision)
                if a.action == "rollback" and a.revision
                else None
            )
            if a.action == "rollback" and (not old or old["data"] is None):
                p.error("select a valid --revision")
            details = dict(
                reason=a.reason,
                restored_revision=a.revision if old else None,
                origins=old["details"].get("origins", {}) if old else {},
            )
            print(s.put(old["data"] if old else None, source=a.action, details=details))


if __name__ == "__main__":
    main()
