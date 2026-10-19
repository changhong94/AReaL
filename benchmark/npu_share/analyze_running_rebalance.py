#!/usr/bin/env python3
"""Summarize R001 decisions and audit ownership from one benchmark arm."""

import argparse
import collections
import json
from pathlib import Path


def analyze(arm):
    events = []
    for path in arm.rglob("events-*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))
    events.sort(key=lambda row: row["time"])
    reasons = collections.Counter()
    active, versions, violations, moves, forecasts = {}, {}, [], [], []
    for row in events:
        event = row["event"]
        rid = row.get("request_id")
        if event == "running_rebalance_shadow":
            reasons[row["reason"]] += 1
            forecasts.append(
                {k: v for k, v in row.items() if k not in {"route", "engines"}}
            )
        elif event == "queue_rebalance_result" and row.get("running"):
            moves.append(row)
        elif event == "route_acquire":
            if rid in active:
                violations.append(dict(kind="overlapping_owners", request_id=rid))
            if (
                row.get("running_handoff")
                and rid in versions
                and versions[rid] != row["version"]
            ):
                violations.append(dict(kind="handoff_version_changed", request_id=rid))
            active[rid] = row["lease_id"]
            versions[rid] = row["version"]
        elif event == "route_release":
            if active.get(rid) == row["lease_id"]:
                active.pop(rid)
    completed = {r["request_id"]: r for r in events if r["event"] == "answer_end"}
    for forecast in forecasts:
        for row in forecast.get("predictions", []):
            end = completed.get(row["request_id"])
            if end and end["time"] >= forecast["time"]:
                row["observed_remaining_s"] = end["time"] - forecast["time"]
                row["observed_output_tokens"] = end["output_tokens"]
    return dict(
        arm=str(arm),
        trace_event_count=len(events),
        reasons=dict(reasons),
        forecasts=forecasts,
        acknowledged_running_moves=sum(r["withdrawn"] for r in moves),
        moves=moves,
        ownership_violations=violations,
        outstanding_leases=len(active),
        caveat="Trace audit checks ownership and handoff epochs; it does not prove identical sampled answers or causal speedup.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", type=Path)
    args = parser.parse_args()
    result = analyze(args.arm)
    target = args.arm / "r001-analysis.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(target)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "reasons",
                    "acknowledged_running_moves",
                    "ownership_violations",
                    "outstanding_leases",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
