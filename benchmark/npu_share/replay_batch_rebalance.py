#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay saved causal snapshots with measured KV transport and explicit unknowns.

This produces counterfactual predictions only. It neither moves requests nor
claims observed speedup. Probe packing layout is not the vLLM block layout.
"""

import argparse
import json
import statistics
from pathlib import Path

from areal.infra.batch_rebalance import KVTransferCost, plan_batch_rebalance
from areal.infra.rolling_drain import RollingBatchCalibration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", type=Path)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overhead-seconds", type=float, default=0.02)
    p.add_argument("--block-tokens", type=int, default=64)
    args = p.parse_args()
    left = json.loads((args.probe / "rank-0.json").read_text())
    right = json.loads((args.probe / "rank-1.json").read_text())
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Require matching probe rows at two or more sizes")
    measurements = []
    for a, b in zip(left, right):
        if (
            a["computed_tokens"] != b["computed_tokens"]
            or a["payload_bytes"] != b["payload_bytes"]
            or len(a["measurements"]) != len(b["measurements"])
            or len(a["measurements"]) < 2
        ):
            raise ValueError("Mismatched probe metadata or repeat count")
        if not (
            a["kv_exact"]
            and b["kv_exact"]
            and a["generated"] == b["generated"]
            and a["replay_pending_token_equal"]
            and b["replay_pending_token_equal"]
        ):
            raise RuntimeError("KV probe correctness failed")
        durations = [
            max(x["total_s"], y["total_s"])
            for x, y in zip(a["measurements"][1:], b["measurements"][1:])
        ]
        measurements.append(
            dict(
                bytes=a["payload_bytes"],
                tokens=a["computed_tokens"],
                total_s=statistics.median(durations),
            )
        )
    measurements.sort(key=lambda row: row["bytes"])
    if len({row["bytes"] for row in measurements}) < 2:
        raise ValueError("At least two distinct payload sizes are required")
    # Affine envelope over measured medians, not a guaranteed interpolation.
    # Do not extrapolate above probe size.
    slope = max(
        0,
        (measurements[-1]["total_s"] - measurements[0]["total_s"])
        / (measurements[-1]["bytes"] - measurements[0]["bytes"]),
    )
    intercept = max(x["total_s"] - slope * x["bytes"] for x in measurements)
    bpt = measurements[0]["bytes"] // measurements[0]["tokens"]
    events = sorted(
        [
            json.loads(line)
            for f in args.arm.rglob("events-*.jsonl")
            for line in f.read_text().splitlines()
            if line.strip()
        ],
        key=lambda r: r["time"],
    )
    calibration = RollingBatchCalibration()
    results = []
    for row in events:
        if row["event"] == "batch_drain_calibration_input":
            calibration.freeze(row["engines"], row["route"], row["handoffs"])
        elif row["event"] == "running_rebalance_shadow":
            cases = []
            # Full scheduling ownership latency is unmeasured: show sensitivity.
            for ownership in (None, 0.0, 0.05, 0.5, 1.0, 2.0):
                cost = (
                    None
                    if ownership is None
                    else KVTransferCost(
                        bpt,
                        args.block_tokens,
                        intercept,
                        slope,
                        max(x["bytes"] for x in measurements),
                        ownership,
                    )
                )
                result = plan_batch_rebalance(
                    calibration,
                    row["route"],
                    row["engines"],
                    args.overhead_seconds,
                    transport=cost,
                )
                cases.append(dict(ownership_s=ownership, plan=result))
            results.append(
                dict(time=row["time"], window=calibration.window, cases=cases)
            )
    report = dict(
        probe_measurements=measurements,
        transport_intercept_s=intercept,
        transport_seconds_per_byte=slope,
        overhead_assumption_s=args.overhead_seconds,
        layout_assumption=dict(bytes_per_token=bpt, block_tokens=args.block_tokens),
        caveat="Offline prediction only; vLLM freeze/import overhead and block layout not measured. No scheduler execution is authorized.",
        decisions=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for d in results:
        print("WINDOW", d["window"])
        for c in d["cases"]:
            plan = c["plan"]
            print(
                json.dumps(
                    dict(
                        ownership_s=c["ownership_s"],
                        reason=plan["reason"],
                        moved=len(plan["selected"]),
                        kv=sum(x["method"] == "kv" for x in plan["selected"]),
                        baseline=plan.get("baseline_tail_s"),
                        predicted=plan.get("predicted_tail_s"),
                    )
                )
            )


if __name__ == "__main__":
    main()
