# SPDX-License-Identifier: Apache-2.0
"""One-shot, calibrated queue stealing after the shared replica returns."""

import math

from areal.infra.drain_prediction import simulate_decode
from areal.infra.rolling_drain import residual_distribution


def plan_rebalance(
    calibration,
    route,
    engines,
    overhead,
    source_rank=1,
    target_rank=0,
    *,
    running=False,
):
    """Compare expected completion of the live set; never select running work.

    Estimates use the already frozen rolling window, not future completions.
    Public waiting tickets take precedence over stealing native queued work.
    """
    result = dict(selected=[], reason="no_queued_work", predictions=[])
    candidates = set(
        engines[source_rank]["queue"].get("running" if running else "queued", [])
    )
    if running:
        result["reason"] = "no_running_work"
    if not candidates:
        return result
    if route["waiting"]:
        return dict(result, reason="public_queue_first")
    if (
        calibration is None
        or not calibration.lengths
        or any(x is None for x in calibration.first)
        or any(not x for x in calibration.decode)
    ):
        return dict(result, reason="uncalibrated")
    progress = [{r["request_id"]: r for r in e["active"]} for e in engines]
    rows = []
    for lease in route["active"]:
        meta = lease.get("metrics", {})
        if not all(
            k in meta for k in ("generated_tokens", "max_new_tokens", "prompt_tokens")
        ):
            return dict(result, reason="missing_metadata")
        rank = lease["rank"]
        live = progress[rank].get(lease["request_id"])
        if live is None:
            return dict(result, reason="snapshot_race")
        generated = meta["generated_tokens"] + live["tokens"]
        values, cdf = residual_distribution(
            generated,
            meta["max_new_tokens"],
            calibration.lengths,
            calibration.tail_rate,
        )
        mean = sum(
            v * (cdf[i] - (cdf[i - 1] if i else 0)) for i, v in enumerate(values)
        )
        if mean <= 0:
            return dict(result, reason="finishing_request")
        queue = engines[rank]["queue"]
        rows.append(
            dict(
                lease,
                context=meta["prompt_tokens"] + generated,
                draws=[mean],
                started=lease["request_id"]
                not in queue.get("waiting", queue["queued"]),
            )
        )
    capacity = min(route["capacity"], engines[target_rank]["queue"]["width"])
    room = max(
        0,
        capacity
        - sum(r["rank"] == target_rank for r in rows)
        - route.get("reserved", 0),
    )
    if not room:
        return dict(result, reason="target_full")

    def service(rank, n, context):
        sample = calibration.service(rank, n, context)
        if (
            sample is None
            or not math.isfinite(sample["seconds"])
            or sample["seconds"] <= 0
        ):
            raise ValueError("Invalid decode calibration")
        return sample["seconds"]

    migration = max(overhead, (calibration.migration or {}).get("seconds", overhead))

    def forecast(moved):
        output = {}
        for rank in range(len(engines)):
            indices = [
                i
                for i, r in enumerate(rows)
                if (target_rank if i in moved else r["rank"]) == rank
            ]
            running = [i for i in indices if rows[i]["started"] and i not in moved]
            waiting = [i for i in indices if i not in running]
            native_order = {
                rid: j
                for j, rid in enumerate(
                    engines[rank]["queue"].get(
                        "waiting", engines[rank]["queue"]["queued"]
                    )
                )
            }
            waiting.sort(
                key=lambda i: native_order.get(
                    rows[i]["request_id"], len(native_order) + i
                )
            )
            output.update(
                simulate_decode(
                    rows,
                    indices,
                    running,
                    waiting,
                    max(
                        len(running),
                        min(route["capacity"], engines[rank]["queue"]["width"]),
                    ),
                    rank,
                    True,
                    0,
                    migration,
                    calibration.first[rank]["unit"],
                    service,
                )
            )
        return output

    moved = set()
    current = forecast(moved)
    # Re-evaluate total completion after each proposed move. Do not pursue
    # equal request counts at the expense of the critical tail.
    for _ in range(min(room, 1) if running else room):
        choices = []
        for i, r in enumerate(rows):
            if (
                i in moved
                or r["rank"] != source_rank
                or r["request_id"] not in candidates
                or (running and r["request_id"] in route.get("running_moved", []))
            ):
                continue
            predicted = forecast(moved | {i})
            gain = max(current.values(), default=0) - max(predicted.values(), default=0)
            own_gain = current[i] - predicted[i]
            if gain > overhead and own_gain > overhead:
                choices.append((gain, own_gain, i, predicted))
        if not choices:
            break
        gain, own_gain, i, predicted = max(choices, key=lambda x: (x[0], x[1]))
        result["predictions"].append(
            dict(
                request_id=rows[i]["request_id"],
                replay_tokens=rows[i]["context"],
                replay_s=rows[i]["context"] * calibration.first[target_rank]["unit"],
                remaining_tokens=rows[i]["draws"][0],
                stay_s=current[i],
                move_s=predicted[i],
                tail_gain_s=gain,
                overhead_s=overhead,
                transfer_s=migration,
            )
        )
        moved.add(i)
        current = predicted
    result["selected"] = [rows[i]["request_id"] for i in sorted(moved)]
    result["reason"] = "predicted_gain" if moved else "no_predicted_gain"
    return result
