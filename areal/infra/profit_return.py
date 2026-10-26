# SPDX-License-Identifier: Apache-2.0
"""Causal next-batch return decision; no device operations or model RNG use."""

import statistics

from areal.infra.drain_prediction import BucketService, simulate_decode
from areal.infra.rolling_drain import residual_distribution


def estimate_return(calibration, route, engines, needed, costs, margin=0.0):
    """Compare group completion with a resident actor and a delayed extra replica.

    Running fixed requests keep their placement. Only queued/withdrawn work can
    benefit from a return; no free migration or inverse-N speedup is assumed.
    The return cost is additional shared readiness AFTER fixed publication.
    """
    base = dict(needed_questions=needed, estimator="profit_return_v1")
    if needed <= 0:
        return dict(base, action="keep", reason="next_batch_ready", gain_seconds=0.0)
    if (
        not costs
        or not calibration.lengths
        or any(
            not d or first is None
            for d, first in zip(calibration.decode, calibration.first, strict=True)
        )
    ):
        return dict(base, action="return", reason="calibration_probe")
    publish = statistics.median(c["publish"] for c in costs)
    restore = statistics.median(c["restore"] for c in costs)
    live = [{r["request_id"]: r for r in e["active"]} for e in engines]
    leases = {r["request_id"]: dict(r, rank=-1) for r in route["waiting"]}
    leases.update({r["request_id"]: r for r in route["active"]})
    rows, groups = [], {}
    for rid, lease in leases.items():
        meta = lease.get("metrics") or {}
        if not all(
            k in meta
            for k in (
                "question_id",
                "generated_tokens",
                "prompt_tokens",
                "max_new_tokens",
            )
        ):
            return dict(base, action="return", reason="missing_metadata_probe")
        rank = lease["rank"]
        progress = live[rank].get(rid, {}) if rank >= 0 else {}
        generated = meta["generated_tokens"] + progress.get("tokens", 0)
        values, cdf = residual_distribution(
            generated,
            meta["max_new_tokens"],
            calibration.lengths,
            calibration.tail_rate,
        )
        residual = sum(
            v * (c - (cdf[i - 1] if i else 0.0))
            for i, (v, c) in enumerate(zip(values, cdf, strict=True))
        )
        idx = len(rows)
        rows.append(
            dict(
                rank=rank,
                context=meta["prompt_tokens"] + generated,
                draws=[residual],
                running=bool(progress.get("tokens", 0)),
            )
        )
        q = groups.setdefault(
            meta["question_id"], dict(size=lease["group_size"], ids=set(), rows=[])
        )
        q["ids"].add(rid)
        q["rows"].append(idx)
    known = [
        q["rows"]
        for qid, q in groups.items()
        if len(q["ids"] | set(route["completed"].get(qid, []))) == q["size"]
    ]
    if len(known) < needed:
        return dict(
            base,
            action="return",
            reason="incomplete_horizon_probe",
            known_questions=len(known),
        )
    service = BucketService(calibration.decode, calibration.service)
    widths = [
        max(r.get("max_concurrency", r["concurrency"]) for r in d.values())
        for d in calibration.decode
    ]

    def forecast(returning):
        ranks = list(calibration.fixed)
        if returning:
            ranks += sorted(calibration.borrowed)
        assignments = {rank: [] for rank in ranks}
        loads = {
            rank: publish if rank in calibration.borrowed else 0.0 for rank in ranks
        }
        pending = []
        for i, row in enumerate(rows):
            rank = row["rank"]
            if rank in calibration.fixed and row["running"]:
                assignments[rank].append(i)
                loads[rank] += (
                    row["draws"][0]
                    * service(rank, widths[rank], row["context"])
                    / widths[rank]
                )
            else:
                pending.append(i)
        for i in pending:
            row = rows[i]
            rank = min(ranks, key=lambda k: (loads[k], k))
            assignments[rank].append(i)
            loads[rank] += (
                row["draws"][0]
                * service(rank, widths[rank], row["context"])
                / widths[rank]
            )
        times = {}
        for rank, ids in assignments.items():
            if not ids:
                continue
            running = [
                i
                for i in ids
                if rows[i]["rank"] == rank
                and rank in calibration.fixed
                and rows[i]["running"]
            ]
            waiting = [i for i in ids if i not in running]
            result = simulate_decode(
                rows,
                ids,
                running,
                waiting,
                max(len(running), min(len(ids), widths[rank])),
                rank,
                False,
                0,
                0.0,
                calibration.first[rank]["unit"],
                service,
            )
            delay = publish if rank in calibration.borrowed else 0.0
            times.update({i: t + delay for i, t in result.items()})
        data_ready = sorted(max(times[i] for i in q) for q in known)[needed - 1]
        # A returned actor must be restored even when all data was already ready.
        training_ready = max(data_ready, publish) + restore if returning else data_ready
        return data_ready, training_ready

    keep_data, keep = forecast(False)
    return_data, switch = forecast(True)
    gain = keep - switch
    return dict(
        base,
        action="return" if gain > margin else "keep",
        reason="positive_gain" if gain > margin else "switch_cost_exceeds_gain",
        keep_seconds=keep,
        return_seconds=switch,
        gain_seconds=gain,
        keep_data_seconds=keep_data,
        return_data_seconds=return_data,
        publish_seconds=publish,
        restore_seconds=restore,
        margin_seconds=margin,
        window_count=len(calibration.frames),
        known_questions=len(known),
        cost_samples=len(costs),
        length_samples=len(calibration.lengths),
    )
