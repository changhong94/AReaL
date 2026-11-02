# SPDX-License-Identifier: Apache-2.0
"""Predict the next usable training boundary from deployment and live observations."""

import math


def predict_layout(
    current,
    compute,
    entry,
    publication,
    missing,
    rate,
    leases,
    lease_seconds,
    terminal,
    *,
    baseline,
    retains_inference,
    progress=None,
):
    if not terminal and missing and (rate is None or rate <= 0):
        data = None
    else:
        data = missing / rate if missing and not terminal else 0.0
    costs = {}
    fraction = 1.0 if progress is None else progress["unfinished_fraction"]
    if progress is not None and progress.get("drain_seconds") is not None:
        lease_seconds = progress["drain_seconds"]
    for w in compute:
        training = (
            None if entry[w] is None or compute[w] is None else entry[w] + compute[w]
        )
        # DP1 leaves a generating replica, DP2 suspends all generation.
        drain = 0.0 if terminal or retains_inference[w] or not leases else lease_seconds
        publishing = 0.0 if terminal else publication[w]
        if training is None or publishing is None or drain is None or data is None:
            horizon = None
        else:
            actor_ready = training + drain + publishing
            # Old leases resume before publication; their completion can also
            # contribute to next-step data. Do not add that interval twice.
            data_ready = (
                data
                if retains_inference[w] or not data
                else max(
                    data,
                    training
                    + data * fraction
                    + (progress or {}).get("score_tail", 0.0),
                )
            )
            horizon = max(actor_ready, data_ready)
        costs[w] = dict(
            compute=compute[w],
            entry=entry[w],
            publication=publishing,
            drain=drain,
            total=horizon,
        )
    # Unknown durations have a non-negative lower bound, never a free actual cost.
    lower = {
        w: sum(
            x or 0.0 for x in (c["entry"], c["compute"], c["publication"], c["drain"])
        )
        for w, c in costs.items()
    }
    complete = [
        w
        for w, c in costs.items()
        if c["total"] is not None and math.isfinite(c["total"])
    ]
    best = (
        min(complete, key=lambda w: (costs[w]["total"], w != current))
        if complete
        else None
    )
    proven = best is not None and all(
        w in complete or lower[w] > costs[best]["total"] for w in costs
    )
    reason = (
        ("next_boundary" if len(complete) == len(costs) else "dominates_lower_bound")
        if proven
        else "insufficient_current_run_evidence"
    )
    if not proven:
        best = current
        if (
            not terminal
            and leases
            and missing
            and retains_inference.get(baseline)
            and baseline in complete
        ):
            best = baseline
            reason = "protect_live_inference_unpriced_alternative"
    return dict(
        world_size=best,
        current_world=current,
        candidates=costs,
        missing_questions=missing,
        question_seconds=data,
        lease_count=leases,
        terminal=terminal,
        lower_bounds=lower,
        generation_progress=progress,
        reason=reason,
    )
