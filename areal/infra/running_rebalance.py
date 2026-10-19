# SPDX-License-Identifier: Apache-2.0
"""Read-only R001 forecasts; no source abort or destination reservation.

The estimator uses the frozen causal window and includes prefix replay.
Predicted savings are hypotheses, not measured counterfactual outcomes.
"""

import math

from areal.infra.queue_rebalance import plan_rebalance


def plan_running_shadow(
    calibration, route, engines, overhead, source_rank, target_rank
):
    result = dict(selected=[], predictions=[], reason="invalid_snapshot", mode="shadow")
    if not math.isfinite(overhead) or overhead < 0:
        return result
    if not (0 <= source_rank < len(engines) and 0 <= target_rank < len(engines)):
        return result
    if source_rank == target_rank:
        return result
    if not engines[target_rank].get("generation_running"):
        return dict(result, reason="target_transitioning")
    if not engines[target_rank].get("budget_valid"):
        return dict(result, reason="inference_budget_not_admitted")
    # A mixed-version snapshot cannot support a same-version handoff forecast.
    versions = route.get("versions", [])
    ready = route.get("ready", [])
    if len(versions) != len(engines) or len(ready) != len(engines):
        return dict(result, reason="missing_epoch")
    version = versions[target_rank]
    if (
        not ready[source_rank]
        or not ready[target_rank]
        or versions[source_rank] != version
    ):
        return dict(result, reason="incompatible_epoch")
    if any(r.get("version") != versions[r["rank"]] for r in route["active"]):
        return dict(result, reason="stale_lease")
    if engines[source_rank]["queue"].get("queued"):
        return dict(result, reason="queued_work_first")
    if route.get("task_affinity", False):
        return dict(result, reason="task_affinity")
    try:
        prediction = plan_rebalance(
            calibration,
            route,
            engines,
            overhead,
            source_rank,
            target_rank,
            running=True,
        )
    except (ValueError, KeyError, IndexError, TypeError, OverflowError) as exc:
        return dict(result, reason="invalid_calibration", error=type(exc).__name__)
    if any(
        not math.isfinite(v)
        for row in prediction["predictions"]
        for v in row.values()
        if isinstance(v, (int, float))
    ):
        return dict(result, reason="nonfinite_prediction")
    # Reserve enough logical KV space for the entire continuation. The
    # scheduler capacity already reflects the minimum admitted TP rank budget.
    headroom = engines[target_rank]["queue"].get("free_kv_tokens")
    leases = {r["request_id"]: r for r in route["active"]}
    required = max(
        (
            leases[r]["metrics"]["prompt_tokens"]
            + leases[r]["metrics"]["max_new_tokens"]
            for r in prediction["selected"]
        ),
        default=0,
    )
    if required and (headroom is None or headroom < required):
        return dict(
            result,
            reason="insufficient_kv_headroom",
            required_tokens=required,
            free_kv_tokens=headroom,
        )
    return dict(
        prediction,
        mode="shadow",
        version=version,
        source_rank=source_rank,
        target_rank=target_rank,
        execution_path="abort_then_prefix_replay",
        uncertainty="training_contention_and_handoff_cost_not_yet_validated",
    )
