# SPDX-License-Identifier: Apache-2.0
"""Snapshot-only replay/KV planner; transport is dispatched by kv_rebalance.

Offline predictions remain counterfactual. Live execution separately validates
request identity, version, memory ownership, and measured handoff completion.
"""

import math
import time
from dataclasses import dataclass

from areal.infra.drain_prediction import simulate_decode
from areal.infra.rolling_drain import residual_distribution


@dataclass(frozen=True)
class KVTransferCost:
    bytes_per_token: int
    block_tokens: int
    setup_s: float
    seconds_per_byte: float
    max_measured_bytes: int
    ownership_s: float

    def __post_init__(self):
        if min(self.bytes_per_token, self.block_tokens, self.max_measured_bytes) <= 0:
            raise ValueError("KV sizes must be positive")
        if any(
            not math.isfinite(x) or x < 0
            for x in (self.setup_s, self.seconds_per_byte, self.ownership_s)
        ):
            raise ValueError("KV costs must be finite and nonnegative")

    def estimate(self, computed):
        if computed <= 0:
            return None
        size = (
            math.ceil(computed / self.block_tokens)
            * self.block_tokens
            * self.bytes_per_token
        )
        if size > self.max_measured_bytes:
            return None  # Do not extrapolate a tiny transport probe silently.
        return size, self.setup_s + size * self.seconds_per_byte + self.ownership_s


def plan_batch_rebalance(
    calibration,
    route,
    engines,
    overhead,
    *,
    transport=None,
    source=1,
    target=0,
    max_moves=None,
    request_local=False,
    kv_only=False,
    aggressive=None,
):
    """Scan one immutable snapshot, then evaluate bounded candidate subsets.

    Objectives are predicted makespan and total completion time, not equal
    request counts. All transport/ownership figures are explicit hypotheses.
    """
    planning_started = time.perf_counter()
    result = dict(
        selected=[],
        reason="invalid_snapshot",
        candidates=[],
        scenarios=[],
        pending_requests=[],
        completed_requests=[],
        execution_ready=False,
        scope="offline_counterfactual",
    )
    if not math.isfinite(overhead) or overhead < 0:
        return result
    if source == target or not (
        0 <= source < len(engines) and 0 <= target < len(engines)
    ):
        return result
    if (
        len(route.get("ready", [])) != len(engines)
        or not all(route["ready"])
        or len(route.get("versions", [])) != len(engines)
    ):
        return result
    if route["versions"][source] != route["versions"][target]:
        return dict(result, reason="incompatible_epoch")
    if not engines[target].get("budget_valid") or not engines[target].get(
        "generation_running"
    ):
        return dict(result, reason="target_unavailable")
    if route.get("waiting") or route.get("task_affinity"):
        return dict(result, reason="admission_priority")
    if (
        calibration is None
        or not calibration.lengths
        or any(x is None for x in calibration.first)
        or any(not x for x in calibration.decode)
    ):
        return dict(result, reason="uncalibrated")
    progress = [{r["request_id"]: r for r in e["active"]} for e in engines]
    rows, options, replay_options = [], {}, {}
    queues = engines[source]["queue"]
    candidates = set(queues.get("running", [])) | set(queues.get("queued", []))
    for lease in route["active"]:
        rank, rid, meta = lease["rank"], lease["request_id"], lease["metrics"]
        if lease["version"] != route["versions"][rank]:
            return dict(result, reason="stale_lease")
        live = progress[rank].get(rid)
        pending = live is None
        if pending:
            # A routed continuation carries the original prompt size plus its
            # retained generated prefix. Match that exact attempt length: an old
            # aborted attempt with a shorter prompt is not a completed lease.
            current_prompt = meta["prompt_tokens"] + meta["generated_tokens"]
            finished = any(
                h["request_id"] == rid
                and h["prompt_tokens"] == current_prompt
                and h["tokens"] > 0
                for h in engines[rank].get("history", [])
            )
            if finished:
                result["completed_requests"].append(rid)
                continue
            # Router admission precedes HTTP/native admission. Account for this
            # work as waiting, reserve its destination capacity, and never move it.
            live = dict(tokens=0)
            result["pending_requests"].append(rid)
        native = engines[rank]["queue"]
        if (
            not pending
            and rid not in native["running"]
            and rid not in native.get("waiting", native["queued"])
        ):
            return dict(result, reason="native_snapshot_race")
        generated = meta["generated_tokens"] + live["tokens"]
        values, cdf = residual_distribution(
            generated,
            meta["max_new_tokens"],
            calibration.lengths,
            calibration.tail_rate,
        )
        remaining = sum(
            v * (cdf[i] - (cdf[i - 1] if i else 0)) for i, v in enumerate(values)
        )
        if not math.isfinite(remaining) or remaining <= 0:
            return dict(result, reason="finishing_request")
        context = meta["prompt_tokens"] + generated
        index = len(rows)
        rows.append(
            dict(
                lease,
                context=context,
                draws=[remaining],
                started=not pending and rid in native["running"],
            )
        )
        if (
            pending
            or rank != source
            or rid not in candidates
            or rid in route.get("running_moved", [])
        ):
            continue
        replay = context * calibration.first[target]["unit"]
        choices = [
            dict(
                method="replay",
                bytes=0,
                setup_s=overhead,
                prefill_s=replay,
                total_setup_s=overhead + replay,
            )
        ]
        # Live computed-token counts are absent in old snapshots: a hypothetical
        # upper bound is labeled, never used to authorize an actual transfer.
        computed = live.get("computed_tokens", max(0, context - 1))
        kv = (
            transport.estimate(computed)
            if transport and rid in queues.get("running", []) and 0 < computed < context
            else None
        )
        if kv:
            size, cost = kv
            choices.append(
                dict(
                    method="kv",
                    bytes=size,
                    setup_s=(0.0 if aggressive else overhead) + cost,
                    prefill_s=(context - computed) * calibration.first[target]["unit"],
                    total_setup_s=(0.0 if aggressive else overhead)
                    + cost
                    + (context - computed) * calibration.first[target]["unit"],
                )
            )
        if kv_only:
            choices = [x for x in choices if x["method"] == "kv"]
            if not choices or "computed_tokens" not in live:
                continue
        chosen = min(choices, key=lambda x: x["total_setup_s"])
        options[index] = chosen
        replay_options[index] = choices[0]
        result["candidates"].append(
            dict(
                request_id=rid,
                context_tokens=context,
                expected_remaining=remaining,
                alternatives=choices,
                chosen=chosen["method"],
                computed_tokens_exact="computed_tokens" in live,
                computed_tokens_used=computed,
            )
        )
    room = (
        min(route["capacity"], engines[target]["queue"]["width"])
        - sum(r["rank"] == target for r in route["active"])
        - route.get("reserved", 0)
    )
    if max_moves is not None:
        room = min(room, max_moves)
    free_tokens = engines[target]["queue"].get("free_kv_tokens")
    if room <= 0 or free_tokens is None:
        return dict(result, reason="target_full")

    def continuation_capacity(row):
        tokens = row["metrics"]["prompt_tokens"] + row["metrics"]["max_new_tokens"]
        if transport is not None:
            tokens = math.ceil(tokens / transport.block_tokens) * transport.block_tokens
        return tokens

    free_tokens -= sum(
        continuation_capacity(r)
        for r in rows
        if r["rank"] == target and r["request_id"] in result["pending_requests"]
    )

    groups = {}
    if aggressive:
        for i, row in enumerate(rows):
            qid = row["metrics"].get("question_id")
            if qid is None:
                continue
            group = groups.setdefault(
                qid, dict(ids=set(), indices=[], size=row.get("group_size", 0))
            )
            group["ids"].add(row["request_id"])
            group["indices"].append(i)
        completed = set(result["completed_requests"])
        known = [
            g["indices"]
            for qid, g in groups.items()
            if len(
                g["ids"]
                | set(route.get("completed", {}).get(qid, []))
                | {
                    r["request_id"]
                    for r in route["active"]
                    if r["request_id"] in completed
                    and r["metrics"].get("question_id") == qid
                }
            )
            == g["size"]
        ]
        needed = aggressive["needed_questions"]
        result["objective"] = (
            "batch_ready" if needed > 0 and len(known) >= needed else "mean_completion"
        )
        result["known_questions"] = len(known)
        result["needed_questions"] = needed

    def objective(output):
        if not aggressive:
            return max(output.values(), default=0)
        if result["objective"] == "batch_ready":
            return sorted(max(output[i] for i in group) for group in known)[needed - 1]
        return sum(output.values()) / max(1, len(output))

    def service(rank, n, context):
        sample = calibration.service(rank, n, context)
        value = sample["seconds"]
        # Borrow only the measured concurrency slope, not the other card's
        # absolute speed. This extrapolation is opt-in and logged with the plan.
        if aggressive and rank == target and sample.get("concurrency", n) > n:
            peer_low = calibration.service(source, n, context)
            peer_high = calibration.service(source, sample["concurrency"], context)
            if peer_low.get("concurrency", n) <= n and peer_high["seconds"] > 0:
                ratio = min(1.0, max(0.8, peer_low["seconds"] / peer_high["seconds"]))
                value *= ratio
                result["borrowed_service_slope"] = True
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid service calibration")
        return value

    def forecast(selected, variant="mixed"):
        costs = replay_options if variant == "replay" else options
        moved = set(selected)
        updated = [dict(r) for r in rows]
        # Initial protocol assumption: freeze both engines for a serialized KV
        # batch, then resume decode and replay. Charge stalls to unmoved work.
        pause = (
            0.0
            if request_local or aggressive
            else sum(
                costs[i]["setup_s"] for i in selected if costs[i]["method"] == "kv"
            )
        )
        barrier = (
            (aggressive["coordination_s"] + sum(costs[i]["setup_s"] for i in selected))
            if aggressive and selected
            else 0.0
        )
        serial = 0.0
        for i in selected:
            if costs[i]["method"] == "replay" or request_local:
                serial += costs[i]["setup_s"]
            updated[i].update(
                handoff_ready_s=barrier
                if aggressive
                else serial
                if costs[i]["method"] == "replay" or request_local
                else 0.0,
                handoff_prefill_s=costs[i]["prefill_s"],
            )
        if aggressive:
            for i in selected:
                updated[i]["handoff_ready_s"] = barrier
        output = {}
        for rank, engine in enumerate(engines):
            ids = [
                i
                for i, r in enumerate(rows)
                if (target if i in moved else r["rank"]) == rank
            ]
            active = [i for i in ids if rows[i]["started"] and i not in moved]
            waiting = [i for i in ids if i not in active]
            order = {
                rid: j
                for j, rid in enumerate(
                    engine["queue"].get("waiting", engine["queue"]["queued"])
                )
            }
            waiting.sort(key=lambda i: order.get(rows[i]["request_id"], len(order) + i))
            output.update(
                simulate_decode(
                    updated,
                    ids,
                    active,
                    waiting,
                    max(len(active), min(route["capacity"], engine["queue"]["width"])),
                    rank,
                    True,
                    0,
                    overhead,
                    calibration.first[rank]["unit"],
                    service,
                )
            )
        if aggressive and selected:
            # Selected requests wait for the common staging barrier. Other
            # requests only pay measured worker stalls, not HTTP coordination.
            for i in output:
                if i not in moved and rows[i]["rank"] in (source, target):
                    key = (
                        "source_stall_s"
                        if rows[i]["rank"] == source
                        else "target_stall_s"
                    )
                    output[i] += len(selected) * aggressive[key]
        return {
            i: t
            + (
                pause
                if (target if i in moved else rows[i]["rank"]) in (source, target)
                else 0
            )
            for i, t in output.items()
        }

    baseline = forecast([])
    tail, total = max(baseline.values(), default=0), sum(baseline.values())
    result.update(baseline_tail_s=tail, baseline_total_s=total)
    baseline_objective = objective(baseline)
    result["baseline_objective_s"] = baseline_objective
    if aggressive:
        training_needed = aggressive.get("training_needed_questions", 0)
        if 0 < training_needed <= len(known):
            result["training_ready_s"] = sorted(
                max(baseline[i] for i in group) for group in known
            )[training_needed - 1]

    best = (tail, total, [], "mixed")
    best_objective = baseline_objective
    orders = [
        sorted(options, key=lambda i: -rows[i]["draws"][0]),
        sorted(options, key=lambda i: options[i]["total_setup_s"]),
        sorted(options, key=lambda i: -baseline[i]),
    ]
    if aggressive:
        orders.append(sorted(options, key=lambda i: rows[i]["draws"][0]))
    probe = None
    visited = set()
    for order in orders:
        selected, used = [], 0
        for i in order:
            need = continuation_capacity(rows[i])
            if len(selected) >= room:
                break
            if used + need > free_tokens:
                continue
            used += need
            selected.append(i)
            variants = (
                ["mixed", "replay"]
                if not kv_only and any(options[j]["method"] == "kv" for j in selected)
                else ["mixed"]
            )
            for variant in variants:
                key = (tuple(selected), variant)
                if key in visited:
                    continue
                visited.add(key)
                after = forecast(selected, variant)
                candidate_tail, candidate_total = (
                    max(after.values(), default=0),
                    sum(after.values()),
                )
                costs = replay_options if variant == "replay" else options
                result["scenarios"].append(
                    dict(
                        count=len(selected),
                        variant=variant,
                        tail_s=candidate_tail,
                        total_s=candidate_total,
                        kv_count=sum(costs[j]["method"] == "kv" for j in selected),
                    )
                )
                candidate_objective = objective(after)
                result["scenarios"][-1]["objective_s"] = candidate_objective
                improves = (
                    (
                        candidate_objective < best_objective - 0.005
                        and candidate_tail <= tail + min(0.2, 0.02 * tail)
                        and candidate_total <= total
                    )
                    if aggressive
                    else (
                        candidate_tail < tail - overhead
                        and (candidate_tail, candidate_total) < best[:2]
                    )
                )
                if aggressive and aggressive.get("explore", False):
                    stage_cost = aggressive["coordination_s"] + sum(
                        costs[j]["setup_s"] for j in selected
                    )
                    eligible_probe = (
                        len(selected) <= 5
                        and stage_cost <= 1.6
                        and len(engines[source]["queue"]["running"]) >= 8
                        and len(engines[target]["active"]) * 1.5
                        < len(engines[source]["active"])
                        and all(
                            rows[j]["draws"][0]
                            * service(
                                source,
                                len(engines[source]["active"]),
                                rows[j]["context"],
                            )
                            >= 3 * stage_cost
                            for j in selected
                        )
                        and candidate_tail <= tail + 1.5
                        and candidate_objective <= baseline_objective + 1.5
                        and candidate_total <= total + 0.3 * len(rows)
                    )
                    if eligible_probe:
                        score = sum(rows[j]["draws"][0] for j in selected) / max(
                            0.001, stage_cost
                        )
                        if probe is None or score > probe[0]:
                            probe = (
                                score,
                                candidate_tail,
                                candidate_total,
                                list(selected),
                                variant,
                                candidate_objective,
                                stage_cost,
                            )
                if improves:
                    best_objective = candidate_objective
                    best = (candidate_tail, candidate_total, list(selected), variant)
    exploring = False
    if not best[2] and probe is not None:
        _, probe_tail, probe_total, selected, variant, best_objective, stage_cost = (
            probe
        )
        best = (probe_tail, probe_total, selected, variant)
        exploring = True
        result["exploration_stage_budget_s"] = stage_cost
    best_costs = replay_options if best[3] == "replay" else options
    result.update(
        selected=[
            dict(request_id=rows[i]["request_id"], **best_costs[i]) for i in best[2]
        ],
        predicted_tail_s=best[0],
        predicted_gain_s=tail - best[0],
        reason="predicted_gain" if best[2] else "no_predicted_gain",
    )
    if aggressive:
        result.update(
            predicted_objective_s=best_objective,
            predicted_gain_s=baseline_objective - best_objective,
            predicted_tail_gain_s=tail - best[0],
            policy="aggressive",
            cost_model=dict(aggressive),
        )
    if exploring:
        result.update(reason="bounded_exploration", exploration=True)
    planning_s = time.perf_counter() - planning_started
    result["planning_s"] = planning_s
    result["net_predicted_gain_s"] = result["predicted_gain_s"] - planning_s
    if result["selected"] and not exploring and result["net_predicted_gain_s"] <= 0:
        result["rejected_candidate"] = result["selected"]
        result.update(selected=[], reason="planning_cost_exceeds_gain")
    return result
