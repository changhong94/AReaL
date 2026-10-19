# SPDX-License-Identifier: Apache-2.0
"""Opt-in bounded KV handoff coordinator for a same-version TP=1 pair."""

import os
import statistics
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests

from areal.infra.batch_rebalance import KVTransferCost, plan_batch_rebalance
from areal.utils import request_trace


def rpc(address, action, data):
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(
            f"http://{address}/areal_kv_handoff",
            json=dict(action=action, data=data),
            timeout=25,
        )
        response.raise_for_status()
        return response.json()


def initialize_pair(addresses, source, target):
    from areal.utils.network import find_free_ports, gethostip

    data = dict(
        init_method=f"tcp://{gethostip()}:{find_free_ports(1)[0]}",
        group_name="areal-kv-" + uuid.uuid4().hex,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [
            executor.submit(rpc, addresses[rank], "init", dict(data, rank=i))
            for i, rank in enumerate((source, target))
        ]
        geometry = [job.result() for job in jobs]
    if geometry[0] != geometry[1]:
        raise RuntimeError("KV peer geometries differ")
    return geometry[0]


def exploration_timing_guard(plan, remaining_questions, elapsed_s=0.0):
    """Avoid speculative handoffs near training; never reject positive-gain plans.

    A batch-ready forecast is usable here; mean completion is not a readiness
    estimate. Missing forecasts retain the exact accepted-question fallback.
    """
    if not plan.get("exploration", False):
        return None
    if remaining_questions <= 1:
        return "training_near_ready"
    if plan.get("training_ready_s") is not None:
        baseline = plan["training_ready_s"]
        handoff = plan.get("exploration_stage_budget_s", 0.0)
        if baseline - elapsed_s <= handoff:
            return "training_horizon_too_short"
    return None


def rebalance(controller, version, route, engines, overhead, source, target):
    pool = controller._request_pool
    if getattr(controller, "_kv_rebalance_disabled", False):
        return
    if controller.training_prefetch_ready(controller.training_prefetch_target()):
        return
    # Start above the measured 0.273 s one-request RL handoff. Update from
    # observed whole-transaction latency,
    # never equate the isolated HCCL microbenchmark with production handoff cost.
    counts = [
        sum(row["rank"] == rank for row in route["active"]) for rank in (source, target)
    ]
    if counts[0] <= counts[1] + 1:
        request_trace.emit("kv_rebalance_skip", reason="balanced_or_single_tail")
        return
    policy = os.getenv("AREAL_KV_REBALANCE_POLICY", "exploratory")
    if policy not in {"conservative", "aggressive", "exploratory"}:
        raise ValueError("Unknown KV rebalance policy")
    aggressive = None
    observed = getattr(controller, "_kv_handoff_cost", 0.3)
    if policy in {"aggressive", "exploratory"}:
        samples = getattr(controller, "_kv_stage_samples", ())
        observed = statistics.median(samples) if samples else 0.18
        aggressive = dict(
            explore=policy == "exploratory"
            and getattr(controller, "_kv_probe_attempts", 0) < 2
            and getattr(controller, "_kv_probe_spent", 0.0) < 2.0,
            training_needed_questions=max(
                0,
                controller.training_prefetch_target()
                - controller.staleness_manager.get_stats().accepted,
            ),
            coordination_s=0.08,
            source_stall_s=0.02,
            target_stall_s=0.05,
            needed_questions=max(
                1,
                controller.training_prefetch_target()
                + (0 if controller._borrowed_pool.preserve_requests else 1)
                - controller.staleness_manager.get_stats().accepted,
            ),
        )
        for key in ("source_stall_s", "target_stall_s", "coordination_s"):
            history = getattr(controller, "_kv_" + key, ())
            if history:
                aggressive[key] = statistics.median(history)
    geometry = controller._kv_peer_geometry
    block_tokens = controller._kv_block_tokens
    cost = KVTransferCost(
        geometry["tensors"] * geometry["block_bytes"] // block_tokens,
        block_tokens,
        observed,
        0.0,
        64 * 1024**2,
        0.0,
    )
    started = time.monotonic()
    plan = plan_batch_rebalance(
        getattr(controller, "_queue_calibration", None),
        route,
        engines,
        overhead,
        transport=cost,
        source=source,
        target=target,
        max_moves=16,
        # A common barrier and device RPC stalls are conservatively charged to
        # both peers until all selected transfers are staged and aborted.
        request_local=False,
        kv_only=True,
        aggressive=aggressive,
    )
    # Require a margin above planning and snapshot cost; make at most one bounded
    # transaction per recovery. No periodic scans of a balanced pair.
    margin = 0.005 if aggressive else max(0.25, observed * 2)
    if (
        plan["selected"]
        and not plan.get("exploration", False)
        and plan.get("net_predicted_gain_s", 0) <= margin
    ):
        plan.update(selected=[], reason="insufficient_margin")
    if os.getenv("AREAL_KV_TRAINING_HORIZON_GATE", "0") == "1":
        remaining = (
            controller.training_prefetch_target()
            - controller.staleness_manager.get_stats().accepted
        )
        reason = exploration_timing_guard(plan, remaining, time.monotonic() - started)
        plan["training_remaining_questions"] = remaining
        if reason and plan["selected"]:
            plan.update(rejected_candidate=plan["selected"], selected=[], reason=reason)
    chosen = [x for x in plan["selected"] if x["method"] == "kv"]
    request_trace.emit("kv_rebalance_plan", version=version, **plan)
    request_trace.emit(
        "kv_rebalance_snapshot",
        version=version,
        route=route,
        engines=engines,
        overhead=overhead,
        transfer_cost=vars(cost),
        aggressive=aggressive,
    )
    if not chosen:
        return
    if os.getenv("AREAL_KV_REBALANCE_EXECUTE", "1") != "1":
        request_trace.emit("kv_rebalance_shadow", selected=chosen, version=version)
        return
    if len(chosen) != len(plan["selected"]):
        request_trace.emit("kv_rebalance_skip", reason="mixed_transport_plan")
        return
    leases = {x["request_id"]: x["lease_id"] for x in route["active"]}
    reservation = pool.reserve_rebalance(
        {x["request_id"]: leases[x["request_id"]] for x in chosen},
        version,
        source,
        target,
        running=True,
    )
    if set(reservation) != {x["request_id"] for x in chosen}:
        pool.cancel_rebalance(reservation)
        return
    exploring = plan.get("exploration", False)
    if exploring:
        controller._kv_probe_attempts = getattr(controller, "_kv_probe_attempts", 0) + 1
    planning_seconds = time.monotonic() - started
    transport_started = time.monotonic()
    stage_samples, source_stalls, target_stalls = [], [], []
    transfers = []
    submitted = False
    try:
        for rid in reservation:
            stage_started = time.monotonic()
            if controller.training_prefetch_ready(
                controller.training_prefetch_target()
            ):
                request_trace.emit("kv_rebalance_skip", reason="training_became_ready")
                return
            data = dict(
                ticket=uuid.uuid4().hex, request_id=engines[source]["internal_ids"][rid]
            )
            transfers.append((rid, data))
            deadline = time.monotonic() + 0.25
            while True:
                exported = rpc(pool.addresses[source], "export", data)
                if "skipped" not in exported:
                    break
                if (
                    exported["skipped"] != "unstable_computed_tokens"
                    or time.monotonic() >= deadline
                ):
                    return
                time.sleep(0.002)
            transfer = dict(data, bytes=exported["bytes"], geometry=geometry)
            with ThreadPoolExecutor(max_workers=2) as executor:
                jobs = [
                    executor.submit(
                        rpc,
                        pool.addresses[rank],
                        "transfer",
                        dict(transfer, source=rank == source),
                    )
                    for rank in (source, target)
                ]
                transfer_results = [job.result() for job in jobs]
            rpc(
                pool.addresses[target],
                "register",
                dict(
                    exported,
                    ticket=data["ticket"],
                    external_id=rid,
                    version=version,
                ),
            )
            stage_samples.append(time.monotonic() - stage_started)
            source_stalls.append(
                exported.get("worker_s", exported.get("pack_s", 0.0))
                + transfer_results[0].get(
                    "worker_s", transfer_results[0].get("transfer_s", 0.0)
                )
            )
            target_stalls.append(
                transfer_results[1].get(
                    "worker_s", transfer_results[1].get("transfer_s", 0.0)
                )
            )
        staging_seconds = time.monotonic() - transport_started
        if controller.training_prefetch_ready(controller.training_prefetch_target()):
            request_trace.emit("kv_rebalance_skip", reason="training_became_ready")
            return
        ids = [data["request_id"] for _, data in transfers]
        submitted = True
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                f"http://{pool.addresses[source]}/areal_queue_rebalance",
                json=dict(action="abort_running", internal_ids=ids),
                timeout=5,
            )
            response.raise_for_status()
            if set(response.json()["aborted"]) != set(ids):
                raise RuntimeError("Frozen source abort was not acknowledged")
        deadline = time.monotonic() + 5.0
        expected = {data["ticket"] for _, data in transfers}
        while True:
            reports = rpc(pool.addresses[target], "import_status", {})["results"]
            if expected.issubset(reports):
                if any(reports[t]["outcome"] != "import_applied" for t in expected):
                    controller._kv_rebalance_disabled = True
                    request_trace.emit(
                        "kv_rebalance_fallback", tickets=sorted(expected)
                    )
                break
            if time.monotonic() >= deadline:
                controller._kv_rebalance_disabled = True
                request_trace.emit(
                    "kv_rebalance_import_pending", tickets=sorted(expected)
                )
                break
            time.sleep(0.01)
        for i, (_, data) in enumerate(transfers):
            target_stalls[i] += reports.get(data["ticket"], {}).get("import_s", 0.0)
        elapsed = time.monotonic() - started
        controller._kv_handoff_cost = max(observed, elapsed / len(ids))
        if aggressive:
            observations = {
                "stage_samples": stage_samples,
                "source_stall_s": source_stalls,
                "target_stall_s": target_stalls,
                "coordination_s": [
                    max(0.0, elapsed - planning_seconds - staging_seconds)
                ],
            }
            for key, values in observations.items():
                history = getattr(controller, "_kv_" + key, deque(maxlen=32))
                history.extend(values)
                setattr(controller, "_kv_" + key, history)
        request_trace.emit(
            "kv_rebalance_handoff",
            request_ids=list(reservation),
            seconds=elapsed,
            policy=policy,
            planning_seconds=planning_seconds,
            staging_seconds=staging_seconds,
            source_stalls=source_stalls,
            target_stalls=target_stalls,
            version=version,
        )
    except Exception as exc:
        controller._kv_rebalance_disabled = True
        with pool.lock:
            pool.error = "KV handoff failed: " + type(exc).__name__
        raise
    finally:
        if exploring:
            controller._kv_probe_spent = (
                getattr(controller, "_kv_probe_spent", 0.0)
                + time.monotonic()
                - transport_started
            )
            request_trace.emit(
                "kv_rebalance_exploration_budget",
                attempts=controller._kv_probe_attempts,
                spent_s=controller._kv_probe_spent,
            )
        cleanup_errors = []
        for rid, data in transfers:
            cleanup = ([] if submitted else [(target, "discard")]) + [
                (source, "discard"),
                (source, "unfreeze"),
            ]
            for rank, action in cleanup:
                try:
                    rpc(pool.addresses[rank], action, data)
                except Exception as exc:
                    cleanup_errors.append(type(exc).__name__)
        if not submitted:
            pool.cancel_rebalance(reservation)
        if cleanup_errors:
            controller._kv_rebalance_disabled = True
            with pool.lock:
                pool.error = "KV cleanup failed: " + ",".join(cleanup_errors)
            raise RuntimeError(pool.error)
