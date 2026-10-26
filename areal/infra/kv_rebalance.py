# SPDX-License-Identifier: Apache-2.0
"""Opt-in bounded KV handoff coordinator for a same-version equal-TP pair."""

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


def pair_key(source, target):
    return f"replica-{source}-to-{target}"


def initialize_pairs(addresses, sources, targets):
    """Warm directed fixed-to-borrowed channels in deterministic serial order."""
    if (
        not sources
        or not targets
        or len(set(sources)) != len(sources)
        or len(set(targets)) != len(targets)
        or set(sources) & set(targets)
        or any(rank < 0 or rank >= len(addresses) for rank in [*sources, *targets])
    ):
        raise ValueError("KV peer sets must be distinct, nonempty valid replicas")
    peers = {}
    for source in sources:
        for target in targets:
            peers[source, target] = initialize_pair(addresses, source, target)
    return peers


def initialize_pair(addresses, source, target):
    from areal.utils.network import find_free_ports, gethostip

    info = [rpc(addresses[rank], "info", {}) for rank in (source, target)]
    if info[0] != info[1]:
        raise RuntimeError("KV peer geometries differ")
    width = info[0].get("tp_size", 1)
    data = dict(
        pair_id=pair_key(source, target),
        init_methods=[f"tcp://{gethostip()}:{port}" for port in find_free_ports(width)],
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
    geometry = controller._kv_peers[source, target]
    block_tokens = geometry["block_tokens"]
    cost = KVTransferCost(
        geometry["tensors"]
        * geometry["block_bytes"]
        * geometry.get("tp_size", 1)
        // block_tokens,
        block_tokens,
        observed,
        0.0,
        64 * 1024**2 * geometry.get("tp_size", 1),
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
    # transaction per pair in a serialized recovery sweep. No periodic balanced scans.
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
    request_trace.emit(
        "kv_rebalance_plan", version=version, source=source, target=target, **plan
    )
    request_trace.emit(
        "kv_rebalance_snapshot",
        source=source,
        target=target,
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
            transfer = dict(
                data,
                bytes=exported["bytes"],
                geometry=geometry,
                pair_id=pair_key(source, target),
            )
            # No rank enters HCCL until both replicas have staged every shard.
            preparation = []
            for rank in (source, target):
                preparation.append(
                    rpc(
                        pool.addresses[rank],
                        "prepare_transfer",
                        dict(transfer, source=rank == source),
                    )
                )
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
                preparation[0].get("worker_s", 0.0)
                + exported.get("worker_s", exported.get("pack_s", 0.0))
                + transfer_results[0].get(
                    "worker_s", transfer_results[0].get("transfer_s", 0.0)
                )
            )
            target_stalls.append(
                preparation[1].get("worker_s", 0.0)
                + transfer_results[1].get(
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
            source=source,
            target=target,
            pair_id=pair_key(source, target),
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


def evacuate_retained(controller):
    """Mandatory R027 evacuation, distinct from economically selected R001 moves.

    A single request is staged at a time. Source pause/backups remain intact until
    abort acknowledgement; target import is observed before staging the next.
    Different versions and unsupported requests use token-prefix replay.
    """
    pool = controller._request_pool
    timeout = controller.config.request_timeout
    with requests.Session() as session:
        session.trust_env = False
        for source in controller.borrowed_ranks:
            leases = pool.force_prefix_evacuation(source, list(controller.fixed_ranks))
            progress = session.post(
                f"http://{pool.addresses[source]}/areal_generation_progress",
                json={"include_internal_ids": True},
                timeout=timeout,
            )
            progress.raise_for_status()
            # Registry keys are native attempt IDs, exposed as internal_id below.
            native = progress.json()["internal_ids"]
            request_trace.emit(
                "retained_evacuation_begin",
                source=source,
                request_ids=[r["request_id"] for r in leases],
                economic_filter=False,
            )
            for lease in leases:
                rid = lease["request_id"]
                targets = [
                    r
                    for r in controller.fixed_ranks
                    if pool.ready[r]
                    and r not in pool.closing
                    and pool.versions[r] == lease["version"]
                    and (source, r) in getattr(controller, "_kv_peers", {})
                ]
                if not targets or not native.get(rid):
                    request_trace.emit(
                        "retained_evacuation_fallback",
                        request_id=rid,
                        reason="no_same_version_peer_or_native_request",
                        transport="prefix_replay",
                    )
                    continue
                target = min(targets, key=lambda r: len(pool.active_requests(r)))
                data = dict(
                    ticket=uuid.uuid4().hex,
                    request_id=native[rid],
                    retained_source=True,
                )
                submitted = False
                pool.pin_retained_kv(lease, target)
                try:
                    exported = rpc(pool.addresses[source], "export", data)
                    if "skipped" in exported:
                        request_trace.emit(
                            "retained_evacuation_fallback",
                            request_id=rid,
                            reason=exported["skipped"],
                            transport="prefix_replay",
                        )
                        continue
                    transfer = dict(
                        ticket=data["ticket"],
                        bytes=exported["bytes"],
                        geometry=controller._kv_peers[source, target],
                        pair_id=pair_key(source, target),
                        retained_transaction=True,
                    )
                    preparation = [
                        rpc(
                            pool.addresses[rank],
                            "prepare_transfer",
                            dict(
                                transfer,
                                source=rank == source,
                                retained_source=rank == source,
                            ),
                        )
                        for rank in (source, target)
                    ]
                    if any("rejected" in result for result in preparation):
                        request_trace.emit(
                            "retained_evacuation_fallback",
                            request_id=rid,
                            reason="target_staging_budget",
                            transport="prefix_replay",
                        )
                        continue
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        jobs = [
                            executor.submit(
                                rpc,
                                pool.addresses[rank],
                                "transfer",
                                dict(
                                    transfer,
                                    source=rank == source,
                                    retained_source=rank == source,
                                ),
                            )
                            for rank in (source, target)
                        ]
                        for job in jobs:
                            job.result()
                    rpc(
                        pool.addresses[target],
                        "register",
                        dict(
                            exported,
                            ticket=data["ticket"],
                            external_id=rid,
                            version=lease["version"],
                        ),
                    )
                    submitted = True
                    response = session.post(
                        f"http://{pool.addresses[source]}/areal_evacuate_retained",
                        json=dict(internal_ids=[data["request_id"]]),
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    if response.json()["aborted"] != [data["request_id"]]:
                        raise RuntimeError("Retained source abort was not acknowledged")
                    deadline = time.monotonic() + timeout
                    while True:
                        report = rpc(pool.addresses[target], "import_status", {})[
                            "results"
                        ].get(data["ticket"])
                        if report is not None:
                            pool.complete_retained_kv(lease)
                            request_trace.emit(
                                "retained_evacuation_import",
                                request_id=rid,
                                source=source,
                                target=target,
                                version=lease["version"],
                                import_report=report,
                            )
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Retained KV continuation was not acknowledged"
                            )
                        time.sleep(0.02)
                except Exception as exc:
                    with pool.lock:
                        pool.error = "Retained evacuation failed: " + type(exc).__name__
                    raise
                finally:
                    # Never invalidate target staging after ownership has moved.
                    if not submitted:
                        pool.cancel_retained_kv(lease, list(controller.fixed_ranks))
                        rpc(pool.addresses[target], "discard", data)
                    rpc(pool.addresses[source], "discard", data)
                    rpc(pool.addresses[source], "unfreeze", data)
            response = session.post(
                f"http://{pool.addresses[source]}/areal_evacuate_retained",
                json={},
                timeout=timeout,
            )
            response.raise_for_status()
            pool.wait_empty(source, timeout=timeout)
            request_trace.emit(
                "retained_evacuation_complete",
                source=source,
                released_source_leases=len(leases),
                remaining=0,
            )
