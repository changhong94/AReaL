# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from threading import Lock
from typing import Any

from flask import Flask, jsonify, request
from torchdata.stateful_dataloader import StatefulDataLoader
from werkzeug.serving import make_server

from areal.api import (
    InferenceEngine,
    Job,
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    RolloutWorkflow,
    Scheduler,
    WeightUpdateMeta,
    Worker,
    WorkflowLike,
)
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    InferenceEngineConfig,
    PerfTracerConfig,
    SchedulingSpec,
    SchedulingStrategy,
)
from areal.infra.borrowed_rollout import BorrowedRolloutPool
from areal.infra.request_router import RequestLeasePool
from areal.infra.rpc.serialization import deserialize_value
from areal.infra.utils.concurrent import get_executor, run_async_task
from areal.utils import logging, perf_tracer, request_trace
from areal.utils.data import cycle_dataloader
from areal.utils.dynamic_import import import_from_string
from areal.utils.network import find_free_ports, format_hostport, gethostip
from areal.utils.perf_tracer import trace_perf

from ..staleness_manager import StalenessManager
from ..workflow_executor import BatchTaskDispatcher, TaskIdGenerator

logger = logging.getLogger("RolloutController")


# NOTE: remote task input has a slightly different
# type annotation, which disallows workflow object or types
@dataclass
class _RemoteRolloutTaskInput:
    task_id: int
    data: dict[str, Any]
    workflow: str | None
    workflow_kwargs: dict[str, Any]
    should_accept_fn: str | None
    is_eval: bool = False
    group_size: int = 1
    proxy_addr: str | None = None


@dataclass
class _RemoteRolloutResult:
    task_id: int
    trajectory: dict[str, Any]


class RolloutController:
    def __init__(
        self,
        inf_engine: type[InferenceEngine],
        config: InferenceEngineConfig,
        scheduler: Scheduler,
    ):
        self.inf_engine = inf_engine
        self.config = config
        self.scheduler = scheduler

        # Parse allocation from config.backend
        self.rollout_alloc = ModelAllocation.from_str(config.backend)

        # Worker management
        self.workers: list[Worker] = []  # List of Worker objects from scheduler
        self.server_infos: list[LocalInfServerInfo] = []
        self._worker_role: str

        # Round-robin scheduling
        self._current_worker_idx = 0
        self._shared_publication = None
        self._shared_publication_lock = threading.RLock()
        self._shared_loading = False
        self._borrowed_pool: BorrowedRolloutPool | None = None
        self._request_pool: RequestLeasePool | None = None
        self._worker_roles: list[str] = []

        # State
        self._version_lock = Lock()
        self._version = 0

        self._task_id_generator = TaskIdGenerator()

        # Use provided staleness manager or create a default one
        # The manager will be properly initialized in initialize()
        self._staleness_manager: StalenessManager | None = None

        # Dispatcher will be initialized in initialize() after staleness_manager is ready
        self._dispatcher: (
            BatchTaskDispatcher[_RemoteRolloutTaskInput, _RemoteRolloutResult] | None
        ) = None

        # HTTP callback server
        self._callback_app: Flask | None = None
        self._callback_server = None
        self._route_push_server = None
        self._callback_server_thread: threading.Thread | None = None
        self._callback_port: int | None = None
        self._callback_host: str | None = None
        self._callback_loop: asyncio.AbstractEventLoop | None = None
        self._callback_loop_ready = threading.Event()
        self._callback_control_lock = Lock()

        # Task completion futures
        self._pending_futures: dict[int, asyncio.Future] = {}
        self._generation_workers: dict[int, int] = {}
        self._futures_lock = threading.Lock()

        # Proxy worker management (for AgentWorkflow support)
        self.proxy_workers: list[Worker] = []
        self.proxy_addrs: list[str] = []
        self._proxy_started = False

        # Proxy gateway server (for online/external access)
        self._proxy_gateway_app = None
        self._proxy_gateway_server = None
        self._proxy_gateway_thread: threading.Thread | None = None
        self._proxy_gateway_port: int | None = None
        self._proxy_gateway_host: str | None = None

    @property
    def _proxy_role(self) -> str:
        """Generate a unique proxy role name based on the worker role.

        This avoids collisions when multiple controllers (e.g., rollout and
        eval-rollout) each fork proxy workers into the same scheduler.
        """
        if not hasattr(self, "_worker_role"):
            raise RuntimeError(
                "Cannot access _proxy_role before initialize() is called"
            )
        return f"proxy-{self._worker_role}"

    def _proxy_engine_name(self, rank: int) -> str:
        """Generate engine name for a proxy worker rank."""
        return f"{self._proxy_role}/{rank}"

    def _engine_name(self, rank: int) -> str:
        """Generate engine name for a worker rank.

        Engine names follow the "role/index" format (e.g., "rollout/0", "rollout/1").
        """
        return f"{self._worker_role}/{rank}"

    def enable_npu_share_pool(self, borrowed_replicas: int | None = None) -> None:
        """Colocate the leading replicas on the actor's physical device group."""
        total = self.rollout_alloc.parallel.dp_size
        borrowed = total // 2 if borrowed_replicas is None else borrowed_replicas
        if not 0 < borrowed < total:
            raise ValueError("NPU sharing requires nonempty borrowed and fixed groups")
        self._borrowed_replica_count = borrowed
        self._preserve_borrowed_requests = (
            os.getenv("AREAL_NPU_SHARE_PAUSE_KEEP", "0") == "1"
        )
        self._borrowed_pool = BorrowedRolloutPool(
            self.config.consumer_batch_size,
            preserve_requests=self._preserve_borrowed_requests,
            replicas_per_group=borrowed,
            fixed_replicas=total - borrowed,
        )

    @property
    def has_borrowed_replica(self) -> bool:
        return self._borrowed_pool is not None

    def _pool_event(self, event: str, **fields) -> None:
        logger.info("NPU_SHARE_POOL " + json.dumps({"event": event, **fields}))

    def _worker_rpc(self, rank: int, method: str, **kwargs):
        return run_async_task(
            self.scheduler.async_call_engine,
            worker_id=self.workers[rank].id,
            engine_name=self._engine_name(rank),
            method=method,
            **kwargs,
        )

    @property
    def borrowed_ranks(self):
        return tuple(
            range(
                getattr(
                    self,
                    "_borrowed_replica_count",
                    self.rollout_alloc.parallel.dp_size // 2,
                )
            )
        )

    @property
    def fixed_ranks(self):
        return tuple(
            range(
                len(self.borrowed_ranks),
                self.rollout_alloc.parallel.dp_size,
            )
        )

    def _group_rpc(self, ranks, method, **kwargs):
        if len(ranks) == 1:
            return [self._worker_rpc(ranks[0], method, **kwargs)]

        async def call():
            return await asyncio.gather(
                *[
                    self.scheduler.async_call_engine(
                        worker_id=self.workers[rank].id,
                        engine_name=self._engine_name(rank),
                        method=method,
                        **kwargs,
                    )
                    for rank in ranks
                ]
            )

        return run_async_task(call)

    def _borrowed_requests(self):
        return [
            request
            for rank in self.borrowed_ranks
            for request in self._request_pool.active_requests(rank)
        ]

    def training_prefetch_target(self) -> int:
        if self._borrowed_pool is None:
            raise RuntimeError("No borrowed replica configured")
        return self._borrowed_pool.training_prefetch_target()

    def training_prefetch_ready(self, target: int) -> bool:
        return self.staleness_manager.get_stats().accepted >= target

    def wait_running_rebalance(self) -> None:
        """Fence native pauses against an in-flight running-request handoff."""
        if os.getenv("AREAL_RUNNING_REBALANCE", "off") != "kv":
            return
        future = getattr(self, "_queue_rebalance_future", None)
        if future is not None:
            started = time.monotonic()
            future.result()
            request_trace.emit(
                "kv_rebalance_transition_wait", seconds=time.monotonic() - started
            )

    def borrow_for_training(self) -> None:
        """Only the actor's replica stops; the other replica keeps generating."""
        if self._borrowed_pool is None:
            raise RuntimeError("No borrowed replica configured")
        self.wait_shared_publication()
        self.wait_running_rebalance()
        self._borrowed_pool.withdraw()
        pool = getattr(self, "_request_pool", None)
        if pool is not None:
            for replica_rank in self.borrowed_ranks:
                pool.close(replica_rank)
        else:
            self._group_rpc(self.borrowed_ranks, "pause", http_timeout=60.0)
        preserve = getattr(self, "_preserve_borrowed_requests", False)
        if preserve and pool is None:
            raise RuntimeError("Retained requests require request lease routing")
        method = "pause_generation_preserve" if preserve else "pause_generation"
        self._group_rpc(self.borrowed_ranks, method, http_timeout=60.0)
        if pool is not None and not preserve:
            for replica_rank in self.borrowed_ranks:
                pool.wait_empty(replica_rank)
        self._pool_event(
            "borrowed", ranks=list(self.borrowed_ranks), version=self.get_version()
        )

    def begin_shared_drain(self, timeout: float) -> float:
        """Close new leases without interrupting old-version native generation."""
        pool = getattr(self, "_request_pool", None)
        if pool is None or self._borrowed_pool is None:
            raise RuntimeError("Shared drain requires request lease routing")
        self.wait_shared_publication()
        self.wait_running_rebalance()
        self._borrowed_pool.withdraw()
        for replica_rank in self.borrowed_ranks:
            pool.close(replica_rank)
        deadline = time.monotonic() + timeout
        request_trace.emit(
            "shared_drain_begin",
            versions=[pool.versions[r] for r in self.borrowed_ranks],
            timeout=timeout,
            request_ids=self._borrowed_requests(),
        )
        return deadline

    def freeze_drain_window(self) -> None:
        if os.getenv("AREAL_BATCH_DRAIN_METRICS", "0") != "1":
            return
        from threading import Event

        import requests

        from areal.infra.rolling_drain import RollingBatchCalibration

        if not hasattr(self, "_batch_calibration"):
            self._batch_calibration = RollingBatchCalibration(
                replicas=self.rollout_alloc.parallel.dp_size,
                borrowed_replicas=len(self.borrowed_ranks),
            )
            self._batch_drain_wakeup = Event()
        with requests.Session() as session:
            session.trust_env = False
            engines = []
            for address in self._request_pool.addresses:
                response = session.post(
                    f"http://{address}/areal_generation_progress",
                    json={"rotate": True},
                    timeout=2,
                )
                response.raise_for_status()
                engines.append(response.json())
        route = self._request_pool.drain_snapshot(rotate=True)
        handoffs = getattr(self, "_batch_handoff_parts", [])
        request_trace.emit(
            "batch_drain_calibration_input",
            engines=engines,
            route=route,
            handoffs=handoffs,
        )
        self._return_observation = (engines, route)
        summary = self._batch_calibration.freeze(engines, route, handoffs)
        if os.getenv("AREAL_QUEUE_REBALANCE", "1") == "1":
            import copy

            # Publish an immutable completed calibration for the independent
            # rebalance worker; freeze() mutates the live estimator next batch.
            self._queue_calibration = copy.deepcopy(self._batch_calibration)
        self._batch_handoffs = []
        self._batch_handoff_parts = []
        request_trace.emit("batch_drain_calibration", **summary)

    def decide_inference_return(self, target: int) -> dict:
        from areal.infra.profit_return import estimate_return, uncalibrated_return

        accepted = self.staleness_manager.get_stats().accepted
        observation = getattr(self, "_return_observation", None)
        calibration = getattr(self, "_batch_calibration", None)
        if target <= accepted:
            result = dict(action="keep", reason="next_batch_ready", gain_seconds=0.0)
        elif observation is None or calibration is None:
            result = uncalibrated_return("missing_observation_probe")
        else:
            engines, route = observation
            result = estimate_return(
                calibration,
                route,
                engines,
                target - accepted,
                getattr(self, "_return_costs", []),
                margin=float(os.getenv("AREAL_RETURN_MARGIN_SECONDS", "0")),
            )
        request_trace.emit(
            "inference_return_decision", target=target, accepted=accepted, **result
        )
        return result

    def evacuate_retained_requests(self) -> None:
        """Evacuate every paused request; economics never veto a continuation."""
        from areal.infra.kv_rebalance import evacuate_retained

        self.wait_running_rebalance()
        if self._request_pool is None:
            raise RuntimeError("Retained evacuation requires request lease routing")
        if getattr(self, "_shared_held_for_training", False):
            if any(self._request_pool.active_requests(r) for r in self.borrowed_ranks):
                raise RuntimeError("Held training replica unexpectedly owns requests")
            return
        evacuate_retained(self)

    def record_inference_return(self, seconds: float) -> None:
        self._return_publish_cost = seconds

    def skip_inference_return(self, version: int) -> None:
        # No snapshot exists for this version. Keep leases closed and old model
        # versions unchanged until a later full load ACK, including during train.
        self._shared_held_for_training = True
        self._batch_publish_seconds = None
        self._batch_transition_started = None
        request_trace.emit("inference_return_skipped", version=version)

    def record_training_ready(
        self, prepare_seconds: float, idle_seconds: float | None = None
    ) -> None:
        if os.getenv("AREAL_BATCH_DRAIN_METRICS", "0") != "1":
            return
        # Publication tail finishes on this same executor BEFORE preparation.
        # Never await a future queued behind the drain that is making a decision.
        return_cost = getattr(self, "_return_publish_cost", None)
        if return_cost is not None:
            from collections import deque

            if not hasattr(self, "_return_costs"):
                self._return_costs = deque(maxlen=3)
            self._return_costs.append(
                dict(publish=return_cost, restore=prepare_seconds)
            )
            self._return_publish_cost = None
        publish = getattr(self, "_batch_publish_seconds", None)
        if publish is not None:
            self._batch_handoffs = getattr(self, "_batch_handoffs", [])
            handoff = publish + prepare_seconds
            if idle_seconds is not None:
                handoff = (
                    time.monotonic() - self._batch_transition_started - idle_seconds
                )
            self._batch_handoffs.append(handoff)
            self._batch_handoff_parts = getattr(self, "_batch_handoff_parts", [])
            self._batch_handoff_parts.append(
                dict(tail=max(0.0, handoff - prepare_seconds), restore=prepare_seconds)
            )
            self._batch_publish_seconds = None
            self._batch_transition_started = None
            request_trace.emit(
                "batch_drain_training_ready",
                publish_seconds=publish,
                prepare_seconds=prepare_seconds,
                idle_seconds=idle_seconds,
                handoff=handoff,
            )

    def _adaptive_drain_reason(self, target: int) -> str:
        import requests

        from areal.infra.adaptive_drain import estimate_drain

        pool = self._request_pool
        # A watchdog is failure protection, not the normal scheduling deadline.
        watchdog = time.monotonic() + self.config.request_timeout
        votes = 0
        batch_mode = os.getenv("AREAL_BATCH_DRAIN", "0") == "1"
        with requests.Session() as session:
            session.trust_env = False
            while True:
                if not self._borrowed_requests():
                    return "empty"
                if self.training_prefetch_ready(target):
                    return "next_batch_ready"
                if (
                    self.get_version()
                    - min(pool.versions[r] for r in self.borrowed_ranks)
                    > self.config.max_head_offpolicyness
                ):
                    return "version_limit"
                if time.monotonic() >= watchdog:
                    return "watchdog"
                engines = []
                try:
                    for address in pool.addresses:
                        response = session.post(
                            f"http://{address}/areal_generation_progress",
                            json={},
                            timeout=2,
                        )
                        response.raise_for_status()
                        engines.append(response.json())
                except (requests.RequestException, ValueError) as exc:
                    request_trace.emit(
                        "adaptive_drain_estimate",
                        action="stop",
                        reason="telemetry_error",
                        error=str(exc),
                    )
                    return "telemetry_error"
                # Every endpoint constructs a fresh CPU snapshot. HTTP timeouts
                # bound retrieval; monotonic clocks are not comparable across hosts.
                if any(
                    not engines[r]["budget_valid"]
                    or not engines[r]["generation_running"]
                    for r in self.borrowed_ranks
                ):
                    return "budget_invalid"
                if batch_mode:
                    accepted = self.staleness_manager.get_stats().accepted
                    route = pool.drain_snapshot()
                    needed = target - accepted
                    prefetch_needed = max(0, self.training_prefetch_target() - accepted)
                    request_trace.emit(
                        "batch_drain_prediction_input",
                        route=route,
                        engines=engines,
                        needed=needed,
                        prefetch_needed=prefetch_needed,
                    )
                    started = time.perf_counter()
                    estimate = self._batch_calibration.estimate(
                        route, engines, needed, prefetch_needed=prefetch_needed
                    )
                    estimate["prediction_seconds"] = time.perf_counter() - started
                else:
                    estimate = estimate_drain(
                        pool.drain_snapshot(),
                        engines,
                        target - self.staleness_manager.get_stats().accepted,
                        getattr(self, "_shared_turnaround_ema", 1.0),
                    )
                request_trace.emit(
                    "batch_drain_estimate" if batch_mode else "adaptive_drain_estimate",
                    **estimate,
                )
                if batch_mode:
                    # Prediction is CPU work: recheck progress before acting.
                    if not self._borrowed_requests():
                        return "empty"
                    if self.training_prefetch_ready(target):
                        return "next_batch_ready"
                    if (
                        self.get_version()
                        - min(pool.versions[r] for r in self.borrowed_ranks)
                        > self.config.max_head_offpolicyness
                    ):
                        return "version_limit"
                if batch_mode and estimate["action"] == "stop":
                    return estimate["reason"]
                if estimate["reason"] == "next_batch_ready":
                    return "next_batch_ready"
                votes = votes + 1 if estimate["action"] == "stop" else 0
                if votes >= 2:
                    return "migration_faster"
                if batch_mode:
                    self._batch_drain_wakeup.wait(0.5)
                    self._batch_drain_wakeup.clear()
                else:
                    time.sleep(0.5)

    def finish_shared_drain(self, deadline: float, target: int) -> None:
        pool = self._request_pool
        if os.getenv("AREAL_ADAPTIVE_DRAIN", "0") == "1":
            reason = self._adaptive_drain_reason(target)
        else:
            reason = "empty"
            for rank in self.borrowed_ranks:
                reason = pool.drain_until(
                    rank, deadline, lambda: self.training_prefetch_ready(target)
                )
                if reason != "empty":
                    break
        remaining = self._borrowed_requests()
        self._batch_transition_started = time.monotonic()
        request_trace.emit(
            "shared_drain_decision", reason=reason, request_ids=remaining, target=target
        )
        # Even natural completion needs native idle ACK before capacity/model mutation.
        # On timeout this aborts only the remaining leases, whose clients reroute.
        self._group_rpc(self.borrowed_ranks, "pause_generation", http_timeout=60.0)
        for replica_rank in self.borrowed_ranks:
            pool.wait_empty(replica_rank)
        self._shared_transition_started = time.monotonic()
        request_trace.emit(
            "shared_drain_end", reason=reason, remaining_at_decision=len(remaining)
        )

    def resume_training_inference(self) -> None:
        if getattr(self, "_shared_held_for_training", False):
            raise RuntimeError("Shared model is stale after a skipped return")
        if getattr(self, "_request_pool", None) is None:
            raise RuntimeError("Concurrent training requires request lease routing")
        self._group_rpc(self.borrowed_ranks, "continue_generation", http_timeout=60.0)
        for replica_rank in self.borrowed_ranks:
            self._request_pool.reopen(replica_rank, self.get_version())
        self._group_rpc(self.borrowed_ranks, "resume", http_timeout=60.0)
        self._pool_event(
            "training_inference_resumed",
            ranks=list(self.borrowed_ranks),
            version=self.get_version(),
        )

    def wait_shared_publication(self) -> None:
        future = getattr(self, "_shared_publication", None)
        if future is not None:
            future.result()  # Surface failures before another actor update or evaluation.

    def prepare_fixed_publication(self) -> None:
        self.wait_shared_publication()
        # Start every publication cycle, including cycles with no shared drain.
        # A later drain decision narrows this to the non-overlapped handoff tail.
        self._batch_transition_started = time.monotonic()
        self._batch_publish_seconds = None
        if self._borrowed_pool is None:
            raise RuntimeError("Asymmetric publication requires a borrowed replica")
        self.wait_running_rebalance()
        self._borrowed_pool.withdraw()
        pool = getattr(self, "_request_pool", None)
        if pool is not None:
            for replica_rank in self.fixed_ranks:
                pool.close(replica_rank)
        self._group_rpc(self.fixed_ranks, "pause_generation", http_timeout=60.0)
        if pool is not None:
            for replica_rank in self.fixed_ranks:
                pool.wait_empty(replica_rank)
        self._pool_event("fixed_pause", version=self.get_version())

    def publish_fixed_version(self, version: int) -> None:
        # Caller has awaited all HCCL buckets and the receiver's load ACK.
        self._group_rpc(
            self.fixed_ranks, "set_version", version=version, http_timeout=60.0
        )
        with self._version_lock:
            self._version = version
        self._group_rpc(self.fixed_ranks, "continue_generation", http_timeout=60.0)
        pool = getattr(self, "_request_pool", None)
        if pool is not None:
            for replica_rank in self.fixed_ranks:
                pool.open(replica_rank, version)
            self._collective_rpc("resume", http_timeout=60.0)
        else:
            self._group_rpc(self.fixed_ranks, "resume", http_timeout=60.0)
        self.dispatcher.resume()
        self._pool_event(
            "fixed_ready",
            ranks=list(self.fixed_ranks),
            version=version,
            time=time.time(),
        )

    def load_shared_snapshot_async(
        self, meta: WeightUpdateMeta, restore_memory: bool = False
    ) -> None:
        self.wait_shared_publication()
        if meta.version != self.get_version() or not meta.npu_share_snapshot:
            raise ValueError("Shared snapshot must match the published fixed version")
        with self._shared_publication_lock:
            self._shared_loading = True
            self._shared_publication = get_executor(
                "shared_publication", max_workers=1
            ).submit(
                self._load_shared_snapshot,
                replace(meta, clear_checkpoint_after_load=False),
                restore_memory,
            )

    def _schedule_queue_rebalance(self, version: int) -> None:
        if os.getenv("AREAL_QUEUE_REBALANCE", "1") != "1":
            return
        if os.getenv("AREAL_BATCH_DRAIN_METRICS", "0") != "1":
            return
        previous = getattr(self, "_queue_rebalance_future", None)
        if previous is not None and not previous.done():
            request_trace.emit(
                "queue_rebalance_skipped", version=version, reason="worker_busy"
            )
            return
        self._queue_rebalance_future = get_executor(
            "queue_rebalance", max_workers=1
        ).submit(self._rebalance_queued_once, version)

    def _rebalance_queued_once(self, version: int) -> None:
        # Refresh each pair's snapshot after reserving earlier moves.
        for target_rank in self.borrowed_ranks:
            for source_rank in self.fixed_ranks:
                self._rebalance_pair(version, source_rank, target_rank)

    def _rebalance_pair(self, version: int, source_rank: int, target_rank: int) -> None:
        if os.getenv("AREAL_QUEUE_REBALANCE", "1") != "1":
            return
        pool = getattr(self, "_request_pool", None)
        if pool is None or os.getenv("AREAL_BATCH_DRAIN_METRICS", "0") != "1":
            return
        with pool.lock:
            if (
                not all(pool.ready)
                or any(v != version for v in pool.versions)
                or pool.error
            ):
                return
        import requests

        from areal.infra.queue_rebalance import plan_rebalance

        reservations = {}
        submitted = False
        try:
            with requests.Session() as session:
                session.trust_env = False
                start = time.monotonic()
                engines = []
                for address in pool.addresses:
                    response = session.post(
                        f"http://{address}/areal_queue_rebalance",
                        json={"action": "snapshot"},
                        timeout=2,
                    )
                    response.raise_for_status()
                    engines.append(response.json())
                overhead = time.monotonic() - start
                route = pool.drain_snapshot()
                running_mode = os.getenv("AREAL_RUNNING_REBALANCE", "off")
                shadow = None
                if running_mode in {"shadow", "active", "budget"}:
                    from areal.infra.running_rebalance import plan_running_shadow

                    shadow_started = time.monotonic()
                    shadow = plan_running_shadow(
                        getattr(self, "_queue_calibration", None),
                        route,
                        engines,
                        overhead,
                        source_rank,
                        target_rank,
                    )
                    request_trace.emit(
                        "running_rebalance_shadow",
                        **shadow,
                        planning_seconds=time.monotonic() - shadow_started,
                        route=route,
                        engines=engines,
                    )
                plan = plan_rebalance(
                    getattr(self, "_queue_calibration", None),
                    route,
                    engines,
                    overhead,
                    source_rank,
                    target_rank,
                )
                if running_mode == "kv" and not plan["selected"]:
                    from areal.infra.kv_rebalance import rebalance

                    rebalance(
                        self,
                        version,
                        route,
                        engines,
                        overhead,
                        source_rank,
                        target_rank,
                    )
                    return
                running_move = False
                if (
                    not plan["selected"]
                    and running_mode == "active"
                    and shadow
                    and shadow["selected"]
                ):
                    if self.training_prefetch_ready(self.training_prefetch_target()):
                        shadow.update(selected=[], reason="training_imminent")
                    else:
                        plan = {k: v for k, v in shadow.items() if k != "version"}
                        running_move = True
                if running_mode in {"shadow", "active", "budget"}:
                    request_trace.emit(
                        "running_rebalance_decision",
                        policy=running_mode,
                        selected=plan["selected"] if running_move else [],
                        reason=shadow["reason"] if shadow else "not_planned",
                        version=version,
                    )
                planning_cost = time.monotonic() - start
                plan["planning_seconds"] = planning_cost
                if any(
                    min(p["tail_gain_s"], p["stay_s"] - p["move_s"]) <= planning_cost
                    for p in plan["predictions"]
                ):
                    plan.update(selected=[], reason="planning_cost_exceeds_gain")
                request_trace.emit("queue_rebalance_plan", version=version, **plan)
                if not plan["selected"]:
                    return
                leases = {r["request_id"]: r["lease_id"] for r in route["active"]}
                reservations = pool.reserve_rebalance(
                    {rid: leases[rid] for rid in plan["selected"]},
                    version,
                    source_rank,
                    target_rank,
                    **({"running": True} if running_move else {}),
                )
                if not reservations:
                    return
                native = engines[source_rank]["internal_ids"]
                ids = [native[rid] for rid in reservations]
                submitted = True
                response = session.post(
                    f"http://{pool.addresses[source_rank]}/areal_queue_rebalance",
                    json={
                        "action": "abort_running" if running_move else "abort",
                        "internal_ids": ids,
                    },
                    timeout=2,
                )
                response.raise_for_status()
                acknowledged = set(response.json()["aborted"])
                pool.cancel_rebalance(
                    {
                        rid: lease
                        for rid, lease in reservations.items()
                        if native[rid] not in acknowledged
                    }
                )
                request_trace.emit(
                    "queue_rebalance_result",
                    running=running_move,
                    execution_path="prefix_replay"
                    if running_move
                    else "queued_restart",
                    version=version,
                    requested=len(ids),
                    withdrawn=len(acknowledged),
                    request_ids=[
                        rid for rid in reservations if native[rid] in acknowledged
                    ],
                    seconds=time.monotonic() - start,
                )
        except Exception as exc:
            # A timed-out optimization cannot grant execution. Keep the source
            # lease until its ordinary generation HTTP response confirms abort.
            if not submitted:
                pool.cancel_rebalance(reservations)
            request_trace.emit(
                "queue_rebalance_skipped",
                version=version,
                reason=type(exc).__name__,
                uncertain=submitted,
            )

    def _load_shared_snapshot(
        self, meta: WeightUpdateMeta, restore_memory: bool = False
    ) -> None:
        started = time.perf_counter()
        self._pool_event(
            "shared_load_start",
            ranks=list(self.borrowed_ranks),
            version=meta.version,
            time=time.time(),
        )
        try:
            if restore_memory:
                self._group_rpc(
                    self.borrowed_ranks, "colocation_memory", action="restore"
                )
            if getattr(self, "_preserve_borrowed_requests", False):
                # Memory policy has restored the old model and its live KV.
                # Keep admission closed until all old-version leases complete.
                request_trace.emit("shared_retained_drain_begin", version=meta.version)
                self._group_rpc(
                    self.borrowed_ranks, "continue_generation", http_timeout=60.0
                )
                for replica_rank in self.borrowed_ranks:
                    self._request_pool.wait_empty(
                        replica_rank, timeout=self.config.request_timeout
                    )
                request_trace.emit("shared_retained_drain_end", version=meta.version)
                self._group_rpc(
                    self.borrowed_ranks, "pause_generation", http_timeout=60.0
                )
            request_trace.emit("shared_model_load_begin", version=meta.version)
            self._group_rpc(self.borrowed_ranks, "update_weights_from_disk", meta=meta)
            request_trace.emit("shared_model_load_end", version=meta.version)
            with self._shared_publication_lock:
                if meta.version != self.get_version():
                    raise RuntimeError("Shared publication was superseded before ACK")
                self._group_rpc(
                    self.borrowed_ranks,
                    "set_version",
                    version=meta.version,
                    http_timeout=60.0,
                )
                self._group_rpc(
                    self.borrowed_ranks, "continue_generation", http_timeout=60.0
                )
                pool = getattr(self, "_request_pool", None)
                if pool is not None:
                    for replica_rank in self.borrowed_ranks:
                        pool.open(replica_rank, meta.version)
                self._group_rpc(self.borrowed_ranks, "resume", http_timeout=60.0)
                self._shared_held_for_training = False
                self._borrowed_pool.release(self.staleness_manager.get_stats().accepted)
                self._shared_loading = False
                transition = getattr(self, "_shared_transition_started", None)
                if transition is not None:
                    elapsed = time.monotonic() - transition
                    self._shared_turnaround_ema = (
                        0.7 * getattr(self, "_shared_turnaround_ema", elapsed)
                        + 0.3 * elapsed
                    )
                batch_transition = getattr(self, "_batch_transition_started", None)
                if batch_transition is not None:
                    self._batch_publish_seconds = time.monotonic() - batch_transition
                request_trace.emit("shared_admission_ready", version=meta.version)
                self._schedule_queue_rebalance(meta.version)
                self._pool_event(
                    "returned",
                    ranks=list(self.borrowed_ranks),
                    version=meta.version,
                    routing_state=self._borrowed_pool.state,
                    seconds=time.perf_counter() - started,
                    time=time.time(),
                )
            # Only the shared reader uses this snapshot; its ACK includes device sync.
            shutil.rmtree(meta.path, ignore_errors=True)
        except Exception as exc:
            self._borrowed_pool.withdraw()
            self.dispatcher.pause()
            self.dispatcher.fail_fast(exc)
            self._pool_event(
                "shared_load_failed",
                ranks=list(self.borrowed_ranks),
                version=meta.version,
            )
            raise

    def initialize(
        self,
        role: str,
        server_args: dict[str, Any] | None = None,
        server_infos: list[LocalInfServerInfo] | None = None,
        *args,
        **kwargs,
    ):
        # Get scheduling config from kwargs or use defaults
        # Schedule inference engines in the granularity of instance sizes,
        # usually TP x PP.
        self._worker_role = role

        instance_size = (
            self.rollout_alloc.parallel.tp_size * self.rollout_alloc.parallel.pp_size
        )
        dp_size = self.rollout_alloc.parallel.dp_size

        # The first element of `self.config.scheduling_spec` is the resource spec
        # of workers, aka the RPC server process. Since a worker exactly matches
        # to a single engine instance in the local environment, we can dirrectly
        # use the spec of engines  as the spec of workers here. Engine scheduling
        # specs are ignored.
        sch_spec = SchedulingSpec(**asdict(self.config.scheduling_spec[0]))
        sch_spec.cpu *= instance_size
        sch_spec.mem *= instance_size
        if sch_spec.gpu > 0:
            sch_spec.gpu = instance_size

        job = Job(
            replicas=dp_size,
            tasks=[sch_spec for _ in range(dp_size)],
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )

        # Call async scheduler methods synchronously
        try:
            run_async_task(
                self._async_initialize, job, server_args, server_infos, *args, **kwargs
            )
        except BaseException:
            self._delete_worker_roles()
            raise

        # Initialize staleness manager for global capacity control
        max_concurrent_rollouts = (
            self.config.max_concurrent_rollouts or self.config.consumer_batch_size
        )
        consumer_batch_size = self.config.consumer_batch_size
        concurrency_target = None
        if (
            os.getenv("AREAL_ADAPTIVE_CONCURRENCY", "off") == "capacity"
            and self._borrowed_pool is not None
        ):
            from areal.infra.adaptive_concurrency import CapacityConcurrency

            concurrency_target = CapacityConcurrency(
                max_concurrent_rollouts,
                len(self.fixed_ranks),
                self._concurrency_snapshot,
            )
            # Bound admission by the configured replica topology, not a fixed multiple.
            max_concurrent_rollouts = concurrency_target.capacity_for(
                self.rollout_alloc.parallel.dp_size
            )
        self._staleness_manager = StalenessManager(
            version_provider=self,
            max_concurrent_rollouts=max_concurrent_rollouts,
            consumer_batch_size=consumer_batch_size,
            max_staleness=self.config.max_head_offpolicyness,
            concurrency_target=concurrency_target,
        )

        # Create and initialize the dispatcher
        qsize = (
            self.config.queue_size
            or (self.config.max_concurrent_rollouts or self.config.consumer_batch_size)
            * 16
        )
        self._dispatcher = BatchTaskDispatcher[
            _RemoteRolloutTaskInput, _RemoteRolloutResult
        ](
            max_queue_size=qsize,
            task_factory=self._create_submit_callback,
            staleness_manager=self._staleness_manager,
            enable_tracing=self.config.enable_rollout_tracing,
        )
        # Initialize the dispatcher's async task runner
        self._dispatcher.initialize(logger=logger)

        # Start callback server for weight sync coordination
        if self._borrowed_pool is not None:
            self._request_pool = RequestLeasePool(
                [format_hostport(info.host, info.port) for info in self.server_infos],
                capacity=server_args["max_num_seqs"],
            )
        self._start_callback_server()
        if self._request_pool is not None:
            self._collective_rpc("configure_request_router", address=self.callback_addr)
            if os.getenv("AREAL_RUNNING_REBALANCE", "off") == "kv":
                from areal.infra.kv_rebalance import initialize_pairs

                if self.rollout_alloc.parallel.pp_size != 1:
                    raise ValueError(
                        "KV handoff requires equal-TP inference replicas and PP=1"
                    )
                self._kv_peers = initialize_pairs(
                    self._request_pool.addresses,
                    self.fixed_ranks,
                    self.borrowed_ranks,
                )
                if self._preserve_borrowed_requests:
                    self._kv_peers.update(
                        initialize_pairs(
                            self._request_pool.addresses,
                            self.borrowed_ranks,
                            self.fixed_ranks,
                        )
                    )

    async def _async_initialize(
        self,
        job: Job,
        server_args: dict[str, Any],
        server_infos: list[LocalInfServerInfo] | None = None,
        *args,
        **kwargs,
    ):
        # Create workers via scheduler
        logger.info("Creating workers via scheduler...")
        jobs = [job]
        if self._borrowed_pool is not None:
            jobs = [
                replace(
                    job,
                    replicas=len(self.borrowed_ranks),
                    tasks=job.tasks[: len(self.borrowed_ranks)],
                ),
                replace(
                    job,
                    role=f"{job.role}-fixed",
                    replicas=len(self.fixed_ranks),
                    tasks=job.tasks[len(self.borrowed_ranks) :],
                    scheduling_strategy=SchedulingStrategy(type="separation"),
                ),
            ]
        worker_ids = []
        try:
            for part in jobs:
                self._worker_roles.append(part.role)
                worker_ids.extend(self.scheduler.create_workers(job=part))
                self.workers.extend(self.scheduler.get_workers(role=part.role))
        except Exception:
            self._delete_worker_roles()
            raise
        logger.info(f"Workers created: {worker_ids}")

        # Wait for workers to be ready
        logger.info("Waiting for workers to be ready...")
        logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Get engine class path for dynamic import on workers
        engine_class = self.inf_engine

        # Create and initialize engines on workers
        logger.info("Creating engines...")
        tasks = [
            self.scheduler.create_engine(
                worker_id=worker.id,
                engine=f"{engine_class.__module__}.{engine_class.__name__}",
                engine_name=self._engine_name(rank),
                config=self.config,
            )
            for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)
        logger.info("Engine created on all workers!")

        logger.info("Calling engine initialization...")
        # Workers are controller-managed: the controller handles staleness
        # globally, so workers must NOT apply their own dp-scaled staleness
        # constraints. Force train_data_parallel_size=1 unless explicitly
        # configured; an explicit None must not survive, or workers fall back
        # to dividing capacity by dist.get_world_size().
        if kwargs.get("train_data_parallel_size") is None:
            kwargs["train_data_parallel_size"] = 1
        if server_infos is not None:
            # Connecting to existing local servers for evaluation
            self.server_infos = server_infos
            assert len(self.server_infos) == len(self.workers), (
                len(self.server_infos),
                len(self.workers),
            )
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._engine_name(rank),
                    # args in `engine_api`
                    engine_id=str(rank),
                    addr=f"{info.host}:{info.port}",
                    engine_rank=rank,
                    num_engines=len(self.workers),
                    *args,
                    **kwargs,
                )
                for rank, (worker, info) in enumerate(
                    zip(self.workers, self.server_infos)
                )
            ]
            await asyncio.gather(*tasks)
        else:
            self.server_infos = await self._collective_rpc_async(
                "launch_server", server_args=server_args
            )
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._engine_name(rank),
                    # args in `engine_api`
                    engine_id=str(rank),
                    engine_rank=rank,
                    num_engines=len(self.workers),
                    *args,
                    **kwargs,
                )
                for rank, worker in enumerate(self.workers)
            ]
            await asyncio.gather(*tasks)

        logger.info("All engines are initialized...")

    def _delete_worker_roles(self) -> None:
        failed = []
        for role in reversed(self._worker_roles):
            try:
                self.scheduler.delete_workers(role=role)
            except Exception:
                failed.append(role)
                logger.error(f"Error deleting {role}: {traceback.format_exc()}")
        self._worker_roles = list(reversed(failed))
        if not failed:
            self.workers.clear()

    def destroy(self):
        try:
            self.wait_shared_publication()
        except Exception:
            logger.exception("Shared publication failed before teardown")
        try:
            self.wait_running_rebalance()
        except Exception:
            logger.exception("Running handoff failed before teardown")
        # Stop background threads and shutdown the async task runner
        if self._dispatcher is not None:
            self._dispatcher.destroy()

        self._stop_callback_server()

        try:
            self._collective_rpc("destroy", http_timeout=60.0)
        except Exception:
            logger.error(f"Error destroying engines: {traceback.format_exc()}")
        finally:
            self._delete_worker_roles()

        # Delete proxy workers if initialized
        if self._proxy_started:
            try:
                self.scheduler.delete_workers(role=self._proxy_role)
                self.proxy_workers.clear()
                self.proxy_addrs.clear()
                self._proxy_started = False
                logger.info("Proxy workers deleted")
            except Exception:
                logger.error(f"Error deleting proxy workers: {traceback.format_exc()}")

        # Shutdown proxy gateway if initialized
        self._stop_proxy_gateway()
        with self._futures_lock:
            self._pending_futures.clear()
            self._generation_workers.clear()

    def start_proxy(self) -> None:
        """Initialize proxy workers for AgentWorkflow support.

        Creates proxy workers colocated with rollout workers. Each proxy worker
        runs a ProxyRolloutServer that connects to the same inference server
        as its corresponding rollout worker.
        """
        if self._proxy_started:
            logger.warning("Proxy workers already initialized")
            return

        if not self.server_infos:
            raise RuntimeError(
                "Cannot initialize proxy workers: rollout not initialized. "
                "Call initialize() first."
            )

        run_async_task(self._async_start_proxy)
        self._proxy_started = True

    async def _async_start_proxy(self) -> None:
        """Async implementation of proxy worker initialization."""
        command = "areal.experimental.openai.proxy.proxy_rollout_server"
        job = Job(
            role=self._proxy_role,
            replicas=len(self.server_infos),
            tasks=[SchedulingSpec(gpu=0, port_count=2)],
            scheduling_strategy=SchedulingStrategy(
                type="colocation",
                target=self._worker_role,
                fork=True,
            ),
        )
        worker_ids = self.scheduler.fork_workers(job, command=command)
        logger.info(f"Proxy workers forked: {worker_ids}")

        self.proxy_workers = self.scheduler.get_workers(role=self._proxy_role)
        logger.info(f"Proxy workers: {[w.id for w in self.proxy_workers]}")

        engine_class = f"{self.inf_engine.__module__}.{self.inf_engine.__name__}"

        create_tasks = []
        for rank, worker in enumerate(self.proxy_workers):
            create_tasks.append(
                self.scheduler.create_engine(
                    worker_id=worker.id,
                    engine=engine_class,
                    engine_name=self._proxy_engine_name(rank),
                    config=self.config,
                )
            )
        await asyncio.gather(*create_tasks)
        logger.info("Proxy engines created")

        init_tasks = []
        for rank, (worker, server_info) in enumerate(
            zip(self.proxy_workers, self.server_infos, strict=True)
        ):
            init_tasks.append(
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._proxy_engine_name(rank),
                    addr=f"{server_info.host}:{server_info.port}",
                )
            )
            self.proxy_addrs.append(
                f"http://{format_hostport(worker.ip, int(worker.worker_ports[0]))}"
            )
        await asyncio.gather(*init_tasks)

        logger.info(f"Proxy servers initialized. Addresses: {self.proxy_addrs}")

    def get_proxy_addr(self, rank: int) -> str:
        """Get the proxy server address for a given rollout worker rank.

        Parameters
        ----------
        rank : int
            The rank of the rollout worker

        Returns
        -------
        str
            The HTTP address of the corresponding proxy server
        """
        if not self._proxy_started:
            raise RuntimeError(
                "Proxy workers not initialized. Call start_proxy() first."
            )
        if rank >= len(self.proxy_addrs):
            raise IndexError(
                f"Invalid rank {rank}, only {len(self.proxy_addrs)} proxy workers"
            )
        return self.proxy_addrs[rank]

    def start_proxy_gateway(self) -> None:
        """Start the proxy gateway for external access.

        Creates a FastAPI server that routes requests to backend proxy
        workers. Requires ``start_proxy()`` to have been called first.
        """
        if not self._proxy_started:
            raise RuntimeError(
                "Proxy workers not initialized. Call start_proxy() first."
            )
        if self._proxy_gateway_host is not None:
            logger.warning("Proxy gateway already running")
            return

        from areal.experimental.openai.proxy.proxy_gateway import (
            create_proxy_gateway_app,
        )

        agent_cfg = self.config.agent

        app = create_proxy_gateway_app(
            proxy_addrs=self.proxy_addrs,
            admin_api_key=agent_cfg.admin_api_key
            if agent_cfg is not None
            else "areal-admin-key",
        )

        self._proxy_gateway_port = find_free_ports(1)[0]
        self._proxy_gateway_host = gethostip()
        self._proxy_gateway_app = app

        def serve():
            import uvicorn

            try:
                config = uvicorn.Config(
                    app,
                    host="0.0.0.0",
                    port=self._proxy_gateway_port,
                    log_level="warning",
                    access_log=False,
                )
                server = uvicorn.Server(config)
                self._proxy_gateway_server = server
                server.run()
            except Exception:
                logger.error("Proxy gateway thread crashed", exc_info=True)

        self._proxy_gateway_thread = threading.Thread(target=serve, daemon=True)
        self._proxy_gateway_thread.start()

        # Wait for uvicorn to bind the port before propagating the address
        # to worker engines via collective RPC.
        import time

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if (
                self._proxy_gateway_server is not None
                and self._proxy_gateway_server.started
            ):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                "Proxy gateway failed to start within 10s. "
                f"Cannot propagate address "
                f"{self._proxy_gateway_host}:{self._proxy_gateway_port}"
            )

        logger.info(
            "Proxy gateway started on "
            f"{self._proxy_gateway_host}:{self._proxy_gateway_port}"
        )

        # Propagate proxy_gateway_addr to all rollout worker engines
        # so that _resolve_workflow can pick it up for online mode.
        self._collective_rpc(
            "set_proxy_gateway_addr",
            addr=self.proxy_gateway_addr,
        )

    @property
    def proxy_gateway_addr(self) -> str:
        """Single URL for external users."""
        if self._proxy_gateway_host is None:
            raise RuntimeError("Proxy gateway not started")
        return f"http://{format_hostport(self._proxy_gateway_host, self._proxy_gateway_port)}"

    def _stop_proxy_gateway(self) -> None:
        """Stop the proxy gateway server if running."""
        if self._proxy_gateway_host is None:
            return
        logger.info("Stopping proxy gateway...")
        if self._proxy_gateway_server is not None:
            self._proxy_gateway_server.should_exit = True
        if self._proxy_gateway_thread is not None:
            self._proxy_gateway_thread.join(timeout=30.0)
            if self._proxy_gateway_thread.is_alive():
                logger.warning(
                    "Proxy gateway thread did not exit within 30s; "
                    "daemon thread will be killed on process exit"
                )
        self._proxy_gateway_app = None
        self._proxy_gateway_server = None
        self._proxy_gateway_thread = None
        self._proxy_gateway_port = None
        self._proxy_gateway_host = None

    def _run_callback_control(self, coroutine):
        # Control callbacks share one asyncio loop, while request lease callbacks
        # must remain serviceable during a blocking pause/drain operation.
        with self._callback_control_lock:
            return self._callback_loop.run_until_complete(coroutine)

    def _start_callback_server(self):
        """Start Flask HTTP server to receive callbacks from RolloutCallback."""
        if self._callback_server is not None:
            logger.warning("Callback server already running")
            return

        app = Flask(__name__)
        app.logger.disabled = True

        @app.route("/callback/request_route", methods=["POST"])
        def request_route():
            payload = request.get_json()
            pool = self._request_pool
            if pool is None:
                raise RuntimeError("Request routing is disabled")
            action = payload.pop("action")
            if action == "push_endpoint":
                return jsonify(
                    self._route_push_server.url if self._route_push_server else None
                )
            if action == "acquire":
                return jsonify(pool.acquire(**payload))
            if action == "release":
                pool.release(**payload)
            elif action == "cancel":
                pool.cancel_wait(**payload)
            elif action == "finish_group":
                pool.finish_group(**payload)
            else:
                raise ValueError("Unknown request route action")
            if pool.error:
                self.dispatcher.fail_fast(RuntimeError(pool.error))
            return jsonify(True)

        @app.route("/callback/init_weights_group", methods=["POST"])
        def init_weights_group():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            self._run_callback_control(self.init_weights_update_group(meta))
            return jsonify({"status": "ok"})

        @app.route("/callback/update_weights_xccl", methods=["POST"])
        def update_weights():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            param_specs = deserialize_value(payload.get("param_specs"))
            self._run_callback_control(
                self.update_weights_from_distributed(meta, param_specs)
            )
            return jsonify({"status": "ok"})

        @app.route("/callback/update_weights_disk", methods=["POST"])
        def update_weights_disk():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            self._run_callback_control(self.update_weights_from_disk(meta))
            return jsonify({"status": "ok"})

        @app.route("/callback/update_weights_awex", methods=["POST"])
        def update_weights_awex():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            step_id = payload.get("step_id")
            kwargs = deserialize_value(payload.get("kwargs"))
            self._run_callback_control(
                self.update_weights_from_awex(meta, step_id=step_id, kwargs=kwargs)
            )
            return jsonify({"status": "ok"})

        @app.route("/callback/pause_generation", methods=["POST"])
        def pause_generation():
            self._run_callback_control(self.pause_generation())
            return jsonify({"status": "ok"})

        @app.route("/callback/continue_generation", methods=["POST"])
        def continue_generation():
            self._run_callback_control(self.continue_generation())
            return jsonify({"status": "ok"})

        @app.route("/callback/rollout_complete", methods=["POST"])
        def rollout_complete():
            payload = request.get_json() or {}
            task_id = payload.get("task_id")
            try:
                if payload.get("event") == "generation_complete":
                    if self.staleness_manager.on_generation_complete(task_id):
                        with self._futures_lock:
                            self._generation_workers.pop(task_id, None)
                        with self.dispatcher._input_cv:
                            self.dispatcher._input_cv.notify_all()
                        self._pool_event("generation_slot_released", task_id=task_id)
                else:
                    self._resolve_task_future(task_id)
                return jsonify({"status": "ok"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.errorhandler(Exception)
        def handle_error(e):
            logger.error(f"Callback handler error: {e}")
            return jsonify({"error": str(e)}), 500

        self._callback_port = find_free_ports(1)[0]
        self._callback_host = gethostip()
        self._callback_app = app
        if (
            self._request_pool is not None
            and os.getenv("AREAL_ROUTE_TRANSPORT", "push") != "poll"
        ):
            from areal.infra.request_push import PushRouteServer

            self._route_push_server = PushRouteServer(
                self._request_pool, self._callback_host
            )
            self._route_push_server.start()
        self._callback_server = make_server(
            self._callback_host, self._callback_port, app, threaded=True
        )

        # Suppress Werkzeug access logs (e.g., "POST /callback/rollout_complete 200 -")
        # Override log_request directly on the request handler class
        self._callback_server.RequestHandlerClass.log_request = (
            lambda self, *args, **kwargs: None
        )

        # Also configure Werkzeug logger level for any other log messages
        import logging as stdlib_logging

        werkzeug_logger = stdlib_logging.getLogger("werkzeug")
        werkzeug_logger.setLevel(stdlib_logging.WARNING)

        def serve_forever():
            # Create and set event loop for this thread
            self._callback_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._callback_loop)
            # Signal that the loop is ready
            self._callback_loop_ready.set()
            logger.info(
                f"Callback server started on {format_hostport(self._callback_host, self._callback_port)}"
            )
            self._callback_server.serve_forever()

        self._callback_server_thread = threading.Thread(
            target=serve_forever, daemon=True
        )
        self._callback_server_thread.start()
        # Wait for loop to be created
        self._callback_loop_ready.wait()

    def _stop_callback_server(self):
        """Stop the callback server if running."""
        if self._route_push_server is not None:
            self._route_push_server.stop()
            self._route_push_server = None
        if self._callback_server is not None:
            logger.info("Stopping callback server...")
            self._callback_server.shutdown()
            if self._callback_loop is not None:
                self._callback_loop.close()
            self._callback_server = None
            self._callback_app = None
            self._callback_server_thread = None
            self._callback_port = None
            self._callback_host = None
            self._callback_loop = None
            self._callback_loop_ready.clear()

    @property
    def callback_addr(self) -> str:
        """Return callback server address as 'host:port'."""
        if self._callback_host is None or self._callback_port is None:
            raise RuntimeError("Callback server not started")
        return format_hostport(self._callback_host, self._callback_port)

    def _resolve_task_future(self, task_id: int):
        """Resolve a pending future with the task result."""
        with self._futures_lock:
            future = self._pending_futures.pop(task_id, None)
        if future:
            future.get_loop().call_soon_threadsafe(future.set_result, None)

    def _collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        return run_async_task(self._collective_rpc_async, method, *args, **kwargs)

    async def _collective_rpc_async(self, method: str, *args, **kwargs) -> list[Any]:
        return await self._generic_collective_rpc_async(
            method, self.workers, self._engine_name, *args, **kwargs
        )

    def _proxy_collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        return run_async_task(self._proxy_collective_rpc_async, method, *args, **kwargs)

    async def _proxy_collective_rpc_async(
        self, method: str, *args, **kwargs
    ) -> list[Any]:
        return await self._generic_collective_rpc_async(
            method, self.proxy_workers, self._proxy_engine_name, *args, **kwargs
        )

    async def _generic_collective_rpc_async(
        self,
        method: str,
        workers: list[Worker],
        engine_name_fn: Callable[[int], str],
        *args,
        **kwargs,
    ) -> list[Any]:
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method=method,
                engine_name=engine_name_fn(rank),
                *args,
                **kwargs,
            )
            for rank, worker in enumerate(workers)
        ]
        return await asyncio.gather(*tasks)

    def _concurrency_snapshot(self):
        pool = getattr(self, "_request_pool", None)
        if pool is None:
            return dict(
                ready=[True] * len(self.fixed_ranks),
                waiting=0,
                borrowed_available=False,
            )
        state = pool.admission_snapshot(self.get_version())
        borrowed = self._borrowed_pool
        with borrowed._lock:
            state["borrowed_available"] = borrowed.state == "inference"
        return state

    def _choose_worker(self) -> tuple[Worker, int]:
        """Choose a least-loaded eligible worker, rotating ties.

        Returns
        -------
        tuple[Worker, int]
            The chosen worker object and its rank
        """
        if not self.workers:
            raise RuntimeError("No workers available to choose from.")
        with self._futures_lock:
            loads = {
                rank: list(self._generation_workers.values()).count(rank)
                for rank in range(len(self.workers))
            }
        if self._borrowed_pool is not None:
            accepted = self.staleness_manager.get_stats().accepted
            rank, withdrawn = self._borrowed_pool.choose(accepted, loads)
            if withdrawn:
                self._pool_event(
                    "withdraw", ranks=list(self.borrowed_ranks), accepted=accepted
                )
            return self.workers[rank], rank
        ranks = list(range(len(self.workers)))
        rotated = ranks[self._current_worker_idx :] + ranks[: self._current_worker_idx]
        rank = min(rotated, key=lambda r: loads[r])
        self._current_worker_idx = (rank + 1) % len(self.workers)
        return self.workers[rank], rank

    def _resolve_workflow_str(self, workflow: WorkflowLike | None) -> str | None:
        """Resolve workflow to a string import path.

        Handles RolloutWorkflow, agent workflow instances/classes, string paths,
        and ``None`` (online mode).
        """
        # None workflow = online mode (config-driven)
        if workflow is None:
            return None

        # String paths - return as-is
        if isinstance(workflow, str):
            return workflow

        # RolloutWorkflow classes
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            return f"{workflow.__module__}.{workflow.__name__}"

        # RolloutWorkflow instances
        elif isinstance(workflow, RolloutWorkflow):
            return f"{workflow.__module__}.{workflow.__class__.__name__}"

        # Agent-like workflow classes
        elif isinstance(workflow, type):
            return f"{workflow.__module__}.{workflow.__name__}"

        # Agent-like workflow instances
        else:
            return f"{workflow.__module__}.{workflow.__class__.__name__}"

    def _resolve_should_accept_fn(
        self, should_accept_fn: Callable[[dict[str, Any]], bool] | str | None
    ):
        if callable(should_accept_fn):
            raise RuntimeError(
                "If given, `should_accept_fn` must be an importable string path, e.g., 'my_module.filter_func'."
            )
        if should_accept_fn is not None:
            try:
                import_from_string(should_accept_fn)
            except Exception:
                raise RuntimeError(
                    f"Failed to import `should_accept_fn` from string path: {should_accept_fn}"
                )
        return should_accept_fn

    def _rollout_stats(self) -> str:
        stats = self._staleness_manager.get_stats()
        return (
            f"enqueued: {stats.enqueued}, "
            f"running: {stats.running}, "
            f"accepted: {stats.accepted}, "
            f"rejected: {stats.rejected}."
        )

    def _create_submit_callback(self, pending_task: _RemoteRolloutTaskInput):
        async def _submit_then_wait() -> _RemoteRolloutResult | None:
            # Choose by active generation load, excluding pending rewards.
            worker, rank = self._choose_worker()
            with self._futures_lock:
                self._generation_workers[pending_task.task_id] = rank
            engine_name = self._engine_name(rank)
            if self._borrowed_pool is not None:
                self._pool_event("dispatch", rank=rank, task_id=pending_task.task_id)

            # NOTE: No need to call `on_rollout_submitted` here.
            # This function will be passed to `BatchTaskDispather` where
            # `on_rollout_submitted` will be called upon dispatching
            task_id = pending_task.task_id

            manager = self.staleness_manager

            try:
                # Set future for this task
                future = asyncio.get_event_loop().create_future()
                with self._futures_lock:
                    self._pending_futures[task_id] = future

                proxy_addr = pending_task.proxy_addr
                if self._proxy_started and proxy_addr is None:
                    proxy_addr = self.get_proxy_addr(rank)
                engine_task_id = await self.scheduler.async_call_engine(
                    worker.id,
                    "submit",
                    engine_name=engine_name,
                    data=pending_task.data,
                    workflow=pending_task.workflow,
                    workflow_kwargs=pending_task.workflow_kwargs,
                    should_accept_fn=pending_task.should_accept_fn,
                    http_timeout=self.config.request_timeout,
                    is_eval=pending_task.is_eval,
                    group_size=pending_task.group_size,
                    task_id=task_id,
                    callback_addr=f"http://{self.callback_addr}/callback/rollout_complete",
                    proxy_addr=proxy_addr,
                )

                assert task_id == engine_task_id, (task_id, engine_task_id)

                # Wait for callback to resolve the future
                await asyncio.wait_for(future, timeout=self.config.request_timeout)

                # Fetch the result
                result = await self.scheduler.async_call_engine(
                    worker.id,
                    "wait_for_task",
                    engine_name=engine_name,
                    task_id=engine_task_id,
                    timeout=0.1,  # A short time to prevent blocking other requests
                    raise_timeout=False,
                    http_timeout=self.config.request_timeout,
                )

                traj = result
                if traj is not None:
                    manager.on_rollout_accepted(task_id)
                    wakeup = getattr(self, "_batch_drain_wakeup", None)
                    if wakeup is not None:
                        wakeup.set()
                    if self.config.enable_rollout_tracing:
                        logger.info(
                            f"Finish and accept rollout. {self._rollout_stats()}"
                        )
                    return _RemoteRolloutResult(task_id=task_id, trajectory=traj)

                manager.on_rollout_rejected(task_id)
                if self.config.enable_rollout_tracing:
                    logger.info(f"Finish but reject rollout. {self._rollout_stats()}")
                return None

            except asyncio.CancelledError:
                manager.on_rollout_rejected(task_id)
                with self._futures_lock:
                    self._pending_futures.pop(task_id, None)
                raise
            except TimeoutError:
                if task_id is not None:
                    with self._futures_lock:
                        self._pending_futures.pop(task_id, None)
                manager.on_rollout_rejected(task_id)
                logger.error(f"Rollout timed out after {self.config.request_timeout}s")
                return None
            except Exception as exc:
                if task_id is not None:
                    with self._futures_lock:
                        self._pending_futures.pop(task_id, None)
                manager.on_rollout_rejected(task_id)
                logger.error("Workflow execution failed: %s", exc, exc_info=True)
                return None
            finally:
                with self._futures_lock:
                    self._generation_workers.pop(task_id, None)

        return _submit_then_wait

    def get_capacity(self):
        return self.staleness_manager.get_capacity()

    def submit(
        self,
        data: dict[str, Any],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        task_id: int | None = None,
        is_eval: bool = False,
        group_size: int = 1,
        proxy_addr: str | None = None,
    ) -> int:
        workflow_str = self._resolve_workflow_str(workflow)
        should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        if workflow_kwargs is None:
            workflow_kwargs = {}

        # NOTE: RolloutController does not support `should_accept_fn`
        # If the workflow's result should be aborted,
        # `arun_episode` should return None instead.
        if task_id is None:
            task_id = self._task_id_generator.next()
        task_input = _RemoteRolloutTaskInput(
            data=data,
            workflow=workflow_str,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            task_id=task_id,
            is_eval=is_eval,
            group_size=group_size,
            proxy_addr=proxy_addr,
        )

        # Delegate to dispatcher
        self.dispatcher.submit_task_input(task_input)
        return task_id

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        # Delegate to dispatcher and extract trajectories
        results = self.dispatcher.wait_results(count, timeout, raise_timeout)
        # Log and trace
        if self.config.enable_rollout_tracing:
            logger.info("Rollout results are ready!")

        return [r.trajectory if r is not None else None for r in results]

    @trace_perf("rollout_controller.rollout_batch", category="scheduler")
    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        group_size: int = 1,
    ) -> list[dict[str, Any]]:
        perf_tracer.instant(
            "rollout_controller.rollout_batch",
            category="scheduler",
            args={"data": len(data)},
        )
        for item in data:
            self.submit(
                data=item,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
                group_size=group_size,
            )
        results = self.wait(count=len(data))
        # Return list of trajectories
        return [r for r in results if r is not None]

    @trace_perf("rollout_controller.prepare_batch", category="scheduler")
    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> list[dict[str, Any]]:
        """Prepare a batch with controlled staleness.

        Continuously submits from dataloader and waits for results, ensuring at least
        two batches are pending to maximize overlap.

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for parameters.
        """

        workflow_str = self._resolve_workflow_str(workflow)
        if workflow_kwargs is None:
            workflow_kwargs = {}

        def task_input_generator():
            for data in cycle_dataloader(dataloader):
                for item in data:
                    yield _RemoteRolloutTaskInput(
                        data=item,
                        workflow=workflow_str,
                        workflow_kwargs=workflow_kwargs,
                        should_accept_fn=should_accept_fn,
                        task_id=self._task_id_generator.next(),
                        group_size=group_size,
                    )

        if not hasattr(self, "data_generator"):
            self.data_generator = task_input_generator()

        # Delegate to dispatcher
        assert dataloader.batch_size is not None
        results = self.dispatcher.active_submit_and_wait(
            self.data_generator, batch_size=dataloader.batch_size, dynamic_bs=dynamic_bs
        )

        if self._borrowed_pool is not None:
            self._borrowed_pool.batch_ready(len(results))
            self._pool_event("batch_ready", consumed=self._borrowed_pool.consumed)

        if request_trace.enabled():
            request_trace.emit(
                "batch_consumed",
                batch=getattr(self, "_trace_training_batch", None),
                task_ids=[r.task_id for r in results if r is not None],
            )
        # Return list of trajectories
        trajectories = [r.trajectory if r is not None else None for r in results]
        return [t for t in trajectories if t is not None]

    def compute_logp(self, data: list[dict[str, Any]]) -> list[Any]:
        """Compute token log-probabilities for trajectories via remote workers."""
        if len(data) == 0:
            return []

        async def _compute():
            indexed_chunks: list[list[int]] = []
            tasks = []
            n_workers = len(self.workers)
            if n_workers == 0:
                raise RuntimeError("No workers available for compute_logp.")

            for rank, worker in enumerate(self.workers):
                idxs = list(range(rank, len(data), n_workers))
                if not idxs:
                    continue
                chunk = [data[i] for i in idxs]
                indexed_chunks.append(idxs)
                tasks.append(
                    self.scheduler.async_call_engine(
                        worker_id=worker.id,
                        method="compute_logp",
                        engine_name=self._engine_name(rank),
                        data=chunk,
                        http_timeout=self.config.request_timeout,
                    )
                )
            rpc_results = await asyncio.gather(*tasks)
            merged: list[Any] = [None] * len(data)
            for idxs, chunk_result in zip(indexed_chunks, rpc_results):
                if len(chunk_result) != len(idxs):
                    raise RuntimeError(
                        f"compute_logp result length mismatch: got {len(chunk_result)}, expected {len(idxs)}"
                    )
                for out_idx, value in zip(idxs, chunk_result):
                    merged[out_idx] = value
            return merged

        return run_async_task(_compute)

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        This method provides direct access to the inference engine's generation capabilities
        for single requests, bypassing the workflow system.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        # Choose worker and delegate
        worker, rank = self._choose_worker()

        # Call agenerate on engine via scheduler
        return await self.scheduler.async_call_engine(
            worker_id=worker.id,
            method="agenerate",
            engine_name=self._engine_name(rank),
            req=req,
        )

    async def init_weights_update_group(self, meta: WeightUpdateMeta) -> None:
        if meta.npu_share_fixed:
            if (
                self._borrowed_pool is None
                or meta.gen_allocation.world_size
                != len(self.fixed_ranks) * self.rollout_alloc.parallel.tp_size
            ):
                raise ValueError(
                    "Fixed publication allocation must match the dedicated device group"
                )
            await asyncio.gather(
                *[
                    self.scheduler.async_call_engine(
                        worker_id=self.workers[rank].id,
                        engine_name=self._engine_name(rank),
                        method="init_weights_update_group",
                        meta=meta,
                        xccl_group_ranks=[index],
                    )
                    for index, rank in enumerate(self.fixed_ranks)
                ]
            )
            return
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method="init_weights_update_group",
                engine_name=self._engine_name(rank),
                meta=meta,
                xccl_group_ranks=[rank],
            )
            for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)

    async def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ):
        if meta.npu_share_fixed:
            await asyncio.gather(
                *[
                    self.scheduler.async_call_engine(
                        worker_id=self.workers[rank].id,
                        engine_name=self._engine_name(rank),
                        method="update_weights_from_distributed",
                        meta=meta,
                        param_specs=param_specs,
                    )
                    for rank in self.fixed_ranks
                ]
            )
        else:
            await self._collective_rpc_async(
                "update_weights_from_distributed", meta=meta, param_specs=param_specs
            )

    async def update_weights_from_disk(self, meta: WeightUpdateMeta):
        meta.clear_checkpoint_after_load = False
        await self._collective_rpc_async("update_weights_from_disk", meta=meta)
        shutil.rmtree(meta.path, ignore_errors=True)

    async def update_weights_from_awex(
        self,
        meta: WeightUpdateMeta,
        step_id: int | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        await self._collective_rpc_async(
            "update_weights_from_awex", meta=meta, step_id=step_id, kwargs=kwargs
        )

    async def pause_generation(self):
        await asyncio.to_thread(self.wait_shared_publication)
        await asyncio.to_thread(self.wait_running_rebalance)
        pool = getattr(self, "_request_pool", None)
        if pool is not None:
            for rank in range(len(self.workers)):
                pool.close(rank)
        await self._collective_rpc_async("pause_generation")
        if pool is not None:
            for rank in range(len(self.workers)):
                await asyncio.to_thread(pool.wait_empty, rank)

    async def continue_generation(self):
        await self._collective_rpc_async("continue_generation")
        pool = getattr(self, "_request_pool", None)
        if pool is not None:
            for rank in range(len(self.workers)):
                pool.open(rank, self.get_version())

    def colocation_memory(
        self,
        action: str = "status",
        target_free_bytes: int | None = None,
        kv_budget_bytes: int | None = None,
        device_rank: int | None = None,
    ):
        kwargs = {"action": action}
        if kv_budget_bytes is not None:
            kwargs["kv_budget_bytes"] = kv_budget_bytes
        if target_free_bytes is not None:
            kwargs["target_free_bytes"] = target_free_bytes
        if device_rank is not None:
            tp = self.rollout_alloc.parallel.tp_size
            replica, member = divmod(device_rank, tp)
            if self._borrowed_pool is not None and replica not in self.borrowed_ranks:
                raise ValueError(
                    "Memory operation targeted a dedicated inference device"
                )
            result = self._worker_rpc(
                replica, "colocation_memory", member_rank=member, **kwargs
            )
            while isinstance(result, list) and len(result) == 1:
                result = result[0]
            return result[member] if isinstance(result, list) else result
        if self._borrowed_pool is not None:
            from areal.trainer.colocation_memory import single_status

            return single_status(
                self._group_rpc(self.borrowed_ranks, "colocation_memory", **kwargs)
            )
        return self._collective_rpc("colocation_memory", **kwargs)

    def offload(self) -> None:
        """Offload rollout model memory on all inference workers."""
        self._collective_rpc("offload")

    def onload(self, tags: list[str] | None = None) -> None:
        """Onload rollout model memory on all inference workers."""
        self._collective_rpc("onload", tags=tags)

    def set_version(self, version: int) -> None:
        with self._version_lock:
            self._version = version
            self._collective_rpc("set_version", version=version, http_timeout=60.0)
            pool = getattr(self, "_request_pool", None)
            if pool is not None:
                for rank in range(len(self.workers)):
                    pool.set_version(rank, version)
            if self._proxy_started:
                self._proxy_collective_rpc(
                    "set_version", version=version, http_timeout=60.0
                )

    def get_version(self) -> int:
        with self._version_lock:
            return self._version

    def pause(self):
        self.wait_shared_publication()
        self.wait_running_rebalance()
        self.dispatcher.pause()
        pool = getattr(self, "_request_pool", None)
        if pool is not None:
            for rank in range(len(self.workers)):
                pool.close(rank)
        # Proxy workers hold their own engines against the same inference
        # servers. Without pausing them too they keep issuing generation
        # requests during the weight-update window, while pause_generation is
        # clearing those servers' caches â€” which corrupts version attribution
        # and, for VLM rollouts, kills the engine on a stale multimodal hash.
        # Pause them before the rollout workers so that the fewest requests are
        # still in flight once pause_generation aborts and clears.
        if self._proxy_started:
            self._proxy_collective_rpc("pause", http_timeout=60.0)
        self._collective_rpc("pause", http_timeout=60.0)

    def resume(self):
        if getattr(self, "_shared_held_for_training", False):
            self._group_rpc(self.fixed_ranks, "resume", http_timeout=60.0)
            self.dispatcher.resume()
            return
        if self._borrowed_pool is not None and self._shared_publication is not None:
            with self._shared_publication_lock:
                if self._shared_publication.done():
                    self._shared_publication.result()
                if self._shared_loading:
                    self._group_rpc(self.fixed_ranks, "resume", http_timeout=60.0)
                else:
                    self._collective_rpc("resume", http_timeout=60.0)
                self.dispatcher.resume()
            return
        self._collective_rpc("resume", http_timeout=60.0)
        if self._proxy_started:
            self._proxy_collective_rpc("resume", http_timeout=60.0)
        if self._borrowed_pool is not None:
            self._borrowed_pool.release(self.staleness_manager.get_stats().accepted)
            self._pool_event(
                "returned",
                ranks=list(self.borrowed_ranks),
                version=self.get_version(),
                routing_state=self._borrowed_pool.state,
            )
        self.dispatcher.resume()

    def export_stats(self) -> dict[str, float]:
        all_raw_stats = self._collective_rpc(method="export_stats", http_timeout=60.0)
        stats = defaultdict(float)
        counts = defaultdict(int)

        for raw_stats in all_raw_stats:
            for k, v in raw_stats.items():
                if k.endswith("__count"):
                    counts[k] += v
                else:
                    stats[k] += v * raw_stats.get(k + "__count", 0)

        # Average non-count stats
        final_stats = {}
        for k, v in stats.items():
            count_key = k + "__count"
            if count_key in counts and counts[count_key] > 0:
                final_stats[k] = v / counts[count_key]
        return final_stats

    def config_perf_tracer(self, config: PerfTracerConfig, role: str) -> None:
        async def _call():
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="config_perf_tracer",
                    engine_name=self._engine_name(rank),
                    rank=rank,
                    role=role,
                    config=config,
                )
                for rank, worker in enumerate(self.workers)
            ]
            return await asyncio.gather(*tasks)

        run_async_task(_call)

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        self._collective_rpc("save_perf_tracer", step=step, force=force)

    @property
    def staleness_manager(self):
        return self._staleness_manager

    @property
    def dispatcher(
        self,
    ) -> BatchTaskDispatcher[_RemoteRolloutTaskInput, _RemoteRolloutResult]:
        """Get the task dispatcher, ensuring initialization has been called."""
        if self._dispatcher is None:
            raise RuntimeError(
                "RolloutController.initialize() must be called before scheduling rollouts."
            )
        return self._dispatcher

    @property
    def runner(self):
        """For backward compatibility. The runner is now owned by the dispatcher."""
        return self.dispatcher.runner
