# SPDX-License-Identifier: Apache-2.0
"""FIFO admission for generation segments, independent of workflow ownership."""

import asyncio
import os
import time
from collections import OrderedDict
from threading import Lock

import aiohttp

from areal.utils import request_trace


class RequestLeasePool:
    def __init__(self, addresses: list[str], capacity: int, task_affinity: bool = True):
        if not addresses or capacity < 1:
            raise ValueError("Request routing needs replicas and positive capacity")
        self.addresses = addresses
        self.capacity = capacity
        self.ready = [True] * len(addresses)
        self.versions = [0] * len(addresses)
        self.waiting = OrderedDict()
        self.active = {}
        self.serial = 0
        self.error = None
        self.lock = Lock()
        self.task_affinity = task_affinity
        self.groups = {}
        self.completed = {}
        self.lengths = []
        self.window_lengths = []
        self.window_observations = []
        self.batch_metrics = os.getenv("AREAL_BATCH_DRAIN_METRICS", "0") == "1"
        self.window_migrations = []
        self.interrupted = {}
        self.rebalance_reservations = {}

    def acquire(
        self,
        ticket: str,
        request_id: str,
        group_id: str | None = None,
        group_size: int = 1,
        metrics: dict | None = None,
    ):
        if group_size < 1:
            raise ValueError("Group size must be positive")
        group_id = group_id if self.task_affinity else None
        item = dict(
            request_id=request_id,
            group_id=group_id,
            group_size=group_size,
            metrics=metrics or {},
        )
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if ticket in self.active:
                if any(self.active[ticket][k] != v for k, v in item.items()):
                    raise ValueError("Ticket metadata changed")
                return self.active[ticket].copy()
            if self.waiting.setdefault(ticket, item) != item:
                raise ValueError("Ticket metadata changed")
            counts = [
                sum(x["rank"] == r for x in self.active.values())
                + sum(x["rank"] == r for x in self.rebalance_reservations.values())
                for r in range(len(self.addresses))
            ]
            available = [
                r
                for r in range(len(counts))
                if self.ready[r] and counts[r] < self.capacity
            ]
            group_load = [
                sum(g["size"] for g in self.groups.values() if g["rank"] == r)
                for r in range(len(counts))
            ]
            rank = None
            # Oldest eligible ticket wins. A full preferred replica must not
            # block unrelated groups that can use the other replica.
            for waiting_ticket, pending in self.waiting.items():
                if any(
                    x["request_id"] == pending["request_id"]
                    for x in self.active.values()
                ):
                    continue
                group = self.groups.get(pending["group_id"])
                if group and group["size"] != pending["group_size"]:
                    raise ValueError("Group size changed")
                reservation = self.rebalance_reservations.get(pending["request_id"])
                if reservation is not None:
                    r = reservation["rank"]
                    candidates = (
                        [r] if self.ready[r] and counts[r] <= self.capacity else []
                    )
                elif group and self.ready[group["rank"]]:
                    candidates = [group["rank"]] if group["rank"] in available else []
                else:
                    candidates = available
                if not candidates:
                    continue
                if waiting_ticket != ticket:
                    return None
                rank = min(
                    candidates,
                    key=lambda r: (
                        group_load[r] if pending["group_id"] is not None else counts[r],
                        counts[r],
                        (r - self.serial) % len(counts),
                    ),
                )
                break
            if rank is None:
                return None
            if group_id is not None:
                previous = self.groups.get(group_id)
                if previous is None or previous["rank"] != rank:
                    request_trace.emit(
                        "route_group_assign",
                        group_id=group_id,
                        group_size=group_size,
                        rank=rank,
                        previous_rank=previous["rank"] if previous else None,
                    )
                self.groups[group_id] = dict(rank=rank, size=group_size)
            self.serial += 1
            lease = dict(
                ticket=ticket,
                lease_id=self.serial,
                request_id=request_id,
                group_id=group_id,
                group_size=group_size,
                rank=rank,
                server=self.addresses[rank],
                version=self.versions[rank],
                metrics=metrics or {},
            )
            self.waiting.pop(ticket)
            self.rebalance_reservations.pop(request_id, None)
            self.active[ticket] = lease
            previous = self.interrupted.pop(request_id, None)
            if previous is not None and previous[0] != rank:
                self.window_migrations.append(time.monotonic() - previous[1])
            request_trace.emit("route_acquire", **lease)
            return lease.copy()

    def finish_group(self, group_id: str):
        with self.lock:
            if any(x.get("group_id") == group_id for x in self.active.values()) or any(
                x.get("group_id") == group_id for x in self.waiting.values()
            ):
                raise RuntimeError("Group still has outstanding generation requests")
            self.groups.pop(group_id, None)
            self.completed.pop(group_id, None)
            request_trace.emit("route_group_finished", group_id=group_id)

    def release(
        self,
        ticket: str,
        lease_id: int,
        uncertain: bool = False,
        observation: dict | None = None,
    ):
        with self.lock:
            lease = self.active.get(ticket)
            if lease is None or lease["lease_id"] != lease_id:
                return  # Idempotent release, never free a newer generation.
            if uncertain:
                self.error = "Generation transport failed with unknown execution state"
                request_trace.emit("route_uncertain", **lease)
                return  # Fail closed: no second execution and no unsafe weight update.
            if observation and observation.get("finished"):
                self.rebalance_reservations.pop(lease["request_id"], None)
                question = lease.get("metrics", {}).get("question_id")
                if question is not None:
                    self.completed.setdefault(question, set()).add(lease["request_id"])
                terminal = int(observation["total_tokens"]) < lease.get(
                    "metrics", {}
                ).get("max_new_tokens", float("inf"))
                self.lengths.append(int(observation["total_tokens"]))
                if self.batch_metrics:
                    self.window_lengths.append(int(observation["total_tokens"]))
                    self.window_observations.append(
                        dict(
                            request_id=lease["request_id"],
                            question_id=question,
                            length=int(observation["total_tokens"]),
                            done=True,
                            event=terminal,
                        )
                    )
                self.lengths = self.lengths[-256:]
            elif observation and self.batch_metrics:
                self.interrupted[lease["request_id"]] = (
                    lease["rank"],
                    time.monotonic(),
                )
            del self.active[ticket]
            request_trace.emit("route_release", **lease)

    def cancel_wait(self, ticket: str):
        with self.lock:
            pending = self.waiting.pop(ticket, None)
            if pending:
                self.rebalance_reservations.pop(pending["request_id"], None)
            if ticket in self.active:
                self.error = "Request cancelled with an outstanding lease"

    def close(self, rank: int):
        with self.lock:
            self.ready[rank] = False
            self.rebalance_reservations = {
                rid: r
                for rid, r in self.rebalance_reservations.items()
                if r["rank"] != rank
            }

    def open(self, rank: int, version: int):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if any(x["rank"] == rank for x in self.active.values()):
                raise RuntimeError(
                    "Cannot change replica epoch with outstanding leases"
                )
            self.versions[rank] = version
            self.ready[rank] = True

    def set_version(self, rank: int, version: int):
        with self.lock:
            if any(x["rank"] == rank for x in self.active.values()):
                raise RuntimeError("Version change requires drained requests")
            self.versions[rank] = version

    def drain_snapshot(self, rotate=False) -> dict:
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            lengths = list(self.window_lengths)
            migrations = list(self.window_migrations)
            observations = list(self.window_observations)
            if rotate:
                self.window_lengths.clear()
                self.window_migrations.clear()
                self.window_observations.clear()
            return dict(
                capacity=self.capacity,
                reserved=len(self.rebalance_reservations),
                window_lengths=lengths,
                window_migrations=migrations,
                window_observations=observations,
                active=[dict(x) for x in self.active.values()],
                waiting=[dict(x) for x in self.waiting.values()],
                completed={k: list(v) for k, v in self.completed.items()},
                lengths=list(self.lengths),
            )

    def reserve_rebalance(self, request_ids, version):
        """Reserve target slots without releasing any source execution lease."""
        with self.lock:
            if self.error or not all(self.ready) or self.versions != [version, version]:
                return {}
            if self.task_affinity or self.waiting:
                return {}  # Do not break affinity or displace existing FIFO waiters.
            room = (
                self.capacity
                - sum(r["rank"] == 0 for r in self.active.values())
                - len(self.rebalance_reservations)
            )
            chosen = {}
            for rid, expected_lease in request_ids.items():
                leases = [
                    r
                    for r in self.active.values()
                    if r["request_id"] == rid and r["rank"] == 1
                ]
                if room <= 0:
                    break
                if (
                    len(leases) != 1
                    or leases[0]["lease_id"] != expected_lease
                    or leases[0]["version"] != version
                    or rid in self.rebalance_reservations
                ):
                    continue
                lease = leases[0]
                self.rebalance_reservations[rid] = dict(
                    rank=0, version=version, lease_id=lease["lease_id"]
                )
                chosen[rid] = lease["lease_id"]
                room -= 1
            return chosen

    def cancel_rebalance(self, reservations):
        with self.lock:
            for rid, lease_id in reservations.items():
                current = self.rebalance_reservations.get(rid)
                if current and current["lease_id"] == lease_id:
                    del self.rebalance_reservations[rid]

    def active_requests(self, rank: int) -> list[str]:
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            return [x["request_id"] for x in self.active.values() if x["rank"] == rank]

    def drain_until(self, rank: int, deadline: float, should_stop) -> str:
        """Wait only for existing leases; caller must close admission first."""
        with self.lock:
            if self.ready[rank]:
                raise RuntimeError("Drain requires closed admission")
        while True:
            if not self.active_requests(rank):
                return "empty"
            if should_stop():
                return "next_batch_ready"
            if time.monotonic() >= deadline:
                return "timeout"
            time.sleep(0.01)

    def wait_empty(self, rank: int, timeout: float = 60.0):
        deadline = time.monotonic() + timeout
        while True:
            with self.lock:
                if self.error:
                    raise RuntimeError(self.error)
                if not any(x["rank"] == rank for x in self.active.values()):
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Generation leases did not drain before model transition"
                )
            time.sleep(0.01)


async def route_call(session, address: str, action: str, **payload):
    async with session.post(
        f"http://{address}/callback/request_route",
        json=dict(action=action, **payload),
        timeout=aiohttp.ClientTimeout(total=60),
    ) as response:
        response.raise_for_status()
        return await response.json()


async def acquire_route(
    session,
    address: str,
    ticket: str,
    request_id: str,
    group_id: str | None = None,
    group_size: int = 1,
    metrics: dict | None = None,
):
    try:
        while True:
            lease = await route_call(
                session,
                address,
                "acquire",
                ticket=ticket,
                request_id=request_id,
                group_id=group_id,
                group_size=group_size,
                metrics=metrics,
            )
            if lease is not None:
                return lease
            await asyncio.sleep(0.02)
    except BaseException:
        await asyncio.shield(route_call(session, address, "cancel", ticket=ticket))
        raise


def workflow_group_id(task_id: int | None, is_eval: bool) -> str | None:
    return None if task_id is None else f"{'eval' if is_eval else 'rollout'}:{task_id}"
