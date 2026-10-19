# SPDX-License-Identifier: Apache-2.0
"""Causal, CPU-only length calibration and event-driven decode prediction."""

import bisect
import heapq
import math
import statistics


def exponential_rate(observations: list, threshold: float = 0.0):
    """Right-censored exponential MLE above a data-derived threshold."""
    exposure = sum(max(0.0, n - threshold) for n, _ in observations)
    deaths = sum(done and n > threshold for n, done in observations)
    return deaths / exposure if deaths and exposure else None


class BucketService:
    """Exact piecewise nearest-log-bucket lookup, cached by concurrency."""

    def __init__(self, decode, resolve):
        self.decode, self.resolve = decode, resolve
        self.tables, self.used = {}, {}

    def table(self, rank, concurrency):
        key = rank, concurrency
        if key not in self.tables:
            rows = list(self.decode[rank])
            cuts = set()
            for ca, xa in rows:
                for cb, xb in rows:
                    if xa >= xb:
                        continue
                    a, b = math.log2(xa), math.log2(xb)
                    point = (
                        a
                        + b
                        + abs(math.log2(concurrency / cb))
                        - abs(math.log2(concurrency / ca))
                    ) / 2
                    if a <= point <= b:
                        cuts.add(2**point)
            cuts = sorted(cuts)
            probes = (
                (
                    [cuts[0] / 2]
                    + [math.sqrt(a * b) for a, b in zip(cuts, cuts[1:])]
                    + [cuts[-1] * 2]
                )
                if cuts
                else [1.0]
            )
            services = [self.resolve(rank, concurrency, p) for p in probes]
            self.tables[key] = cuts, services
        return self.tables[key]

    def __call__(self, rank, concurrency, context):
        cuts, services = self.table(rank, concurrency)
        row = services[bisect.bisect_right(cuts, context * (1 + 1e-12))]
        self.used[rank, row["concurrency"], row["context"]] = dict(row, rank=rank)
        return row["seconds"]

    def next_boundary(self, rank, concurrency, context):
        cuts, _ = self.table(rank, concurrency)
        idx = bisect.bisect_right(cuts, context * (1 + 1e-12))
        return cuts[idx] if idx < len(cuts) else math.inf


def simulate_decode(
    rows: list,
    indices: list,
    running: list,
    waiting: list,
    width: int,
    rank: int,
    moving: bool,
    trial: int,
    migration: float,
    first_unit: float,
    service,
    dynamic=True,
):
    """Advance all active decodes together at completion/admission/context events.

    Prefill occupies a slot until its measured approximation expires. This does
    not model device-level preemption or contention during mixed prefill/decode.
    """
    if not indices:
        return {}
    result, active, finish_heap, prefills = {}, {}, [], []
    queue = list(waiting)
    cursor, now, clock, context_base = 0, 0.0, 0.0, 0.0
    initial_tpot = service(
        rank, width, statistics.mean(rows[i]["context"] for i in indices)
    )

    def admit(i):
        nonlocal context_base
        base = float(rows[i]["context"]) - clock
        active[i] = base
        context_base += base
        heapq.heappush(finish_heap, (clock + rows[i]["draws"][trial], i))

    for i in running:
        admit(i)
    while len(result) < len(indices):
        while finish_heap and finish_heap[0][0] <= clock + 1e-8:
            _, i = heapq.heappop(finish_heap)
            context_base -= active.pop(i)
            result[i] = now
        while prefills and prefills[0][0] <= now + 1e-9:
            _, i = heapq.heappop(prefills)
            admit(i)
        while cursor < len(queue) and len(active) + len(prefills) < width:
            i = queue[cursor]
            transfer = (
                rows[i].get("handoff_ready_s", migration)
                if moving and rows[i]["rank"] != rank
                else 0.0
            )
            if transfer > now + 1e-9:
                break
            cursor += 1
            prefill = (
                rows[i].get("handoff_prefill_s", rows[i]["context"] * first_unit)
                if moving and rows[i]["rank"] != rank
                else rows[i]["context"] * first_unit
            )
            if prefill <= 1e-9:
                admit(i)
            else:
                heapq.heappush(prefills, (now + prefill, i))
        if finish_heap and finish_heap[0][0] <= clock + 1e-8:
            continue
        if len(result) == len(indices):
            break
        next_times = [prefills[0][0]] if prefills else []
        if cursor < len(queue) and len(active) + len(prefills) < width:
            i = queue[cursor]
            transfer = (
                rows[i].get("handoff_ready_s", migration)
                if moving and rows[i]["rank"] != rank
                else 0.0
            )
            if transfer > now + 1e-9:
                next_times.append(transfer)
        if active:
            mean_context = context_base / len(active) + clock
            tpot = service(rank, len(active), mean_context) if dynamic else initial_tpot
            if tpot <= 0 or not math.isfinite(tpot):
                raise ValueError("Decode duration must be finite and positive")
            next_times.append(now + (finish_heap[0][0] - clock) * tpot)
            if dynamic:
                boundary = (
                    service.next_boundary(rank, len(active), mean_context)
                    if hasattr(service, "next_boundary")
                    else 2
                    ** (math.floor(math.log2(max(1.0, mean_context)) + 0.5) + 0.5)
                )
                next_times.append(now + max(1e-6, boundary - mean_context) * tpot)
        if not next_times:
            raise RuntimeError("Decode simulation has work but no future event")
        end = min(t for t in next_times if t > now)
        if active:
            clock += (end - now) / tpot
        now = end
    return result
