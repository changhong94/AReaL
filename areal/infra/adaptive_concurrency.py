# SPDX-License-Identifier: Apache-2.0
"""Capacity-driven soft admission for changing shared inference replica availability."""

import math
import threading
import time

from areal.utils import request_trace


class CapacityConcurrency:
    """Scale question admission by ready replicas, never by TP rank count.

    Growth is rate-limited; withdrawal is immediate for new admission only.
    This first R025 policy is capacity-driven, not a throughput predictor.
    Router/engine limits, memory budgets and staleness remain independent hard gates.
    """

    def __init__(self, baseline, fixed_replicas, snapshot, interval=1.0):
        if baseline < 1 or fixed_replicas < 1 or interval <= 0:
            raise ValueError("Concurrency parameters must be positive")
        self.baseline = baseline
        self.fixed_replicas = fixed_replicas
        self.snapshot = snapshot
        self.interval = interval
        self.target = baseline
        self.last_growth = float("-inf")
        self.lock = threading.Lock()

    def capacity_for(self, replicas: int) -> int:
        """Scale by equivalent inference replicas without a fixed multiplier cap."""
        return math.ceil(self.baseline * replicas / self.fixed_replicas)

    def __call__(self):
        with self.lock:
            state = self.snapshot()
            ready = sum(state["ready"])
            ceiling = self.capacity_for(ready)
            if not state["borrowed_available"]:
                ceiling = min(ceiling, self.baseline)
            now = time.monotonic()
            previous = self.target
            reason = "stable"
            if ceiling < self.target:
                self.target = ceiling
                self.last_growth = now
                reason = "capacity_withdrawn"
            elif ceiling > self.target and now - self.last_growth >= self.interval:
                # Router backlog blocks further growth of the question target.
                if state["waiting"] == 0:
                    self.target = min(ceiling, self.target + max(1, self.baseline // 4))
                    self.last_growth = now
                    reason = "capacity_available"
            if previous != self.target:
                request_trace.emit(
                    "adaptive_concurrency",
                    previous=previous,
                    target=self.target,
                    ceiling=ceiling,
                    ready_replicas=ready,
                    reason=reason,
                    waiting=state["waiting"],
                    baseline=self.baseline,
                )
            return self.target
