# SPDX-License-Identifier: Apache-2.0

"""Incremental budget for selecting a prebuilt training engine before a batch."""

import math
from collections import deque


class EntryBudget:
    """Separate batch-entry observations from partial-gradient handoff costs.

    Preparation samples are exposed waits, not summed rank service times. The
    single-card path is the baseline; common restoration is never charged twice.
    """

    def __init__(self, speedup=1.5):
        if not math.isfinite(speedup) or not 1 < speedup <= 2:
            raise ValueError("Invalid entry speedup prior")
        self.speedup = speedup
        self.samples = deque()
        self.last_cost = {}

    def observe(self, batch, world, phase, seconds):
        if world not in (1, 2) or phase not in ("compute", "prepare", "return"):
            raise ValueError("Invalid entry observation")
        if math.isfinite(seconds) and seconds >= 0:
            self.samples.append((batch, world, phase, seconds))
            if phase in ("prepare", "return"):
                self.last_cost[(world, phase)] = (batch, world, phase, seconds)
        self._trim(batch)

    def _trim(self, batch):
        while self.samples and self.samples[0][0] < batch - 3:
            self.samples.popleft()

    def cost_samples(self, batch):
        """Use recent measurements, retaining one last-known missing cost."""
        self._trim(batch)
        result = list(self.samples)
        for key, sample in self.last_cost.items():
            if not any(s[1:3] == key for s in result):
                result.append(sample)
        return result

    def decide(self, batch, runnable, terminal=False):
        self._trim(batch)
        costs = self.cost_samples(batch)

        def mean(world, phase):
            observations = self.samples if phase == "compute" else costs
            values = [s[3] for s in observations if s[1:3] == (world, phase)]
            return sum(values) / len(values) if values else None

        if runnable:
            return dict(world_size=1, reason="runnable_inference", runnable=runnable)
        single, dual = mean(1, "compute"), mean(2, "compute")
        if single is None and dual is None:
            return dict(world_size=2, reason="idle_cold_start", rate_source="prior")
        measured = single is not None and dual is not None
        if single is None:
            single = dual * self.speedup
        if dual is None:
            dual = single / self.speedup
        prep1, prep2 = mean(1, "prepare"), mean(2, "prepare")
        # Unknown incremental exposure is explicitly marked, not confused with
        # the old 15-second full mid-update round trip.
        extra_prepare = max(0.0, (prep2 or 0.0) - (prep1 or 0.0))
        extra_return = (
            0.0
            if terminal
            else max(0.0, (mean(2, "return") or 0.0) - (mean(1, "return") or 0.0))
        )
        net = single - dual - extra_prepare - extra_return
        return dict(
            world_size=2 if net > 0 else 1,
            reason="positive_incremental_gain"
            if net > 0
            else "incremental_budget_exhausted",
            single_seconds=single,
            dual_seconds=dual,
            incremental_prepare_seconds=extra_prepare,
            incremental_return_seconds=extra_return,
            net_seconds=net,
            rate_source="measured" if measured else "prior",
            publication_source="terminal_skipped"
            if terminal
            else "last_known_fallback"
            if any(s[0] < batch - 3 and s[2] == "return" for s in costs)
            else "window_or_cold",
            preparation_source="measured"
            if prep1 is not None and prep2 is not None
            else "partial_or_cold",
        )
