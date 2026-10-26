# SPDX-License-Identifier: Apache-2.0

"""Time budget for optional mid-update DP expansion; no device operations."""

import math
from collections import deque


def dual_work(work):
    """FSDP ranks rendezvous each backward round, including the odd dummy."""
    return sum(
        max(work[i], work[i + 1] if i + 1 < len(work) else work[0])
        for i in range(0, len(work), 2)
    )


class SwitchBudget:
    def __init__(self, initial_cost=6.0, initial_speedup=1.5, margin=0.0):
        if not (
            math.isfinite(initial_cost)
            and initial_cost > 0
            and 1 < initial_speedup <= 2
            and math.isfinite(margin)
            and margin >= 0
        ):
            raise ValueError("Invalid idle switch budget priors")
        self.initial_cost = initial_cost
        self.initial_speedup = initial_speedup
        self.margin = margin
        self.samples = deque()
        self.costs = deque()

    def observe(self, batch, world, work, seconds):
        if work > 0 and seconds > 0 and math.isfinite(seconds):
            self.samples.append((batch, world, work, seconds))
        self._trim(batch)

    def cost(self, batch, direction, seconds):
        if seconds > 0 and math.isfinite(seconds):
            self.costs.append((batch, direction, seconds))
        self._trim(batch)

    def _trim(self, batch):
        # Current batch plus the previous three completed batches.
        while self.samples and self.samples[0][0] < batch - 3:
            self.samples.popleft()
        while self.costs and self.costs[0][0] < batch - 3:
            self.costs.popleft()

    def decide(self, batch, remaining, publication_samples=(), terminal=False):
        self._trim(batch)
        rates = {}
        for world in (1, 2):
            values = [s for s in self.samples if s[1] == world]
            if values:
                rates[world] = sum(s[3] for s in values) / sum(s[2] for s in values)
        if not remaining or 1 not in rates:
            return dict(accepted=False, reason="insufficient_work_or_samples")
        single = sum(remaining) * rates[1]
        # Each collective round waits for its heavier microbatch.
        dual_rate = rates.get(2, rates[1] * 2 / self.initial_speedup)
        dual = dual_work(remaining) * dual_rate
        entry = [s[2] for s in self.costs if s[1] == "in"]
        entry_cost = max(entry) if entry else self.initial_cost
        publication = {}
        stale_cost = False
        for world in (1, 2):
            all_values = [s for s in publication_samples if s[1:3] == (world, "return")]
            recent = [s for s in all_values if s[0] >= batch - 3]
            if not recent and all_values:
                recent = [max(all_values, key=lambda s: s[0])]
                stale_cost = True
            publication[world] = [s[3] for s in recent]
        # The DP2 engine publishes directly; there is no mandatory DP2 -> DP1
        # restoration. Charge only its additional publication exposure.
        extra_publication = (
            0.0
            if terminal
            else max(
                0.0, max(publication[2], default=0.0) - max(publication[1], default=0.0)
            )
        )
        cost = entry_cost + extra_publication
        net = single - dual - cost
        return dict(
            accepted=net > self.margin,
            reason="profitable" if net > self.margin else "budget_exhausted",
            single_seconds=single,
            dual_seconds=dual,
            switch_seconds=cost,
            net_seconds=net,
            margin_seconds=self.margin,
            dual_rate_source="measured" if 2 in rates else "prior",
            cost_source="measured_entry" if entry else "entry_prior",
            entry_seconds=entry_cost,
            incremental_publication_seconds=extra_publication,
            publication_source="terminal_skipped"
            if terminal
            else "last_known_fallback"
            if stale_cost
            else "measured"
            if all(publication.values())
            else "partial_or_cold",
            samples=len(self.samples),
        )


def helper_plan(remaining, seconds_per_work, restore_seconds, merge_seconds):
    """Assign a disjoint contiguous suffix prefix; primary keeps the final item.

    Restore can overlap primary backward. Gradient merge is exposed before the
    sole optimizer step. Never assume that rejected expansion implies a gain.
    """
    if len(remaining) < 2 or seconds_per_work <= 0:
        return dict(accepted=False, reason="insufficient_helper_work")
    baseline = sum(remaining) * seconds_per_work
    candidates = []
    for cut in range(1, len(remaining)):
        helper = sum(remaining[:cut]) * seconds_per_work
        primary = sum(remaining[cut:]) * seconds_per_work
        elapsed = max(primary, restore_seconds + helper) + merge_seconds
        candidates.append((elapsed, cut))
    elapsed, cut = min(candidates)
    return dict(
        accepted=elapsed < baseline,
        reason="profitable" if elapsed < baseline else "helper_cost_exceeds_gain",
        helper_microbatches=cut,
        single_seconds=baseline,
        predicted_seconds=elapsed,
        net_seconds=baseline - elapsed,
        restore_seconds=restore_seconds,
        merge_seconds=merge_seconds,
    )


def prefer_helper(expansion, helper):
    """Profitability is necessary; choose the cheaper feasible execution path."""
    return bool(
        helper.get("accepted")
        and (
            not expansion.get("accepted")
            or helper["net_seconds"] > expansion["net_seconds"]
        )
    )
