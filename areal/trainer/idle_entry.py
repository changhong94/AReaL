# SPDX-License-Identifier: Apache-2.0
"""Training-layout selection from observations owned by this budget instance."""

import math
from collections import deque


class EntryBudget:
    """No persisted history, speedup prior, or backend-specific preference.

    One observation is evidence. Missing evidence remains unknown, never zero.
    Layout identifiers are supplied by the executor, not inferred from a backend.
    """

    def __init__(self, baseline_layout=1, candidate_layouts=(1, 2)):
        if baseline_layout not in candidate_layouts or len(
            set(candidate_layouts)
        ) != len(candidate_layouts):
            raise ValueError("Invalid candidate layouts")
        self.baseline_layout = baseline_layout
        self.candidate_layouts = tuple(candidate_layouts)
        self.samples = deque()
        self.drain_samples = deque()

    def observe(self, batch, world, phase, seconds):
        if world not in self.candidate_layouts or phase not in (
            "compute",
            "prepare",
            "return",
            "tail",
            "handoff",
        ):
            raise ValueError("Invalid entry observation")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid observation duration")
        self.samples.append((batch, world, phase, seconds))
        self._trim(batch)

    def observe_drain(self, batch, world, requests, seconds):
        if (
            world not in self.candidate_layouts
            or requests < 0
            or not isinstance(requests, int)
        ):
            raise ValueError("Invalid drain context")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid drain duration")
        self.drain_samples.append((batch, world, requests, seconds))
        self._trim(batch)

    def _trim(self, batch):
        # A recency window is not a minimum sample requirement.
        self.samples = deque(s for s in self.samples if s[0] >= batch - 3)
        self.drain_samples = deque(s for s in self.drain_samples if s[0] >= batch - 3)

    def cost_samples(self, batch):
        self._trim(batch)
        return list(self.samples)

    def estimate(self, batch, world, phase):
        values = [s[3] for s in self.cost_samples(batch) if s[1:3] == (world, phase)]
        return sum(values) / len(values) if values else None

    def decide_paths(
        self,
        batch,
        *,
        current_layout,
        remaining_prepare=None,
        drain_work=None,
        include_tail=False,
        include_handoff=False,
        publication_required=True,
    ):
        self._trim(batch)
        if current_layout not in self.candidate_layouts:
            raise ValueError("Unavailable current layout")
        if remaining_prepare is not None:
            if set(remaining_prepare) != set(self.candidate_layouts) or any(
                v is not None and (not math.isfinite(v) or v < 0)
                for v in remaining_prepare.values()
            ):
                raise ValueError("Invalid remaining preparation costs")
        if drain_work is not None and (
            set(drain_work) != set(self.candidate_layouts)
            or any(
                v is not None and (not isinstance(v, int) or v < 0)
                for v in drain_work.values()
            )
        ):
            raise ValueError("Invalid drain contexts")
        candidates, missing = {}, {}
        for world in self.candidate_layouts:
            prepare = (
                remaining_prepare[world]
                if remaining_prepare is not None
                else self.estimate(batch, world, "prepare")
            )
            publication = (
                self.estimate(batch, world, "return") if publication_required else 0.0
            )
            tail = (
                self.estimate(batch, world, "tail")
                if include_tail and publication_required
                else 0.0
            )
            requests = drain_work[world] if drain_work is not None else 0
            # An empty queue has no old requests to drain; never reuse a busy sample.
            values = [s[3] for s in self.drain_samples if s[1:3] == (world, requests)]
            drain = (
                (sum(values) / len(values) if values else None)
                if requests != 0
                else 0.0
            )
            costs = dict(
                remaining_prepare=prepare,
                handoff=self.estimate(batch, world, "handoff")
                if include_handoff
                else 0.0,
                compute=self.estimate(batch, world, "compute"),
                publication=publication,
                publication_tail=tail,
                drain=drain,
            )
            absent = [key for key, value in costs.items() if value is None]
            candidates[world] = dict(
                costs, total=None if absent else sum(costs.values())
            )
            if absent:
                missing[world] = absent
        baseline = self.baseline_layout
        selected, reason = baseline, "insufficient_current_run_evidence"
        if baseline not in missing:
            eligible = [
                world for world in self.candidate_layouts if world not in missing
            ]
            # Ties keep the original path rather than create an extra transition.
            best = min(
                eligible,
                key=lambda world: (candidates[world]["total"], world != baseline),
            )
            if candidates[best]["total"] < candidates[baseline]["total"]:
                selected, reason = best, "positive_measured_gain"
            else:
                reason = "no_measured_gain" if not missing else reason
        return dict(
            world_size=selected,
            current_world=current_layout,
            reason=reason,
            candidates=candidates,
            missing=missing,
            rate_source="current_run",
            baseline_layout=baseline,
        )

    def decide(self, batch, runnable, terminal=False, remaining_prepare=None):
        if runnable:
            return dict(
                world_size=self.baseline_layout,
                reason="runnable_inference",
                runnable=runnable,
                rate_source="current_run",
            )
        return self.decide_paths(
            batch,
            current_layout=self.baseline_layout,
            remaining_prepare=remaining_prepare,
        )
