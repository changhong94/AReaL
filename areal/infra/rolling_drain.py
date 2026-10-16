# SPDX-License-Identifier: Apache-2.0
"""Three-batch, causal distribution estimates for replica drain decisions."""

import bisect
import copy
import hashlib
import math
import random
import statistics
from collections import Counter, deque

from areal.infra.batch_drain import BatchCalibration
from areal.infra.drain_prediction import (
    BucketService,
    exponential_rate,
    simulate_decode,
)


def residual_distribution(
    generated: int, limit: int, observations: list, tail_rate=None
) -> tuple:
    """Conditional KM body plus a censored exponential tail, capped at limit.

    A missing rate retains the original conservative boundary fallback.
    """
    if generated >= limit:
        return [0.0], [1.0]
    rows = [
        (min(float(n), limit), done)
        for n, done in observations
        if n > generated or (n == generated and not done)
    ]
    if not rows:
        if tail_rate:
            return [
                min(float(limit - generated), -math.log1p(-(i + 0.5) / 16) / tail_rate)
                for i in range(16)
            ], [(i + 1) / 16 for i in range(16)]
        return [float(limit - generated)], [1.0]
    removed = Counter(n for n, _ in rows)
    deaths = Counter(n for n, done in rows if done)
    risk, survival = len(rows), 1.0
    values, cdf = [], []
    cumulative = 0.0
    for n in sorted(removed):
        mass = survival * deaths[n] / risk
        if mass:
            cumulative += mass
            values.append(float(n - generated))
            cdf.append(cumulative)
        survival *= 1.0 - deaths[n] / risk
        risk -= removed[n]
    if survival > 1e-12:
        if tail_rate:
            start = max(
                generated, max((n for n, done in rows if done), default=generated)
            )
            for i in range(16):
                values.append(
                    min(
                        float(limit - generated),
                        start - generated - math.log1p(-(i + 0.5) / 16) / tail_rate,
                    )
                )
                cdf.append(cumulative + survival * (i + 1) / 16)
        else:
            values.append(float(limit - generated))
            cdf.append(1.0)
    cdf[-1] = 1.0
    return values, cdf


def training_start(data: float, gate: float, tail: float, restore: float) -> tuple:
    card = max(tail, gate) + restore
    return max(data, card), card


class RollingBatchCalibration(BatchCalibration):
    """Freeze at PPO completion; retain up to three complete observation windows."""

    def __init__(
        self, samples: int = 256, *, tail_model=True, dynamic_service=True, replicas=2
    ):
        super().__init__()
        if replicas < 2 or replicas % 2:
            raise ValueError("Drain calibration needs equal replica groups")
        self.replicas = replicas
        self.borrowed = set(range(replicas // 2))
        self.fixed = tuple(range(replicas // 2, replicas))
        self.decode = [{} for _ in range(replicas)]
        self.first = [None] * replicas
        if samples < 2:
            raise ValueError("Distribution estimate requires at least two trials")
        self.samples = samples
        self.frames = deque(maxlen=3)
        self.handoff_parts = []
        self.handoff_source = None
        self.tail_model, self.dynamic_service = tail_model, dynamic_service
        self.tail_rate = None

    def freeze(self, engines: list, route: dict, handoffs: list) -> dict:
        self.window += 1
        observed = {r["request_id"]: dict(r) for r in route["window_observations"]}
        live = [{r["request_id"]: r for r in e["active"]} for e in engines]
        for lease in route["active"]:
            meta = lease.get("metrics") or {}
            if "max_new_tokens" not in meta:
                continue
            rid = lease["request_id"]
            # A completion racing the endpoint snapshot is absorbing.
            if rid not in observed:
                observed[rid] = dict(
                    request_id=rid,
                    question_id=meta.get("question_id"),
                    done=False,
                    length=meta["generated_tokens"]
                    + live[lease["rank"]].get(rid, {}).get("tokens", 0),
                )
        for lease in route["waiting"]:
            meta = lease.get("metrics") or {}
            rid = lease["request_id"]
            if rid not in observed and meta.get("generated_tokens", 0):
                observed[rid] = dict(
                    request_id=rid,
                    question_id=meta.get("question_id"),
                    done=False,
                    length=meta["generated_tokens"],
                )
        self.frames.append(
            copy.deepcopy(
                dict(
                    index=self.window,
                    engines=[e["window"] for e in engines],
                    observations=observed,
                    migrations=route.get("window_migrations", []),
                    handoffs=handoffs,
                )
            )
        )
        merged = {}
        for frame in self.frames:
            for rid, row in frame["observations"].items():
                prior = merged.get(rid)
                if prior is None or row["done"] or not prior["done"]:
                    merged[rid] = row
        if merged:
            self.lengths = [
                (r["length"], r.get("event", r["done"])) for r in merged.values()
            ]
            self.length_window = self.window
        if self.lengths:
            threshold = statistics.median(n for n, _ in self.lengths)
            self.tail_rate = (
                exponential_rate(self.lengths, threshold)
                or exponential_rate(self.lengths)
                or self.tail_rate
            )
        engine_counts = []
        for rank in range(self.replicas):
            bins, firsts = {}, []
            for frame in self.frames:
                profile = frame["engines"][rank]
                firsts.extend(profile["first"])
                for row in profile["decode"]:
                    if not row["count"]:
                        continue
                    key = row["concurrency"], row["context"]
                    target = bins.setdefault(
                        key,
                        dict(
                            concurrency=key[0],
                            context=key[1],
                            seconds=0.0,
                            count=0,
                            max_concurrency=0,
                            source_windows=[],
                        ),
                    )
                    target["seconds"] += row["seconds"]
                    target["count"] += row["count"]
                    target["max_concurrency"] = max(
                        target["max_concurrency"], row.get("max_concurrency", key[0])
                    )
                    target["source_windows"].append(frame["index"])
                    target["source_window"] = frame["index"]
            self.decode[rank].update(bins)  # Missing buckets retain last valid values.
            if firsts:
                self.first[rank] = dict(
                    unit=min(r["seconds"] / max(1, r["prompt_tokens"]) for r in firsts),
                    count=len(firsts),
                    source_window=self.window,
                )
            engine_counts.append(
                dict(
                    decode=sum(r["count"] for r in bins.values()),
                    first=len(firsts),
                    cutoff=engines[rank]["window"]["cutoff"],
                )
            )
        migrations = [s for f in self.frames for s in f["migrations"]]
        if migrations:
            self.migration = dict(
                seconds=statistics.median(migrations),
                count=len(migrations),
                source_window=self.window,
            )
        parts = [p for f in self.frames for p in f["handoffs"]]
        if parts:
            self.handoff_parts = parts
            self.handoff_source = self.window
            self.handoff = dict(
                seconds=statistics.mean(p["tail"] + p["restore"] for p in parts),
                count=len(parts),
                source_window=self.window,
            )
        self.summary = dict(
            window=self.window,
            window_ids=[f["index"] for f in self.frames],
            window_count=len(self.frames),
            tail_rate=self.tail_rate,
            engines=engine_counts,
            completed=sum(r["done"] for r in merged.values()),
            censored=sum(not r["done"] for r in merged.values()),
            unique_requests=len(merged),
            length_source=getattr(self, "length_window", None),
            handoff=copy.deepcopy(self.handoff),
            migration=copy.deepcopy(self.migration),
            handoff_parts=copy.deepcopy(self.handoff_parts),
        )
        return copy.deepcopy(self.summary)

    def service(self, rank: int, concurrency: int, context: float):
        result = super().service(rank, concurrency, context)
        if result is not None:
            row = self.decode[rank][result["concurrency"], result["context"]]
            result["source_windows"] = list(row["source_windows"])
        return result

    def estimate(
        self,
        route: dict,
        engines: list,
        needed: int,
        prefetch_needed: int | None = None,
    ) -> dict:
        base = dict(
            window=self.window,
            window_count=len(self.frames),
            kappa=0.0,
            needed_questions=needed,
            estimator="distribution_v4",
            samples=self.samples,
        )
        if needed <= 0:
            return dict(base, action="stop", reason="next_batch_ready")
        if (
            not self.handoff_parts
            or self.migration is None
            or not self.lengths
            or any(x is None for x in self.first)
        ):
            return dict(base, action="wait", reason="uncalibrated_metric")
        if any(not d for d in self.decode):
            return dict(base, action="wait", reason="uncalibrated_decode")
        gate_needed = min(
            needed, max(0, needed - 1 if prefetch_needed is None else prefetch_needed)
        )
        live = [{r["request_id"]: r for r in e["active"]} for e in engines]
        requests = {
            r["request_id"]: dict(r, rank=self.fixed[i % len(self.fixed)])
            for i, r in enumerate(route["waiting"])
        }
        requests.update({r["request_id"]: r for r in route["active"]})
        rows, questions = [], {}
        diagnostics = []
        for rid, lease in requests.items():
            meta = lease.get("metrics") or {}
            if not all(
                k in meta
                for k in (
                    "question_id",
                    "generated_tokens",
                    "prompt_tokens",
                    "max_new_tokens",
                )
            ):
                return dict(base, action="wait", reason="missing_question_metadata")
            progress = live[lease["rank"]].get(rid, {})
            generated = meta["generated_tokens"] + progress.get("tokens", 0)
            row = dict(
                lease,
                request_id=rid,
                progress=progress,
                context=meta["prompt_tokens"] + generated,
            )
            rate = self.tail_rate if self.tail_model else None
            values, cdf = residual_distribution(
                generated, meta["max_new_tokens"], self.lengths, rate
            )
            probs = [v - (cdf[i - 1] if i else 0.0) for i, v in enumerate(cdf)]
            diagnostics.append(
                dict(
                    request_id=rid,
                    question_id=meta["question_id"],
                    generated=generated,
                    expected_remaining=sum(v * p for v, p in zip(values, probs)),
                    cap_mass=sum(
                        p
                        for v, p in zip(values, probs)
                        if v >= meta["max_new_tokens"] - generated
                    ),
                )
            )
            seed = int.from_bytes(
                hashlib.blake2b(rid.encode(), digest_size=8).digest(), "big"
            )
            rng = random.Random(seed)  # Never consumes model/training RNG state.
            row["draws"] = [
                values[bisect.bisect_left(cdf, rng.random())]
                for _ in range(self.samples)
            ]
            idx = len(rows)
            rows.append(row)
            q = questions.setdefault(
                meta["question_id"],
                dict(size=lease["group_size"], ids=set(), indices=[]),
            )
            q["ids"].add(rid)
            q["indices"].append(idx)
        known = [
            q["indices"]
            for qid, q in questions.items()
            if len(q["ids"] | set(route["completed"].get(qid, []))) == q["size"]
        ]
        if len(known) < needed:
            return dict(
                base,
                action="wait",
                reason="incomplete_question_horizon",
                known_questions=len(known),
            )
        services = []

        def plan(indices, rank, moving):
            if not indices:
                return None
            capacity = max(
                r.get("max_concurrency", r["concurrency"])
                for r in self.decode[rank].values()
            )
            running = [
                i
                for i in indices
                if rows[i]["rank"] == rank and rows[i]["progress"].get("tokens", 0)
            ]
            waiting = [i for i in indices if i not in running]
            width = max(len(running), min(len(indices), capacity))
            svc = self.service(
                rank, width, statistics.mean(rows[i]["context"] for i in indices)
            )
            services.append(
                dict(
                    svc,
                    rank=rank,
                    running=len(running),
                    waiting=len(waiting),
                    width=width,
                )
            )
            waiting.sort(key=lambda i: rows[i]["rank"] != rank)
            return dict(
                running=running,
                waiting=waiting,
                width=width,
                tpot=svc["seconds"],
                rank=rank,
                moving=moving,
            )

        plans = [
            plan([i for i, r in enumerate(rows) if r["rank"] == rank], rank, False)
            for rank in range(self.replicas)
        ]
        assignments = {
            rank: [i for i, row in enumerate(rows) if row["rank"] == rank]
            for rank in self.fixed
        }
        # Preserve fixed work, route withdrawn work to the least loaded fixed instance.
        loads = {
            rank: sum(sum(rows[i]["draws"]) / self.samples for i in ids)
            for rank, ids in assignments.items()
        }
        for i, row in enumerate(rows):
            if row["rank"] in self.borrowed:
                rank = min(self.fixed, key=lambda rank: (loads[rank], rank))
                assignments[rank].append(i)
                loads[rank] += sum(row["draws"]) / self.samples
        move_plans = [plan(assignments[rank], rank, True) for rank in self.fixed]

        service_seconds = BucketService(self.decode, self.service)

        def schedule(p, trial, result):
            if p is None:
                return
            result.update(
                simulate_decode(
                    rows,
                    p["running"] + p["waiting"],
                    p["running"],
                    p["waiting"],
                    p["width"],
                    p["rank"],
                    p["moving"],
                    trial,
                    self.migration["seconds"],
                    self.first[p["rank"]]["unit"],
                    service_seconds,
                    self.dynamic_service,
                )
            )

        totals = {
            k: []
            for k in (
                "stay_eta",
                "move_eta",
                "stay_data_ready",
                "move_data_ready",
                "stay_card_ready",
                "move_card_ready",
                "stay_gate_ready",
                "move_gate_ready",
            )
        }
        deltas = []
        for trial in range(self.samples):
            stay, move = {}, {}
            for p in plans:
                schedule(p, trial, stay)
            for p in move_plans:
                schedule(p, trial, move)
            parts = self.handoff_parts[trial % len(self.handoff_parts)]
            for name, times in (("stay", stay), ("move", move)):
                qtimes = sorted(max(times[i] for i in q) for q in known)
                data = qtimes[needed - 1]
                gate = qtimes[gate_needed - 1] if gate_needed else 0.0
                drain = (
                    0.0
                    if name == "move"
                    else min(
                        data,
                        max(
                            (
                                times[i]
                                for i, r in enumerate(rows)
                                if r["rank"] in self.borrowed
                            ),
                            default=0.0,
                        ),
                    )
                )
                # stay stops only when empty or the full target batch is ready;
                # move routes all known old work to rank1. Neither path assumes
                # a critical known rank0 request survives training withdrawal.
                eta, card = training_start(
                    data, gate, drain + parts["tail"], parts["restore"]
                )
                for key, value in (
                    ("eta", eta),
                    ("data_ready", data),
                    ("card_ready", card),
                    ("gate_ready", gate),
                ):
                    totals[name + "_" + key].append(value)
            deltas.append(totals["stay_eta"][-1] - totals["move_eta"][-1])
        mean = {k: statistics.mean(v) for k, v in totals.items()}
        better = mean["move_eta"] < mean["stay_eta"]
        return dict(
            base,
            **mean,
            action="stop" if better else "wait",
            reason="migration_faster" if better else "stay_faster",
            services=services,
            handoff=self.handoff["seconds"],
            handoff_source=self.handoff_source,
            paired_delta_se=statistics.stdev(deltas) / math.sqrt(self.samples),
            migrating=sum(r["rank"] in self.borrowed for r in rows),
            known_questions=len(known),
            prefetch_needed=gate_needed,
            horizon="currently_known_requests",
            length_diagnostics=diagnostics,
            dynamic_services=list(service_seconds.used.values()),
            tail_rate=self.tail_rate,
        )
