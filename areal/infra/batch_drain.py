# SPDX-License-Identifier: Apache-2.0
"""Batch-frozen service estimates for old-version replica draining.

All calibration observations precede the current publication. Live progress is
read separately; an unfinished answer is censored, never retrospectively filled.
CPU output intervals measure effective service, not device-only kernel time.
"""

import copy
import heapq
import math
import statistics


def remaining_tokens(generated: int, limit: int, observations: list) -> float:
    """Restricted conditional survival mean, retaining the unobserved tail."""
    if generated >= limit:
        return 0.0
    survivors = [
        (min(int(n), limit), bool(done))
        for n, done in observations
        if n > generated or (n == generated and not done)
    ]
    if not survivors:
        return float(limit - generated)
    risk, survival, previous, area = len(survivors), 1.0, generated, 0.0
    for length in sorted({n for n, _ in survivors}):
        area += survival * (length - previous)
        deaths = sum(n == length and done for n, done in survivors)
        removed = sum(n == length for n, _ in survivors)
        survival *= 1.0 - deaths / risk
        risk -= removed
        previous = length
    return area + survival * (limit - previous)


class BatchCalibration:
    """Replace each observed metric per batch; retain only missing metrics."""

    def __init__(self):
        self.window = 0
        self.decode = [{}, {}]
        self.first = [None, None]
        self.lengths = []
        self.handoff = None
        self.migration = None
        self.summary = {}

    def freeze(self, engines: list, route: dict, handoffs: list) -> dict:
        self.window += 1
        counts = []
        for rank, engine in enumerate(engines):
            profile = engine["window"]
            for row in profile["decode"]:
                if row["count"]:
                    key = (row["concurrency"], row["context"])
                    self.decode[rank][key] = dict(row, source_window=self.window)
            if profile["first"]:
                self.first[rank] = dict(
                    unit=min(
                        r["seconds"] / max(1, r["prompt_tokens"])
                        for r in profile["first"]
                    ),
                    count=len(profile["first"]),
                    source_window=self.window,
                )
            counts.append(
                dict(
                    decode=sum(r["count"] for r in profile["decode"]),
                    first=len(profile["first"]),
                    cutoff=profile["cutoff"],
                )
            )
        lengths = [(n, True) for n in route["window_lengths"]]
        live = [{r["request_id"]: r for r in e["active"]} for e in engines]
        for lease in route["active"]:
            meta = lease.get("metrics") or {}
            if "max_new_tokens" in meta:
                n = meta["generated_tokens"] + live[lease["rank"]].get(
                    lease["request_id"], {}
                ).get("tokens", 0)
                lengths.append((n, False))
        for lease in route["waiting"]:
            meta = lease.get("metrics") or {}
            if meta.get("generated_tokens", 0):
                lengths.append((meta["generated_tokens"], False))
        if lengths:
            self.lengths = lengths
            self.length_window = self.window
        if handoffs:
            self.handoff = dict(
                seconds=statistics.median(handoffs),
                count=len(handoffs),
                source_window=self.window,
            )
        migrations = route.get("window_migrations", [])
        if migrations:
            self.migration = dict(
                seconds=statistics.median(migrations),
                count=len(migrations),
                source_window=self.window,
            )
        self.summary = dict(
            window=self.window,
            engines=counts,
            completed=len(route["window_lengths"]),
            censored=sum(not done for _, done in lengths),
            length_source=getattr(self, "length_window", None),
            handoff=copy.deepcopy(self.handoff),
            migration=copy.deepcopy(self.migration),
        )
        return copy.deepcopy(self.summary)

    def service(self, rank: int, concurrency: int, context: float):
        """Nearest observed log bucket; never impose inverse-N throughput."""
        rows = self.decode[rank]
        if not rows:
            return None
        key = min(
            rows,
            key=lambda key: (
                abs(math.log2(max(1, concurrency) / key[0]))
                + abs(math.log2(max(1, context) / key[1]))
            ),
        )
        row = rows[key]
        return dict(
            seconds=row["seconds"] / row["count"],
            concurrency=key[0],
            context=key[1],
            source_window=row["source_window"],
            count=row["count"],
        )

    def estimate(self, route: dict, engines: list, needed: int) -> dict:
        base = dict(window=self.window, kappa=0.0, needed_questions=needed)
        if needed <= 0:
            return dict(base, action="stop", reason="next_batch_ready")
        if (
            self.handoff is None
            or self.migration is None
            or not self.lengths
            or any(x is None for x in self.first)
        ):
            return dict(base, action="wait", reason="uncalibrated_metric")
        live = [{r["request_id"]: r for r in e["active"]} for e in engines]
        requests = {r["request_id"]: dict(r, rank=1) for r in route["waiting"]}
        requests.update({r["request_id"]: r for r in route["active"]})
        prepared = []
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
            prepared.append(
                dict(
                    lease,
                    request_id=rid,
                    progress=progress,
                    context=meta["prompt_tokens"] + generated,
                    remaining=remaining_tokens(
                        generated, meta["max_new_tokens"], self.lengths
                    ),
                )
            )
        if any(not rows for rows in self.decode):
            return dict(base, action="wait", reason="uncalibrated_decode")
        services = []

        def schedule(rows, rank, migrating):
            if not rows:
                return {}
            # Empirical execution width is a calibration observation, not the
            # number of admitted leases. Requests beyond it queue for a slot.
            capacity = max(
                r.get("max_concurrency", r["concurrency"])
                for r in self.decode[rank].values()
            )
            running = [
                r for r in rows if r["rank"] == rank and r["progress"].get("tokens", 0)
            ]
            waiting = [r for r in rows if r not in running]
            width = max(len(running), min(len(rows), capacity))
            service = self.service(
                rank, width, statistics.mean(r["context"] for r in rows)
            )
            services.append(
                dict(
                    service,
                    rank=rank,
                    running=len(running),
                    waiting=len(waiting),
                    width=width,
                )
            )
            result = {
                r["request_id"]: r["remaining"] * service["seconds"] for r in running
            }
            slots = list(result.values()) + [0.0] * max(0, width - len(running))
            heapq.heapify(slots)
            # Existing engine work precedes migrated work in the target queue.
            waiting.sort(key=lambda r: r["rank"] != rank)
            for row in waiting:
                start = heapq.heappop(slots)
                transfer = (
                    self.migration["seconds"]
                    if migrating and row["rank"] != rank
                    else 0.0
                )
                end = (
                    max(start, transfer)
                    + row["context"] * self.first[rank]["unit"]
                    + row["remaining"] * service["seconds"]
                )
                result[row["request_id"]] = end
                heapq.heappush(slots, end)
            return result

        stay_times = {}
        for rank in (0, 1):
            stay_times.update(
                schedule([r for r in prepared if r["rank"] == rank], rank, False)
            )
        move_times = schedule(prepared, 1, True)
        questions = {}
        for row in prepared:
            rank, meta = row["rank"], row["metrics"]
            q = questions.setdefault(
                meta["question_id"],
                dict(size=row["group_size"], ids=set(), stay=[], move=[]),
            )
            q["ids"].add(row["request_id"])
            q["stay"].append(stay_times[row["request_id"]])
            q["move"].append(move_times[row["request_id"]])
        known = [
            q
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
        data_stay = sorted(max(q["stay"]) for q in known)[needed - 1]
        data_move = sorted(max(q["move"]) for q in known)[needed - 1]
        empty = max(
            (stay_times[r["request_id"]] for r in prepared if r["rank"] == 0),
            default=0.0,
        )
        handoff = self.handoff["seconds"]
        card_stay = min(data_stay, empty) + handoff
        card_move = handoff
        j_stay, j_move = max(data_stay, card_stay), max(data_move, card_move)
        return dict(
            base,
            action="stop" if j_move < j_stay else "wait",
            reason="migration_faster" if j_move < j_stay else "stay_faster",
            stay_eta=j_stay,
            move_eta=j_move,
            stay_data_ready=data_stay,
            stay_card_ready=card_stay,
            move_data_ready=data_move,
            move_card_ready=card_move,
            handoff=handoff,
            services=services,
            migrating=sum(r["rank"] == 0 for r in prepared),
            known_questions=len(known),
        )
