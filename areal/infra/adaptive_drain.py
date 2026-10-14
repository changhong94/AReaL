# SPDX-License-Identifier: Apache-2.0
"""Conservative online estimates for whole-replica drain decisions.

Uses past completed lengths conditional on progress, not future answer lengths.
ETAs are estimates, not guarantees; event stops always override them.
"""

import statistics


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def residual_tokens(generated, limit, lengths):
    room = max(0, limit - generated)
    surviving = [n - generated for n in lengths if n > generated]
    if len(surviving) >= 4:
        return min(room, quantile(surviving, 0.75)), True
    # Surviving long requests are right-censored. Never infer zero remaining
    # from the absence of a longer completed sample.
    return room, False


def engine_stats(snapshot):
    rows = snapshot["history"] + snapshot["active"]
    rates = [
        (r["tokens"] - 1) / (r["last"] - r["first"])
        for r in rows
        if r["tokens"] > 1 and r["first"] is not None and r["last"] > r["first"]
    ]
    firsts = [r for r in rows if r["first"] is not None and r["prompt_tokens"] > 0]
    if len(rates) < 4 or len(firsts) < 4:
        return None
    # Fastest-quartile observations approximate low-queue effective prefill.
    unit = quantile(
        [(r["first"] - r["queued"]) / r["prompt_tokens"] for r in firsts], 0.25
    )
    queue = statistics.median(
        max(0, r["first"] - r["queued"] - unit * r["prompt_tokens"]) for r in firsts
    )
    running = sum(r["tokens"] > 0 for r in snapshot["active"])
    waiting = [r for r in snapshot["active"] if not r["tokens"]]
    queue += sum(r["prompt_tokens"] * unit for r in waiting) / max(1, running)
    return dict(
        rate=statistics.median(rates), prefill_unit=unit, queue=queue, running=running
    )


def estimate_drain(route, engines, needed_questions, turnaround):
    if needed_questions <= 0:
        return dict(action="stop", reason="next_batch_ready")
    stats = [engine_stats(s) for s in engines]
    if any(s is None for s in stats) or len(route["lengths"]) < 8:
        return dict(action="wait", reason="insufficient_history")
    live = [{r["request_id"]: r for r in s["active"]} for s in engines]
    requests = {r["request_id"]: dict(r, rank=1) for r in route["waiting"]}
    requests.update({r["request_id"]: r for r in route["active"]})
    migrating = sum(r["rank"] == 0 for r in route["active"])
    # Moving the entire replica's outstanding work competes with card 2's
    # existing work; do not extrapolate its unloaded per-request rate.
    contention = (max(1, stats[1]["running"]) + migrating) / max(1, stats[1]["running"])
    target_rate = stats[1]["rate"] / contention
    questions = {}
    low_confidence = 0
    for rid, lease in requests.items():
        meta = lease.get("metrics") or {}
        q = meta.get("question_id")
        if q is None:
            return dict(action="wait", reason="missing_question_metadata")
        rank = lease["rank"]
        progress = live[rank].get(rid, {})
        segment_tokens = progress.get("tokens", 0)
        generated = meta["generated_tokens"] + segment_tokens
        remaining, confident = residual_tokens(
            generated, meta["max_new_tokens"], route["lengths"]
        )
        low_confidence += not confident
        rate = stats[rank]["rate"]
        if segment_tokens > 8 and progress["last"] > progress["first"]:
            rate = (segment_tokens - 1) / (progress["last"] - progress["first"])
        rebuild = (meta["prompt_tokens"] + generated) * stats[1]["prefill_unit"]
        stay = remaining / max(rate, 1e-6)
        if not segment_tokens:
            stay += (
                stats[rank]["queue"]
                + (meta["prompt_tokens"] + generated) * stats[rank]["prefill_unit"]
            )
        if rank == 0:
            move = stats[1]["queue"] + rebuild + remaining / max(target_rate, 1e-6)
        else:
            move = stay * contention
        row = questions.setdefault(
            q,
            dict(size=lease["group_size"], ids=set(), stay=[], stay_lower=[], move=[]),
        )
        row["ids"].add(rid)
        row["stay"].append(stay)
        row["move"].append(move)
        # Unknown tails use zero remaining work in the optimistic stay case
        # and the token limit in the pessimistic migration case. They must not
        # make migration appear attractive solely because their length is unknown.
        row["stay_lower"].append(stay if confident else 0.0)
    stay_questions = []
    stay_lower_questions = []
    move_questions = []
    for q, row in questions.items():
        done = set(route["completed"].get(q, []))
        if len(row["ids"] | done) != row["size"]:
            continue  # Unknown/unsubmitted answers cannot be assumed finished.
        stay_questions.append(max(row["stay"]))
        stay_lower_questions.append(max(row["stay_lower"]))
        move_questions.append(max(row["move"]))
    if len(stay_questions) < needed_questions:
        return dict(
            action="wait",
            reason="incomplete_question_horizon",
            known_questions=len(stay_questions),
        )
    stay_eta = sorted(stay_questions)[needed_questions - 1] + turnaround
    stay_lower_eta = sorted(stay_lower_questions)[needed_questions - 1] + turnaround
    move_eta = max(sorted(move_questions)[needed_questions - 1], turnaround)
    # Require a material margin, then controller requires two consecutive votes.
    better = move_eta < stay_lower_eta * 0.85
    return dict(
        action="stop" if better else "wait",
        reason="migration_faster" if better else "stay_faster_or_uncertain",
        stay_eta=stay_eta,
        stay_lower_eta=stay_lower_eta,
        move_eta=move_eta,
        needed_questions=needed_questions,
        low_confidence_requests=low_confidence,
        migrating=migrating,
        target_contention=contention,
        turnaround=turnaround,
    )
