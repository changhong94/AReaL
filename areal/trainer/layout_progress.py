# SPDX-License-Identifier: Apache-2.0

"""Causal plug-in estimate for the already admitted generation cohort."""

import math
from bisect import bisect_right
from collections import Counter, defaultdict


def conditional_tail(done, censored, generated, cap):
    """KM restricted to (generated, cap]; unsupported tail mass remains at cap."""
    deaths = Counter(min(cap, n) for n in done if n > generated)
    open_ = Counter(min(cap, n) for n in censored if n >= generated)
    risk = sum(deaths.values()) + sum(open_.values())
    if not risk:
        return [(cap - generated, 1.0)]
    surv = 1.0
    out = []
    for n in sorted(deaths.keys() | open_.keys()):
        if deaths[n]:
            surv *= 1.0 - deaths[n] / risk
            out.append((n - generated, 1.0 - surv))
        risk -= deaths[n] + open_[n]
    if not out or out[-1][1] < 1.0:
        out.append((cap - generated, 1.0))
    return out


def estimate_progress(route, engines, needed, group_size):
    if group_size <= 0 or needed < 0:
        raise ValueError("Invalid question horizon")
    credit = sum(
        sorted(
            (min(group_size, len(set(v))) for v in route["completed"].values()),
            reverse=True,
        )[:needed]
    )
    total = needed * group_size
    result = dict(
        unfinished_fraction=(total - credit) / total if total else 0.0,
        completed_answer_credit=credit,
        score_tail=route.get("score_tail", 0.0),
        drain_seconds=None,
        drain_method="unknown",
    )
    if not route["active"]:
        result.update(drain_seconds=0.0, drain_method="empty")
        return result
    if not engines:
        return result
    by_q = defaultdict(list)
    lengths = []
    for x in route["window_observations"]:
        if x.get("done") and x["length"] > 0:
            by_q[x["question_id"]].append(x["length"])
            lengths.append(x["length"])
    if not lengths:
        return result
    # Simultaneously active requests drain together. Integrate the CDF of
    # their maximum residual, not the maximum of their individual means.
    supports = []
    upper = 0.0
    live = [{x["request_id"]: x for x in e["active"]} for e in engines]
    open_q = defaultdict(list)
    open_all = []
    for lease in route["active"]:
        q = lease["metrics"].get("question_id")
        x = live[lease["rank"]].get(lease["request_id"])
        if x is not None:
            n = x["tokens"] + lease["metrics"]["generated_tokens"]
            open_q[q].append(n)
            open_all.append(n)
    for lease in route["active"]:
        rank = lease["rank"]
        m = lease["metrics"]
        x = live[rank].get(lease["request_id"])
        if (
            x is None
            or x["tokens"] <= 0
            or not engines[rank].get("window", {}).get("decode")
        ):
            return result  # inconsistent poll or pending prefill; retain existing estimator
        gen = x["tokens"] + m["generated_tokens"]
        cap = m["max_new_tokens"]
        if gen > cap:
            return result
        bins = engines[rank]["window"]["decode"]
        width = len(live[rank])
        nearest = min(abs(math.log2(b["concurrency"] / max(1, width))) for b in bins)
        sel = [
            b
            for b in bins
            if abs(math.log2(b["concurrency"] / max(1, width))) == nearest
        ]
        count = sum(b["count"] for b in sel)
        if not count:
            return result
        tick = sum(b["seconds"] for b in sel) / count
        q = m.get("question_id")
        siblings = by_q.get(q, ())
        # Condition on the query only if a completed sibling reached this length.
        local = any(n > gen for n in siblings)
        dist = conditional_tail(
            siblings if local else lengths, open_q[q] if local else open_all, gen, cap
        )
        points = [v * tick for v, c in dist]
        cdf = [c for v, c in dist]
        upper = max(upper, (cap - gen) * tick)
        supports.append((points, cdf))
    prev = 0.0
    area = 0.0
    cdf = 0.0
    for point in sorted(set(v for ds, _ in supports for v in ds)):
        area += (point - prev) * (1.0 - cdf)
        cdf = math.prod(
            cs[i - 1] if (i := bisect_right(ds, point)) else 0.0 for ds, cs in supports
        )
        prev = point
    return dict(
        result,
        drain_seconds=area,
        cap_at_observed_cadence=upper,
        drain_method="right_censored_cohort",
        active_samples=len(supports),
        completed_samples=len(lengths),
    )
