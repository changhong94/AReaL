# SPDX-License-Identifier: Apache-2.0
import math

import pytest

from areal.infra.drain_prediction import (
    BucketService,
    exponential_rate,
    simulate_decode,
)
from areal.infra.rolling_drain import residual_distribution


def test_exponential_rate_counts_censored_exposure_not_termination():
    assert exponential_rate([(100, True), (300, False)]) == 1 / 400
    assert exponential_rate([(100, True), (300, False)], 150) is None
    assert exponential_rate([(100, False)]) is None


def test_tail_distributes_unknown_mass_below_cap_and_preserves_known_body():
    values, cdf = residual_distribution(0, 1000, [(100, True), (150, False)], 0.01)
    assert values[0] == 100 and cdf[0] == 0.5
    assert 100 < values[1] < values[-1] < 1000
    assert cdf[-1] == 1 and all(a <= b for a, b in zip(cdf, cdf[1:]))


def test_tail_at_or_beyond_observed_support_is_conditional_and_bounded():
    values, cdf = residual_distribution(200, 250, [(100, True)], 0.01)
    assert 0 < min(values) <= max(values) <= 50
    assert cdf[-1] == 1
    assert residual_distribution(250, 250, [], 0.01) == ([0.0], [1.0])
    assert residual_distribution(200, 250, [(100, False)]) == ([50.0], [1.0])


def test_fractional_observations_never_produce_negative_residual():
    values, _ = residual_distribution(10.5, 100, [(10.6, True), (11.4, True)], 0.1)
    assert min(values) > 0


def rows(lengths, context=100):
    return [dict(context=context, draws=[n], rank=1) for n in lengths]


def run(rr, running, waiting, width, service, migration=0, first=0, moving=False):
    return simulate_decode(
        rr,
        list(range(len(rr))),
        running,
        waiting,
        width,
        1,
        moving,
        0,
        migration,
        first,
        service,
    )


def test_constant_service_matches_analytic_completion_and_queue_order():
    result = run(rows([10, 20, 5]), [0, 1], [2], 2, lambda *_: 2)
    assert result == {0: 20, 1: 40, 2: 30}


def test_dynamic_concurrency_updates_all_running_requests_together():
    result = run(rows([10, 20]), [0, 1], [], 2, lambda rank, n, ctx: n)
    assert result == {0: 20, 1: 30}


def test_context_boundary_recomputes_service_during_long_request():
    result = run(
        rows([100], context=100),
        [0],
        [],
        1,
        lambda rank, n, ctx: 1 if ctx < 128 * math.sqrt(2) else 2,
    )
    boundary = 128 * math.sqrt(2)
    assert result[0] == pytest.approx((boundary - 100) + (200 - boundary) * 2, abs=1e-5)


def test_migration_arrival_and_prefill_occupy_slot_before_decode():
    rr = rows([10, 5])
    rr[1]["rank"] = 0
    result = run(rr, [0], [1], 2, lambda *_: 1, migration=20, first=0.1, moving=True)
    assert result == {0: 10, 1: 35}


def test_simultaneous_completion_is_order_independent():
    rr = rows([10, 10, 20])
    forward = run(rr, [0, 1, 2], [], 3, lambda rank, n, ctx: n)
    reverse = run(rr, [2, 1, 0], [], 3, lambda rank, n, ctx: n)
    assert forward == reverse == {0: 30, 1: 30, 2: 40}


def test_zero_remaining_and_zero_prefill_do_not_spin():
    assert run(rows([0, 0, 10]), [], [0, 1, 2], 1, lambda *_: 1) == {0: 0, 1: 0, 2: 10}


def test_cap_censor_does_not_pin_all_unknown_tail_to_cap():
    values, cdf = residual_distribution(0, 1000, [(100, True), (1000, False)], 0.01)
    assert cdf[0] == 0.5 and max(values) < 1000


def test_bucket_lookup_switches_at_fractional_and_sparse_boundaries():
    from areal.infra.batch_drain import BatchCalibration

    c = BatchCalibration()
    c.decode[1] = {
        (8, 1024): dict(seconds=1, count=1, source_window=1),
        (8, 2048): dict(seconds=2, count=1, source_window=1),
    }
    service = BucketService(c.decode, c.service)
    cut = 1024 * math.sqrt(2)
    assert int(cut - 0.01) == int(cut + 0.01)
    assert service(1, 8, cut - 0.01) == 1
    assert service(1, 8, cut + 0.01) == 2
    c.decode[1] = {
        (8, 128): dict(seconds=1, count=1, source_window=1),
        (8, 512): dict(seconds=2, count=1, source_window=1),
    }
    service = BucketService(c.decode, c.service)
    assert service.next_boundary(1, 8, 200) == 256
    assert service(1, 8, 255) == 1 and service(1, 8, 257) == 2


def test_sibling_completion_lengths_do_not_change_prediction():
    import copy

    from areal.infra.rolling_drain import RollingBatchCalibration

    c = RollingBatchCalibration()
    engine = dict(
        active=[],
        window=dict(
            cutoff=1,
            decode=[
                dict(
                    concurrency=8, max_concurrency=8, context=128, count=10, seconds=0.5
                )
            ],
            first=[dict(seconds=0, prompt_tokens=100)],
        ),
    )
    observations = [
        dict(request_id=str(i), question_id="historic", length=n, done=True, event=True)
        for i, n in enumerate([50, 100, 150, 200])
    ]
    c.freeze(
        [copy.deepcopy(engine), copy.deepcopy(engine)],
        dict(
            window_observations=observations,
            window_migrations=[0.1],
            active=[],
            waiting=[],
        ),
        [dict(tail=1, restore=0.2)],
    )
    route = dict(
        active=[
            dict(
                request_id="live",
                rank=1,
                group_size=6,
                metrics=dict(
                    question_id="q",
                    generated_tokens=10,
                    prompt_tokens=100,
                    max_new_tokens=1024,
                ),
            )
        ],
        waiting=[],
        completed={"q": ["a", "b", "c", "d", "e"]},
    )
    before = c.estimate(route, [dict(active=[]), dict(active=[])], 1)
    route["completed_observations"] = {"q": {"a": dict(length=100000, event=True)}}
    after = c.estimate(route, [dict(active=[]), dict(active=[])], 1)
    assert before == after
    assert before["estimator"] == "distribution_v4"
    assert "question_prior" not in before
    assert all(
        "scale" not in row and "same_question_completed" not in row
        for row in before["length_diagnostics"]
    )
    assert not hasattr(c, "question_prior")
