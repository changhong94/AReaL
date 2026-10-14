# SPDX-License-Identifier: Apache-2.0
import copy
import random

import pytest

from areal.infra.rolling_drain import (
    RollingBatchCalibration,
    residual_distribution,
    training_start,
)


def frame(lengths=(), censored=(), count=1, seconds=1.0, cutoff=1):
    engine = dict(
        active=[],
        window=dict(
            cutoff=cutoff,
            decode=[
                dict(
                    concurrency=8,
                    max_concurrency=8,
                    context=128,
                    count=count,
                    seconds=seconds,
                )
            ],
            first=[dict(seconds=0.0, prompt_tokens=100)],
        ),
    )
    route = dict(
        window_observations=[
            dict(request_id=rid, length=n, done=True) for rid, n in lengths
        ],
        window_migrations=[0.0],
        active=[],
        waiting=[],
    )
    for rid, n in censored:
        route["active"].append(
            dict(
                request_id=rid,
                rank=0,
                metrics=dict(generated_tokens=n, max_new_tokens=200),
            )
        )
    return [copy.deepcopy(engine), copy.deepcopy(engine)], route


def calibrated(lengths=(10, 100), parts=None):
    c = RollingBatchCalibration()
    e, r = frame(lengths=[(str(i), n) for i, n in enumerate(lengths)])
    c.freeze(e, r, parts or [dict(tail=0.0, restore=0.0)])
    return c


def requests(count=6, groups=1, rank=1):
    active = []
    for i in range(count):
        active.append(
            dict(
                request_id="r" + str(i),
                rank=rank,
                group_size=count // groups,
                metrics=dict(
                    question_id="q" + str(i % groups),
                    generated_tokens=0,
                    prompt_tokens=100,
                    max_new_tokens=200,
                ),
            )
        )
    return dict(active=active, waiting=[], completed={}), [
        dict(active=[]),
        dict(active=[]),
    ]


def test_window_uses_available_then_expires_oldest():
    c = RollingBatchCalibration()
    for i in range(1, 5):
        e, r = frame(lengths=[(str(i), i * 10)], cutoff=i)
        info = c.freeze(e, r, [dict(tail=i, restore=1)])
        assert info["window_count"] == min(i, 3)
    assert info["window_ids"] == [2, 3, 4]
    assert sorted(n for n, done in c.lengths) == [20, 30, 40]
    assert len(c.handoff_parts) == 3


def test_decode_pools_samples_not_window_means():
    c = RollingBatchCalibration()
    e, r = frame(count=1, seconds=100)
    c.freeze(e, r, [])
    e, r = frame(count=9, seconds=90)
    c.freeze(e, r, [])
    assert c.service(1, 8, 128)["seconds"] == 19
    assert c.service(1, 8, 128)["count"] == 10


def test_completed_observation_replaces_cross_window_censor_and_is_immutable():
    c = RollingBatchCalibration()
    e, r = frame(censored=[("a", 20)])
    c.freeze(e, r, [])
    e, r = frame(censored=[("a", 40)])
    c.freeze(e, r, [])
    assert c.lengths == [(40, False)]
    e, r = frame(lengths=[("a", 80)])
    c.freeze(e, r, [])
    r["window_observations"][0]["length"] = 900
    assert c.lengths == [(80, True)]
    assert c.summary["unique_requests"] == 1


def test_missing_bucket_keeps_last_valid_and_unknown_remains_unknown():
    c = calibrated()
    original = c.service(1, 8, 128)
    for i in range(4):
        e, r = frame()
        r["window_migrations"] = []
        for x in e:
            x["window"].update(decode=[], first=[])
        c.freeze(e, r, [])
    assert c.service(1, 8, 128) == original
    assert c.handoff_source == 3  # Last aggregate containing the original frame.
    empty = RollingBatchCalibration()
    assert empty.estimate({}, [], 1)["reason"] == "uncalibrated_metric"


def test_distribution_keeps_censored_probability_at_limit():
    values, cdf = residual_distribution(0, 200, [(100, True), (150, False)])
    assert values == [100.0, 200.0]
    assert cdf == [0.5, 1.0]
    assert residual_distribution(150, 200, [(100, True), (150, False)]) == (
        [50.0],
        [1.0],
    )


def test_question_max_is_taken_within_each_trial():
    c = calibrated()
    r, e = requests()
    result = c.estimate(r, e, 1)
    # E[max of six independent {10,100} lengths] = 100 - 90/64.
    assert result["move_data_ready"] == pytest.approx(98.59375, abs=4)
    assert result["move_data_ready"] > 90  # max of individual means would be 55.


def test_training_start_max_is_inside_expectation():
    c = calibrated(parts=[dict(tail=55, restore=0)])
    r, e = requests(count=1)
    result = c.estimate(r, e, 1)
    assert (
        result["move_eta"]
        > max(result["move_data_ready"], result["move_card_ready"]) + 15
    )


def test_shared_trials_are_repeatable_and_do_not_consume_global_rng():
    c = calibrated()
    r, e = requests()
    before = random.getstate()
    a = c.estimate(r, e, 1)
    b = c.estimate(r, e, 1)
    assert a == b and random.getstate() == before
    assert a["move_eta"] == a["stay_eta"]
    assert a["paired_delta_se"] == 0


def test_training_gate_delays_restore_and_one_question_batch_waits_for_answer():
    assert training_start(8, 6, 2, 0.5) == (8, 6.5)
    c = calibrated(lengths=(40,), parts=[dict(tail=2, restore=3)])
    r, e = requests(count=1)
    result = c.estimate(r, e, 1, prefetch_needed=1)
    assert result["move_eta"] == 43
    assert result["move_gate_ready"] == 40


def test_unknown_answers_do_not_appear_finished():
    c = calibrated()
    r, e = requests()
    r["active"].pop()
    assert c.estimate(r, e, 1)["reason"] == "incomplete_question_horizon"


def test_zero_conservatism_uses_mean_paired_objective():
    c = calibrated(parts=[dict(tail=2, restore=1)])
    r, e = requests(rank=0)
    result = c.estimate(r, e, 1)
    assert result["kappa"] == 0
    assert (result["action"] == "stop") == (result["move_eta"] < result["stay_eta"])
