# SPDX-License-Identifier: Apache-2.0
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.infra.batch_drain import BatchCalibration, remaining_tokens
from areal.infra.request_router import RequestLeasePool
from areal.utils.generation_progress import ProgressRegistry


def calibration():
    engine = dict(
        active=[],
        window=dict(
            cutoff=1,
            decode=[
                dict(
                    concurrency=2, context=128, seconds=0.05, count=1, max_concurrency=2
                )
            ],
            first=[dict(seconds=0.1, prompt_tokens=100)],
        ),
    )
    route = dict(window_lengths=[100], window_migrations=[0.1], active=[], waiting=[])
    c = BatchCalibration()
    c.freeze([engine, engine], route, [2.0])
    return c


def test_one_sample_calibrates_and_empty_window_retains_costs():
    c = calibration()
    original = c.service(1, 2, 100)
    empty = dict(active=[], window=dict(cutoff=2, decode=[], first=[]))
    c.freeze([empty, empty], dict(window_lengths=[], active=[], waiting=[]), [])
    assert c.service(1, 2, 100) == original
    assert c.handoff["seconds"] == 2
    assert c.migration["seconds"] == 0.1
    assert c.handoff["source_window"] == 1


def test_uncalibrated_cost_is_unknown_not_zero():
    c = BatchCalibration()
    assert c.estimate({}, [], 1)["reason"] == "uncalibrated_metric"


def test_censored_tail_not_assumed_complete():
    assert remaining_tokens(100, 200, [(40, True), (100, False)]) == 100
    assert remaining_tokens(80, 200, [(100, True)]) == 20
    assert remaining_tokens(200, 200, [(100, False)]) == 0
    assert remaining_tokens(0, 200, [(100, True), (150, False)]) == 150


def test_freeze_copies_observations_and_no_future_completion_leak():
    c = calibration()
    source = dict(active=[], window=dict(cutoff=2, decode=[], first=[]))
    lease = dict(
        request_id="r", rank=0, metrics=dict(generated_tokens=20, max_new_tokens=100)
    )
    source["active"] = [dict(request_id="r", tokens=10)]
    route = dict(window_lengths=[80], active=[lease], waiting=[])
    c.freeze([source, source], route, [])
    route["window_lengths"].append(99)
    source["active"][0]["tokens"] = 75
    assert c.lengths == [(80, True), (30, False)]


def test_service_does_not_assume_inverse_concurrency():
    c = calibration()
    assert c.service(1, 2, 100)["seconds"] == c.service(1, 64, 100)["seconds"]


def fixture(waiting=False):
    rows = [
        dict(
            request_id=str(i),
            rank=0,
            group_size=1,
            metrics=dict(
                question_id=str(i),
                generated_tokens=0,
                prompt_tokens=100,
                max_new_tokens=200,
            ),
        )
        for i in range(4)
    ]
    route = dict(active=rows, waiting=[], completed={})
    engines = [
        dict(
            active=[
                dict(request_id=str(i), tokens=0 if waiting else 10) for i in range(4)
            ]
        ),
        dict(active=[]),
    ]
    return route, engines


def test_waiting_queue_uses_observed_width_and_full_prefill():
    c = calibration()
    route, engines = fixture(waiting=True)
    result = c.estimate(route, engines, 4)
    assert result["services"][0]["running"] == 0
    assert result["services"][0]["width"] == 2
    assert result["services"][0]["waiting"] == 4
    assert result["stay_data_ready"] == pytest.approx(10.2)
    assert result["stay_card_ready"] == pytest.approx(12.2)


def test_objective_uses_max_and_no_fixed_margin():
    c = calibration()
    route, engines = fixture()
    result = c.estimate(route, engines, 1)
    assert result["stay_eta"] == max(
        result["stay_data_ready"], result["stay_card_ready"]
    )
    assert result["move_eta"] == max(
        result["move_data_ready"], result["move_card_ready"]
    )
    assert result["action"] == (
        "stop" if result["move_eta"] < result["stay_eta"] else "wait"
    )
    assert result["kappa"] == 0


def test_registry_window_excludes_prefill_pause_and_old_intervals(monkeypatch):
    monkeypatch.setenv("AREAL_BATCH_DRAIN_METRICS", "1")
    now = [0.0]
    monkeypatch.setattr(
        "areal.utils.generation_progress.time.monotonic", lambda: now[0]
    )
    r = ProgressRegistry()
    r.add("i", "r", 100)
    output = [SimpleNamespace(request_id="i", new_token_ids=[1])]
    r.observe_frame(output)  # prefill
    r.advance("i", 1)
    now[0] = 1
    r.observe_frame(output)
    now[0] = 1.05
    r.observe_frame(output)
    r.break_service_interval()
    now[0] = 10
    r.observe_frame(output)
    snap = r.snapshot(rotate=True)
    assert snap["window"]["decode"][0]["count"] == 1
    assert snap["window"]["decode"][0]["seconds"] == pytest.approx(0.05)
    assert r.snapshot(rotate=True)["window"]["decode"] == []


def test_router_window_rotation_and_migration_clock(monkeypatch):
    monkeypatch.setenv("AREAL_BATCH_DRAIN_METRICS", "1")
    pool = RequestLeasePool(["a", "b"], 8, False)
    first = pool.acquire("t", "r", metrics={"question_id": "q"})
    pool.release(
        "t", first["lease_id"], observation=dict(finished=False, total_tokens=10)
    )
    pool.close(first["rank"])
    second = pool.acquire("t2", "r", metrics={"question_id": "q"})
    pool.release(
        "t2", second["lease_id"], observation=dict(finished=True, total_tokens=30)
    )
    snap = pool.drain_snapshot(rotate=True)
    assert snap["window_lengths"] == [30]
    assert len(snap["window_migrations"]) == 1
    assert pool.drain_snapshot(rotate=True)["window_lengths"] == []


def test_batch_mode_stops_on_first_cost_vote(monkeypatch):
    import requests

    from areal.infra.controller.rollout_controller import RolloutController

    monkeypatch.setenv("AREAL_BATCH_DRAIN", "1")
    c = object.__new__(RolloutController)
    c.rollout_alloc = SimpleNamespace(parallel=SimpleNamespace(dp_size=2))
    c._request_pool = RequestLeasePool(["a", "b"], 8)
    c._request_pool.acquire("t", "r")
    c.config = SimpleNamespace(request_timeout=60, max_head_offpolicyness=2)
    c.get_version = Mock(return_value=1)
    c.training_prefetch_ready = Mock(return_value=False)
    c.training_prefetch_target = Mock(return_value=15)
    c._staleness_manager = Mock()
    c._staleness_manager.get_stats.return_value.accepted = 0
    c._batch_calibration = Mock()
    c._batch_calibration.estimate.return_value = dict(
        action="stop", reason="migration_faster"
    )
    c._batch_drain_wakeup = Event()
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    session.post.return_value.json.return_value = dict(
        budget_valid=True, generation_running=True
    )
    monkeypatch.setattr(requests, "Session", lambda: session)
    assert c._adaptive_drain_reason(16) == "migration_faster"
    assert c._batch_calibration.estimate.call_count == 1


def test_training_ready_includes_preparation_not_idle_wait(monkeypatch):
    from areal.infra.controller.rollout_controller import RolloutController

    monkeypatch.setenv("AREAL_BATCH_DRAIN_METRICS", "1")
    c = object.__new__(RolloutController)
    c._batch_publish_seconds = 1.5
    c.record_training_ready(2.5)
    assert c._batch_handoffs == [4.0]
    assert c._batch_publish_seconds is None


def test_training_ready_includes_executor_tail_and_excludes_idle(monkeypatch):
    from areal.infra.controller.rollout_controller import RolloutController

    monkeypatch.setenv("AREAL_BATCH_DRAIN_METRICS", "1")
    monkeypatch.setattr(
        "areal.infra.controller.rollout_controller.time.monotonic", lambda: 20.0
    )
    c = object.__new__(RolloutController)
    c._batch_transition_started = 10.0
    c._batch_publish_seconds = 1.5
    c.record_training_ready(2.5, idle_seconds=5.0)
    assert c._batch_handoffs == [5.0]  # Includes the additional one-second tail.


def test_publication_cycles_reset_handoff_even_without_drain(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from areal.infra.controller.rollout_controller import RolloutController

    monkeypatch.setenv("AREAL_BATCH_DRAIN_METRICS", "1")
    now = [0.0]
    monkeypatch.setattr(
        "areal.infra.controller.rollout_controller.time.monotonic", lambda: now[0]
    )
    c = object.__new__(RolloutController)
    c._borrowed_pool = Mock()
    c._request_pool = None
    c._borrowed_replica_count = 1
    c.rollout_alloc = SimpleNamespace(parallel=SimpleNamespace(dp_size=2))
    c.wait_shared_publication = Mock()
    c.wait_running_rebalance = Mock()
    c._group_rpc = Mock()
    c._pool_event = Mock()
    c.get_version = Mock(return_value=1)
    c.config = SimpleNamespace()
    for start in [10.0, 30.0, 50.0]:
        now[0] = start
        c.prepare_fixed_publication()
        assert c._batch_transition_started == start
        c._batch_publish_seconds = 1.5
        now[0] += 4.0
        c.record_training_ready(1.0, idle_seconds=1.0)
        assert c._batch_transition_started is None
    assert c._batch_handoffs == [3.0, 3.0, 3.0]
