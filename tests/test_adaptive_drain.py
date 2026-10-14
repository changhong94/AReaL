# SPDX-License-Identifier: Apache-2.0
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.infra.adaptive_drain import estimate_drain, residual_tokens
from areal.infra.request_router import RequestLeasePool
from areal.utils.generation_progress import ProgressRegistry


def test_registry_does_not_retire_not_yet_registered_request():
    registry = ProgressRegistry()
    registry.add("internal", "generate-tokens-external", 100)
    registry.retire_missing({})
    assert registry.snapshot()["active"][0]["request_id"] == "external"
    registry.advance("internal", 2)
    registry.advance("internal", 2)
    registry.retire_missing({})
    assert not registry.snapshot()["active"]
    assert registry.snapshot()["history"][0]["tokens"] == 4


def test_question_tracking_independent_of_affinity_and_abort():
    pool = RequestLeasePool(["a", "b"], 8, task_affinity=False)
    lease = pool.acquire("t", "r", "q", 6, {"question_id": "q"})
    assert lease["group_id"] is None
    pool.release(
        "t", lease["lease_id"], observation={"finished": False, "total_tokens": 20}
    )
    assert not pool.drain_snapshot()["lengths"]
    lease = pool.acquire("t2", "r", "q", 6, {"question_id": "q"})
    pool.release(
        "t2", lease["lease_id"], observation={"finished": True, "total_tokens": 100}
    )
    assert pool.drain_snapshot()["completed"] == {"q": ["r"]}
    pool.finish_group("q")
    assert not pool.drain_snapshot()["completed"]


def test_long_tail_is_not_inferred_complete_from_short_completed_answers():
    assert residual_tokens(1000, 2048, [100] * 20) == (1048, False)
    assert residual_tokens(80, 2048, [100] * 20) == (20, True)


def fixture(rate0=1, rate1=100, missing=False):
    def engine(rate, rid):
        row = dict(
            request_id=rid,
            tokens=11,
            first=1.0,
            last=1.0 + 10 / rate,
            queued=0.0,
            prompt_tokens=100,
        )
        return dict(active=[row], history=[dict(row) for _ in range(8)])

    def lease(rid, rank):
        return dict(
            request_id=rid,
            rank=rank,
            group_size=2,
            metrics=dict(
                question_id="q",
                generated_tokens=0,
                prompt_tokens=100,
                max_new_tokens=2048,
            ),
        )

    route = dict(
        active=[lease("a", 0), lease("b", 1)],
        waiting=[],
        completed={},
        lengths=[100] * 20,
    )
    if missing:
        route["active"].pop()
    return route, [engine(rate0, "a"), engine(rate1, "b")]


def test_cost_requires_complete_question_horizon():
    route, engines = fixture(missing=True)
    assert (
        estimate_drain(route, engines, 1, 1)["reason"] == "incomplete_question_horizon"
    )


def test_migration_accounts_for_destination_speed_and_prefill():
    route, engines = fixture()
    result = estimate_drain(route, engines, 1, 1)
    assert result["action"] == "stop"
    assert result["move_eta"] < result["stay_eta"]
    route, engines = fixture(rate0=100, rate1=1)
    assert estimate_drain(route, engines, 1, 1)["action"] == "wait"


def test_sparse_and_censored_estimates_never_force_migration():
    route, engines = fixture()
    route["lengths"] = []
    assert estimate_drain(route, engines, 1, 1)["reason"] == "insufficient_history"
    route["lengths"] = [5] * 20
    result = estimate_drain(route, engines, 1, 1)
    assert result["action"] == "wait" and result["low_confidence_requests"] == 2


def test_event_stop_does_not_require_telemetry():
    from areal.infra.controller.rollout_controller import RolloutController

    controller = object.__new__(RolloutController)
    controller._request_pool = RequestLeasePool(["a", "b"], 8)
    controller._request_pool.acquire("t", "r")
    controller.config = SimpleNamespace(request_timeout=60, max_head_offpolicyness=2)
    controller.training_prefetch_ready = Mock(return_value=True)
    assert controller._adaptive_drain_reason(16) == "next_batch_ready"


def test_progress_hook_disabled_without_opt_in(monkeypatch):
    from areal.utils import generation_progress

    monkeypatch.delenv("AREAL_DRAIN_METRICS", raising=False)
    # No vLLM import on the disabled path.
    generation_progress.install()


@pytest.mark.parametrize(
    "invalid,expected", [("budget", "budget_invalid"), ("version", "version_limit")]
)
def test_adaptive_safety_event_overrides_estimate(monkeypatch, invalid, expected):
    import requests

    from areal.infra.controller.rollout_controller import RolloutController

    controller = object.__new__(RolloutController)
    controller._request_pool = RequestLeasePool(["a", "b"], 8)
    controller._request_pool.acquire("t", "r")
    controller.config = SimpleNamespace(request_timeout=60, max_head_offpolicyness=2)
    controller.get_version = Mock(return_value=3 if invalid == "version" else 1)
    controller.training_prefetch_ready = Mock(return_value=False)
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    session.post.return_value.json.return_value = dict(
        time=time.monotonic() - (10 if invalid == "stale" else 0),
        budget_valid=invalid != "budget",
        generation_running=True,
    )
    monkeypatch.setattr(requests, "Session", lambda: session)
    assert controller._adaptive_drain_reason(16) == expected


def test_adaptive_cost_requires_two_consecutive_votes(monkeypatch):
    import requests

    from areal.infra import adaptive_drain
    from areal.infra.controller.rollout_controller import RolloutController

    controller = object.__new__(RolloutController)
    controller._request_pool = RequestLeasePool(["a", "b"], 8)
    controller._request_pool.acquire("t", "r")
    controller.config = SimpleNamespace(request_timeout=60, max_head_offpolicyness=2)
    controller.get_version = Mock(return_value=1)
    controller.training_prefetch_ready = Mock(return_value=False)
    controller._staleness_manager = Mock()
    controller._staleness_manager.get_stats.return_value.accepted = 0
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    session.post.return_value.json.return_value = dict(
        time=time.monotonic(), budget_valid=True, generation_running=True
    )
    monkeypatch.setattr(requests, "Session", lambda: session)
    estimate = Mock(
        side_effect=[
            dict(action=a, reason="test") for a in ["stop", "wait", "stop", "stop"]
        ]
    )
    monkeypatch.setattr(adaptive_drain, "estimate_drain", estimate)
    monkeypatch.setattr(
        "areal.infra.controller.rollout_controller.time.sleep", lambda _: None
    )
    assert controller._adaptive_drain_reason(16) == "migration_faster"
    assert estimate.call_count == 4
