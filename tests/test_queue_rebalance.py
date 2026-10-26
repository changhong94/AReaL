# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.engine.vllm_ext.queue_rebalance import eligible, queue_operation
from areal.infra.queue_rebalance import plan_rebalance
from areal.infra.request_router import RequestLeasePool


def request(rid="q", **kwargs):
    return SimpleNamespace(
        **dict(
            dict(
                request_id=rid,
                status="waiting",
                num_computed_tokens=0,
                output_token_ids=[],
            ),
            **kwargs,
        )
    )


def scheduler(req):
    s = SimpleNamespace(
        waiting=[req],
        running=[],
        requests={req.request_id: req},
        _areal_ever_scheduled=set(),
        max_num_running_reqs=64,
    )

    def finish(ids, status):
        removed = [r for r in s.waiting if r.request_id in ids]
        s.waiting = [r for r in s.waiting if r.request_id not in ids]
        for r in removed:
            s.requests.pop(r.request_id)
            r.status = status
        return removed

    s.finish_requests = finish
    return s


@pytest.mark.parametrize("change", ["prefill", "seen", "output", "running", "status"])
def test_never_steal_started_or_preempted_requests(change):
    r = request()
    s = scheduler(r)
    if change == "prefill":
        r.num_computed_tokens = 1
    if change == "seen":
        s._areal_ever_scheduled.add("q")
    if change == "output":
        r.output_token_ids = [1]
    if change == "running":
        s.running.append(r)
    if change == "status":
        r.status = "preempted"
    assert not eligible(s, r, "waiting")


def test_engine_rechecks_snapshot_and_notifies_exact_abort_once():
    r = request()
    s = scheduler(r)
    core = SimpleNamespace(scheduler=s, _send_abort_outputs=Mock())
    assert queue_operation(core, "snapshot", [], "waiting", "aborted")["queued"] == [
        "q"
    ]
    s._areal_ever_scheduled.add("q")
    assert queue_operation(core, "abort", ["q"], "waiting", "aborted")["aborted"] == []
    s._areal_ever_scheduled.clear()
    assert queue_operation(core, "abort", ["q"], "waiting", "aborted")["aborted"] == [
        "q"
    ]
    assert queue_operation(core, "abort", ["q"], "waiting", "aborted")["aborted"] == []
    assert [
        c.args[0] for c in core._send_abort_outputs.call_args_list if c.args[0]
    ] == [[r]]


def source_pool():
    p = RequestLeasePool(["shared", "fixed"], 2, task_affinity=False)
    p.close(0)
    lease = p.acquire("a:1", "a")
    p.open(0, 0)
    assert lease["rank"] == 1
    return p, lease


def test_reservation_keeps_source_lease_until_ack_then_pins_target():
    p, lease = source_pool()
    assert p.reserve_rebalance({"a": lease["lease_id"]}, 0) == {"a": lease["lease_id"]}
    assert p.acquire("a:2", "a") is None
    p.release(lease["ticket"], lease["lease_id"])
    assert p.acquire("a:2", "a")["rank"] == 0
    assert not p.rebalance_reservations


def test_reserved_slot_cannot_be_stolen_by_other_request():
    p, lease = source_pool()
    p.capacity = 1
    p.reserve_rebalance({"a": lease["lease_id"]}, 0)
    assert p.acquire("b:1", "b") is None


def test_close_clears_hint_and_aborted_request_falls_back_safely():
    p, lease = source_pool()
    p.reserve_rebalance({"a": lease["lease_id"]}, 0)
    p.close(0)
    p.release(lease["ticket"], lease["lease_id"])
    assert p.acquire("a:2", "a")["rank"] == 1


def test_stale_version_and_stale_lease_cannot_reserve():
    p, lease = source_pool()
    assert not p.reserve_rebalance({"a": lease["lease_id"]}, 1)
    assert not p.reserve_rebalance({"a": lease["lease_id"] + 1}, 0)


def test_timeout_hint_does_not_duplicate_and_finishing_source_cleans_it():
    p, lease = source_pool()
    p.reserve_rebalance({"a": lease["lease_id"]}, 0)
    assert p.acquire("a:2", "a") is None
    p.release(
        lease["ticket"],
        lease["lease_id"],
        observation={"finished": True, "total_tokens": 3},
    )
    assert not p.rebalance_reservations


def test_ambiguous_source_execution_remains_fail_closed():
    p, lease = source_pool()
    p.reserve_rebalance({"a": lease["lease_id"]}, 0)
    p.release(lease["ticket"], lease["lease_id"], uncertain=True)
    with pytest.raises(RuntimeError):
        p.acquire("a:2", "a")


def inputs():
    leases = [
        dict(
            request_id=r,
            rank=1,
            lease_id=i,
            metrics=dict(generated_tokens=0, max_new_tokens=100, prompt_tokens=10),
        )
        for i, r in enumerate(["running", "queued"])
    ]
    route = dict(active=leases, waiting=[], capacity=64, reserved=0)
    engines = [
        dict(active=[], queue=dict(queued=[], width=1)),
        dict(
            active=[dict(request_id=r, tokens=0) for r in ["running", "queued"]],
            queue=dict(queued=["queued"], width=1),
        ),
    ]
    c = SimpleNamespace(
        lengths=[(100, True)],
        tail_rate=None,
        first=[dict(unit=0.001)] * 2,
        decode=[{1: 1}] * 2,
        migration=dict(seconds=0.01),
        service=lambda rank, n, ctx: dict(seconds=0.1),
    )
    return c, route, engines


def test_cost_model_moves_queued_work_only_when_completion_improves():
    c, r, e = inputs()
    p = plan_rebalance(c, r, e, 0.01)
    assert p["selected"] == ["queued"]
    assert p["predictions"][0]["tail_gain_s"] > 9


def test_transfer_overhead_can_eliminate_benefit():
    c, r, e = inputs()
    c.migration = dict(seconds=20)
    assert plan_rebalance(c, r, e, 0.01)["selected"] == []


def test_public_queue_has_priority_and_uncalibrated_model_skips():
    c, r, e = inputs()
    r["waiting"] = [dict(request_id="new")]
    assert plan_rebalance(c, r, e, 0.01)["reason"] == "public_queue_first"
    r["waiting"] = []
    assert plan_rebalance(None, r, e, 0.01)["reason"] == "uncalibrated"


def test_running_tail_is_never_eligible_even_if_target_idle():
    c, r, e = inputs()
    e[1]["queue"]["queued"] = []
    assert plan_rebalance(c, r, e, 0.01)["selected"] == []


@pytest.mark.parametrize("outcome", ["ack", "started", "timeout"])
def test_controller_handoff_preserves_ack_fence(monkeypatch, outcome):
    import requests

    import areal.infra.queue_rebalance as planner
    from areal.infra.controller.rollout_controller import RolloutController

    monkeypatch.setenv("AREAL_QUEUE_REBALANCE", "1")
    monkeypatch.setenv("AREAL_BATCH_DRAIN_METRICS", "1")
    p, lease = source_pool()
    owner = SimpleNamespace(
        _request_pool=p, _batch_calibration=None, borrowed_ranks=[0], fixed_ranks=[1]
    )
    owner._rebalance_pair = lambda *args: RolloutController._rebalance_pair(
        owner, *args
    )
    monkeypatch.setattr(
        planner,
        "plan_rebalance",
        lambda *a: dict(selected=["a"], reason="test", predictions=[]),
    )
    response = Mock()
    response.json.return_value = dict(internal_ids={"a": "native-a"})
    abort = Mock()
    abort.json.return_value = dict(aborted=["native-a"] if outcome == "ack" else [])
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    session.post.side_effect = [
        response,
        response,
        requests.Timeout() if outcome == "timeout" else abort,
    ]
    monkeypatch.setattr(requests, "Session", lambda: session)
    RolloutController._rebalance_queued_once(owner, 0)
    assert lease["ticket"] in p.active  # Utility ACK never releases the HTTP lease.
    assert ("a" in p.rebalance_reservations) == (outcome != "started")
    assert p.acquire("a:2", "a") is None
    p.release(lease["ticket"], lease["lease_id"])
    assert p.acquire("a:2", "a")["rank"] == (1 if outcome == "started" else 0)


def test_target_reservation_respects_affinity_and_public_waiters():
    p, lease = source_pool()
    p.task_affinity = True
    assert not p.reserve_rebalance({"a": lease["lease_id"]}, 0)
    p.task_affinity = False
    p.waiting["b:1"] = dict(request_id="b")
    assert not p.reserve_rebalance({"a": lease["lease_id"]}, 0)


def test_rebalance_submission_does_not_wait_for_publication(monkeypatch):
    import areal.infra.controller.rollout_controller as module

    monkeypatch.setenv("AREAL_QUEUE_REBALANCE", "1")
    monkeypatch.setenv("AREAL_BATCH_DRAIN_METRICS", "1")
    executor = Mock()
    pending = Mock()
    pending.done.return_value = False
    executor.submit.return_value = pending
    monkeypatch.setattr(module, "get_executor", lambda *a, **k: executor)
    obj = SimpleNamespace(_rebalance_queued_once=Mock())
    module.RolloutController._schedule_queue_rebalance(obj, 3)
    executor.submit.assert_called_once_with(obj._rebalance_queued_once, 3)
    pending.result.assert_not_called()
    module.RolloutController._schedule_queue_rebalance(obj, 4)
    assert executor.submit.call_count == 1


def test_retained_selected_abort_keeps_other_paused_requests_until_finalize():
    running = request("running", status="running", num_computed_tokens=8)
    preempted = request("preempted", status="preempted", num_computed_tokens=0)
    s = scheduler(preempted)
    s.running = [running]
    s.requests["running"] = running

    def finish(ids, status):
        removed = [s.requests.pop(rid) for rid in ids if rid in s.requests]
        s.running = [r for r in s.running if r.request_id not in ids]
        s.waiting = [r for r in s.waiting if r.request_id not in ids]
        return removed

    s.finish_requests = finish
    core = SimpleNamespace(scheduler=s, _send_abort_outputs=Mock())
    assert queue_operation(
        core, "abort_retained_selected", ["running"], "waiting", "aborted"
    )["aborted"] == ["running"]
    assert list(s.requests) == ["preempted"]
    assert queue_operation(core, "abort_retained", [], "waiting", "aborted")[
        "aborted"
    ] == ["preempted"]
    assert not s.requests
