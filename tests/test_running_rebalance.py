# SPDX-License-Identifier: Apache-2.0
"""R001 causal forecasts and version-pinned ownership transitions."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.test_queue_rebalance import inputs, source_pool

from areal.infra.running_rebalance import plan_running_shadow


def fixture():
    c, r, e = inputs()
    r.update(versions=[0, 0], ready=[True, True], task_affinity=False)
    for lease in r["active"]:
        lease["version"] = 0
    e[1]["queue"].update(queued=[], running=["running", "queued"], width=2)
    e[0]["queue"].update(free_kv_tokens=1000)
    e[0].update(budget_valid=True, generation_running=True)
    c.service = lambda rank, n, ctx: dict(seconds=0.1 * n)
    return c, r, e


def test_shadow_predicts_single_move_without_mutation():
    c, r, e = fixture()
    result = plan_running_shadow(c, r, e, 0.01, 1, 0)
    assert len(result["selected"]) == 1
    assert result["execution_path"] == "abort_then_prefix_replay"
    assert result["predictions"][0]["replay_tokens"] == 10
    assert len(r["active"]) == 2


@pytest.mark.parametrize(
    "case", ["budget", "transition", "version", "lease", "queue", "memory", "nan"]
)
def test_shadow_rejects_unsafe_or_uncalibrated_states(case):
    c, r, e = fixture()
    overhead = 0.01
    if case == "budget":
        e[0]["budget_valid"] = False
    if case == "transition":
        e[0]["generation_running"] = False
    if case == "version":
        r["versions"][0] = 1
    if case == "lease":
        r["active"][0]["version"] = 2
    if case == "queue":
        e[1]["queue"]["queued"] = ["queued"]
    if case == "memory":
        e[0]["queue"]["free_kv_tokens"] = 1
    if case == "nan":
        overhead = float("nan")
    assert not plan_running_shadow(c, r, e, overhead, 1, 0)["selected"]


def test_replay_cost_and_cooldown_can_eliminate_gain():
    c, r, e = fixture()
    r["running_moved"] = ["running", "queued"]
    assert not plan_running_shadow(c, r, e, 0.01, 1, 0)["selected"]
    r["running_moved"] = []
    c.first[0] = dict(unit=100)
    assert not plan_running_shadow(c, r, e, 0.01, 1, 0)["selected"]


def test_running_reservation_pins_epoch_until_destination_lease():
    p, old = source_pool()
    assert p.reserve_rebalance({"a": old["lease_id"]}, 0, running=True)
    p.release(old["ticket"], old["lease_id"], observation={"finished": False})
    with pytest.raises(RuntimeError, match="pins replica epoch"):
        p.set_version(0, 1)
    with pytest.raises(RuntimeError, match="pins replica epoch"):
        p.open(0, 1)
    new = p.acquire("a:2", "a")
    assert new["rank"] == 0 and new["version"] == old["version"]
    assert not p.reserve_rebalance({"a": new["lease_id"]}, 0, 0, 1, running=True)


def test_publication_wait_does_not_block_ack_or_admission():
    p, old = source_pool()
    p.reserve_rebalance({"a": old["lease_id"]}, 0, running=True)
    with ThreadPoolExecutor(1) as executor:
        closing = executor.submit(p.close, 0)
        deadline = time.monotonic() + 1
        while 0 not in p.closing and time.monotonic() < deadline:
            time.sleep(0.001)
        assert not closing.done()
        assert p.acquire("a:2", "a") is None
        p.release(old["ticket"], old["lease_id"], observation={"finished": False})
        new = p.acquire("a:2", "a")
        assert new["rank"] == 0 and new["version"] == 0
        closing.result(timeout=1)
    assert not p.ready[0]
    assert p.active_requests(0) == ["a"]


def test_finished_source_clears_handoff_without_replay():
    p, old = source_pool()
    p.reserve_rebalance({"a": old["lease_id"]}, 0, running=True)
    p.release(
        old["ticket"],
        old["lease_id"],
        observation={"finished": True, "total_tokens": 9},
    )
    p.close(0)
    assert not p.rebalance_reservations
    assert not p.running_moved


def test_rejected_destination_does_not_lose_version_constraint():
    p, old = source_pool()
    p.reserve_rebalance({"a": old["lease_id"]}, 0, running=True)
    p.release(
        old["ticket"],
        old["lease_id"],
        observation={"finished": False, "segment_tokens": 17},
    )
    new = p.acquire("a:2", "a")
    assert new["running_handoff"]
    p.release(new["ticket"], new["lease_id"])
    p.set_version(0, 1)
    p.set_version(1, 1)
    with pytest.raises(RuntimeError, match="compatible epoch"):
        p.acquire("a:3", "a")


def test_rejected_handoff_has_bounded_admission_retries():
    p, old = source_pool()
    p.reserve_rebalance({"a": old["lease_id"]}, 0, running=True)
    p.release(old["ticket"], old["lease_id"], observation={"finished": False})
    for segment in range(2, 5):
        new = p.acquire(f"a:{segment}", "a")
        p.release(new["ticket"], new["lease_id"])
    with pytest.raises(RuntimeError, match="retry limit"):
        p.acquire("a:5", "a")


@pytest.mark.parametrize("imminent", [False, True])
def test_active_controller_reserves_before_running_abort(monkeypatch, imminent):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import requests

    import areal.infra.queue_rebalance as queued
    import areal.infra.running_rebalance as running
    from areal.infra.controller.rollout_controller import RolloutController

    monkeypatch.setenv("AREAL_QUEUE_REBALANCE", "1")
    monkeypatch.setenv("AREAL_BATCH_DRAIN_METRICS", "1")
    monkeypatch.setenv("AREAL_RUNNING_REBALANCE", "active")
    p, old = source_pool()
    owner = SimpleNamespace(
        _request_pool=p,
        training_prefetch_target=lambda: 1,
        training_prefetch_ready=lambda _: imminent,
    )
    monkeypatch.setattr(
        queued,
        "plan_rebalance",
        lambda *a: dict(selected=[], predictions=[], reason="no_queued_work"),
    )
    monkeypatch.setattr(
        running,
        "plan_running_shadow",
        lambda *a: dict(
            selected=["a"], predictions=[], reason="predicted_gain", version=0
        ),
    )
    snap = Mock()
    snap.json.return_value = dict(internal_ids={"a": "native-a"})
    ack = Mock()
    ack.json.return_value = dict(aborted=["native-a"])
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)

    def post(*args, **kwargs):
        if kwargs["json"]["action"] == "snapshot":
            return snap
        assert kwargs["json"]["action"] == "abort_running"
        assert p.rebalance_reservations["a"]["running"]
        return ack

    session.post.side_effect = post
    monkeypatch.setattr(requests, "Session", lambda: session)
    RolloutController._rebalance_pair(owner, 0, 1, 0)
    assert old["ticket"] in p.active
    assert ("a" in p.rebalance_reservations) != imminent


@pytest.mark.parametrize("mode", ["off", "shadow", "budget", "active"])
def test_active_budget_is_established_before_shared_admission(monkeypatch, mode):
    from contextlib import nullcontext
    from unittest.mock import Mock, call

    import areal.trainer.rl_trainer as module

    monkeypatch.setenv("AREAL_RUNNING_REBALANCE", mode)
    monkeypatch.setattr(
        module.perf_tracer, "trace_scope", lambda *a, **kw: nullcontext()
    )
    trainer = object.__new__(module.PPOTrainer)
    trainer._started_snapshot_version = 1
    trainer._adaptive_drain = False
    trainer.actor = Mock()
    trainer._finish_training_drain = Mock()
    trainer.weight_update_meta = Mock()
    trainer._memory_policy = Mock()
    trainer.rollout = Mock()
    trainer.eval_rollout = None
    order = Mock()
    order.attach_mock(trainer._memory_policy, "memory")
    order.attach_mock(trainer.rollout, "rollout")
    trainer._finish_shared_weights(1)
    if mode in {"active", "budget"}:
        trainer._memory_policy.inference_budget.assert_called_once_with()
        trainer._memory_policy.concurrent_train.assert_not_called()
        assert order.mock_calls.index(call.memory.inference_budget()) < next(
            i
            for i, c in enumerate(order.mock_calls)
            if c[0] == "rollout.load_shared_snapshot_async"
        )
    else:
        trainer._memory_policy.inference_budget.assert_not_called()
