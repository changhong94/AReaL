# SPDX-License-Identifier: Apache-2.0
"""Coordinator rollback and cheap no-op gates without device operations."""

from types import SimpleNamespace

from areal.infra import kv_rebalance


def test_balanced_pair_never_invokes_planner_or_rpc(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Balanced workload must remain a cheap no-op")

    monkeypatch.setattr(kv_rebalance, "plan_batch_rebalance", forbidden)
    monkeypatch.setattr(kv_rebalance, "rpc", forbidden)
    controller = SimpleNamespace(
        _request_pool=SimpleNamespace(),
        training_prefetch_target=lambda: 1,
        training_prefetch_ready=lambda target: False,
    )
    kv_rebalance.rebalance(
        controller, 0, {"active": [{"rank": 0}, {"rank": 1}]}, [], 0.01, 1, 0
    )


def test_training_becoming_ready_rolls_back_before_abort(monkeypatch):
    calls, canceled = [], []
    checks = iter([False, False, True])
    pool = SimpleNamespace(
        addresses=["target", "source"],
        reserve_rebalance=lambda *args, **kwargs: {"r": 1},
        cancel_rebalance=lambda reservation: canceled.append(reservation),
    )
    controller = SimpleNamespace(
        _request_pool=pool,
        _borrowed_pool=SimpleNamespace(preserve_requests=False),
        staleness_manager=SimpleNamespace(
            get_stats=lambda: SimpleNamespace(accepted=0)
        ),
        training_prefetch_target=lambda: 1,
        training_prefetch_ready=lambda target: next(checks),
        _kv_peer_geometry={"tensors": 2, "block_bytes": 128},
        _kv_block_tokens=16,
    )
    monkeypatch.setattr(
        kv_rebalance,
        "plan_batch_rebalance",
        lambda *args, **kwargs: {
            "selected": [{"request_id": "r", "method": "kv"}],
            "net_predicted_gain_s": 100,
        },
    )

    def rpc(address, action, data):
        calls.append((address, action))
        return {"bytes": 256} if action == "export" else {}

    monkeypatch.setattr(kv_rebalance, "rpc", rpc)
    route = {
        "active": [
            {"request_id": "r", "lease_id": 1, "rank": 1},
            {"request_id": "s", "lease_id": 2, "rank": 1},
        ]
    }
    kv_rebalance.rebalance(
        controller, 0, route, [{}, {"internal_ids": {"r": "native-r"}}], 0.01, 1, 0
    )
    assert ("target", "register") in calls
    assert calls[-3:] == [
        ("target", "discard"),
        ("source", "discard"),
        ("source", "unfreeze"),
    ]
    assert canceled == [{"r": 1}]


def test_exploration_waits_when_one_question_short():
    plan = {"exploration": True}
    assert kv_rebalance.exploration_timing_guard(plan, 1) == "training_near_ready"
    assert kv_rebalance.exploration_timing_guard(plan, 2) is None


def test_positive_gain_is_not_suppressed_by_training_gate():
    assert kv_rebalance.exploration_timing_guard({"exploration": False}, 0) is None


def test_training_forecast_accounts_for_elapsed_planning():
    plan = dict(
        exploration=True,
        objective="batch_ready",
        training_ready_s=2.0,
        predicted_objective_s=2.5,
        predicted_gain_s=-0.5,
        exploration_stage_budget_s=1.5,
    )
    assert (
        kv_rebalance.exploration_timing_guard(plan, 2, 0.6)
        == "training_horizon_too_short"
    )
    assert kv_rebalance.exploration_timing_guard(plan, 2, 0.1) is None
    plan.pop("training_ready_s")
    plan["objective"] = "mean_completion"
    assert kv_rebalance.exploration_timing_guard(plan, 2, 0.6) is None
