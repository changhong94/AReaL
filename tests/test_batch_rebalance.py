# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from types import SimpleNamespace

import pytest

from areal.infra.batch_rebalance import KVTransferCost, plan_batch_rebalance


def fixture(context=10):
    c = SimpleNamespace(
        lengths=[(100, True)],
        tail_rate=None,
        first=[dict(unit=0.001), dict(unit=0.001)],
        decode=[{1: 1}, {1: 1}],
        service=lambda rank, n, ctx: dict(seconds=0.1 if n >= 3 else 0.01),
    )
    active = [
        dict(
            request_id=str(i),
            rank=1,
            version=0,
            metrics=dict(generated_tokens=0, max_new_tokens=100, prompt_tokens=context),
        )
        for i in range(4)
    ]
    route = dict(
        active=active, waiting=[], ready=[True, True], versions=[0, 0], capacity=4
    )

    def engine(ids):
        return dict(
            active=[dict(request_id=i, tokens=0) for i in ids],
            generation_running=True,
            budget_valid=True,
            queue=dict(
                running=ids, queued=[], waiting=[], width=4, free_kv_tokens=10000
            ),
        )

    return c, route, [engine([]), engine([str(i) for i in range(4)])]


def test_batch_crosses_service_bucket_when_single_move_cannot():
    c, r, e = fixture()
    old = deepcopy((r, e))
    assert not plan_batch_rebalance(c, r, e, 0.01, max_moves=1)["selected"]
    result = plan_batch_rebalance(c, r, e, 0.01)
    assert len(result["selected"]) == 2
    assert result["predicted_gain_s"] > 5
    assert (r, e) == old
    assert not result["execution_ready"]


def test_kv_option_avoids_expensive_replay_but_is_not_executable():
    c, r, e = fixture(1000)
    c.first[0]["unit"] = 0.1
    assert not plan_batch_rebalance(c, r, e, 0.01)["selected"]
    cost = KVTransferCost(1024, 16, 0.001, 1e-10, 2**24, 0.01)
    result = plan_batch_rebalance(c, r, e, 0.01, transport=cost)
    assert len(result["selected"]) == 2
    assert all(x["method"] == "kv" for x in result["selected"])
    assert not result["execution_ready"]
    assert not any(x["computed_tokens_exact"] for x in result["candidates"])


def test_tiny_headroom_or_long_handoff_rejects_migration():
    c, r, e = fixture(1000)
    c.first[0]["unit"] = 0.1
    cost = KVTransferCost(1024, 16, 0.001, 1e-10, 2**24, 100)
    assert not plan_batch_rebalance(c, r, e, 0.01, transport=cost)["selected"]
    e[0]["queue"]["free_kv_tokens"] = 1
    assert not plan_batch_rebalance(c, r, e, 0.01)["selected"]


def test_rounding_and_no_unmeasured_size_extrapolation():
    cost = KVTransferCost(1024, 16, 0.001, 1e-9, 32 * 1024, 0)
    assert cost.estimate(17)[0] == 32 * 1024
    assert cost.estimate(33) is None
    with pytest.raises(ValueError):
        KVTransferCost(1024, 16, float("nan"), 1e-9, 32 * 1024, 0)


def test_version_and_budget_are_required_even_in_offline_plan():
    c, r, e = fixture()
    r["versions"][0] = 1
    assert plan_batch_rebalance(c, r, e, 0.01)["reason"] == "incompatible_epoch"
    r["versions"][0] = 0
    e[0]["budget_valid"] = False
    assert not plan_batch_rebalance(c, r, e, 0.01)["selected"]


def test_partial_prefill_charges_all_uncomputed_tokens():
    c, r, e = fixture(1000)
    c.first[0]["unit"] = 0.1
    for live in e[1]["active"]:
        live["computed_tokens"] = 100
    cost = KVTransferCost(1024, 16, 0.001, 1e-10, 2**24, 0)
    result = plan_batch_rebalance(c, r, e, 0.01, transport=cost)
    assert not result["selected"]
    assert result["candidates"][0]["alternatives"][1]["prefill_s"] == 90


def test_native_membership_required():
    c, r, e = fixture()
    e[1]["queue"]["running"].pop()
    assert plan_batch_rebalance(c, r, e, 0.01)["reason"] == "native_snapshot_race"


def test_live_kv_only_requires_exact_computed_tokens():
    c, r, e = fixture(1000)
    c.first[0]["unit"] = 0.1
    cost = KVTransferCost(1024, 16, 0.01, 1e-10, 2**24, 0)
    assert not plan_batch_rebalance(c, r, e, 0.01, transport=cost, kv_only=True)[
        "selected"
    ]
    for live in e[1]["active"]:
        live["computed_tokens"] = 999
    plan = plan_batch_rebalance(c, r, e, 0.01, transport=cost, kv_only=True)
    assert plan["selected"]
    assert all(x["method"] == "kv" for x in plan["selected"])
    assert all(x["variant"] == "mixed" for x in plan["scenarios"])


def test_flat_decode_service_does_not_move_requests_for_balance_alone():
    c, r, e = fixture(1000)
    c.service = lambda rank, n, context: dict(seconds=0.05)
    for live in e[1]["active"]:
        live["computed_tokens"] = 999
    cost = KVTransferCost(1024, 16, 0.1, 1e-10, 2**24, 0)
    plan = plan_batch_rebalance(c, r, e, 0.01, transport=cost, kv_only=True)
    assert not plan["selected"]


def test_common_barrier_charges_handoff_to_unmoved_work():
    c, r, e = fixture(1000)
    c.first[0]["unit"] = 0.1
    for live in e[1]["active"]:
        live["computed_tokens"] = 999
    cost = KVTransferCost(1024, 16, 0.5, 0, 2**24, 0)
    plan = plan_batch_rebalance(c, r, e, 0.01, transport=cost, kv_only=True)
    two = next(x for x in plan["scenarios"] if x["count"] == 2)
    assert two["tail_s"] >= 2 * 0.51


def test_pending_admission_is_accounted_for_but_never_moved():
    c, r, e = fixture()
    e[1]["active"].pop()
    e[1]["queue"]["running"].pop()
    plan = plan_batch_rebalance(c, r, e, 0.01)
    assert plan["pending_requests"] == ["3"]
    assert "3" not in [x["request_id"] for x in plan["candidates"]]
    assert plan["reason"] not in ("snapshot_race", "native_snapshot_race")


def test_completed_attempt_is_not_confused_with_old_prefix_history():
    c, r, e = fixture()
    e[1]["active"].pop()
    e[1]["queue"]["running"].pop()
    e[1]["history"] = [dict(request_id="3", prompt_tokens=10, tokens=5)]
    plan = plan_batch_rebalance(c, r, e, 0.01)
    assert plan["completed_requests"] == ["3"]
    assert plan["pending_requests"] == []
    r["active"][-1]["metrics"]["generated_tokens"] = 5
    plan = plan_batch_rebalance(c, r, e, 0.01)
    assert plan["completed_requests"] == []
    assert plan["pending_requests"] == ["3"]


def test_destination_capacity_rounds_each_continuation_to_blocks():
    c, r, e = fixture(1000)
    c.first[0]["unit"] = 0.1
    for live in e[1]["active"]:
        live["computed_tokens"] = 999
    cost = KVTransferCost(1024, 16, 0.01, 0, 2**24, 0)
    e[0]["queue"]["free_kv_tokens"] = 2200
    assert not plan_batch_rebalance(c, r, e, 0.01, transport=cost, kv_only=True)[
        "selected"
    ]
    e[0]["queue"]["free_kv_tokens"] = 2208
    assert (
        len(
            plan_batch_rebalance(c, r, e, 0.01, transport=cost, kv_only=True)[
                "selected"
            ]
        )
        == 2
    )


def aggressive_options(needed=1):
    return dict(
        coordination_s=0.08,
        source_stall_s=0.02,
        target_stall_s=0.05,
        needed_questions=needed,
    )


def test_aggressive_keeps_common_staging_barrier_and_bounded_tail():
    c, r, e = fixture(1000)
    for live in e[1]["active"]:
        live["computed_tokens"] = 999
    cost = KVTransferCost(1024, 16, 0.18, 0, 2**24, 0)
    plan = plan_batch_rebalance(
        c, r, e, 0.01, transport=cost, kv_only=True, aggressive=aggressive_options()
    )
    assert plan["selected"]
    assert plan["objective"] == "mean_completion"
    assert plan["predicted_tail_s"] <= plan["baseline_tail_s"] + 0.2
    assert all(s["tail_s"] >= 0.08 + s["count"] * 0.18 for s in plan["scenarios"])


def test_aggressive_question_objective_requires_complete_answer_horizon():
    c, r, e = fixture(1000)
    for i, row in enumerate(r["active"]):
        row["metrics"]["question_id"] = str(i // 2)
        row["group_size"] = 2
    for live in e[1]["active"]:
        live["computed_tokens"] = 999
    cost = KVTransferCost(1024, 16, 0.01, 0, 2**24, 0)
    plan = plan_batch_rebalance(
        c, r, e, 0, transport=cost, kv_only=True, aggressive=aggressive_options(2)
    )
    assert plan["known_questions"] == 2
    assert plan["objective"] == "batch_ready"
    r["active"][0]["group_size"] = 3
    plan = plan_batch_rebalance(
        c, r, e, 0, transport=cost, kv_only=True, aggressive=aggressive_options(2)
    )
    assert plan["known_questions"] == 1
    assert plan["objective"] == "mean_completion"


def test_aggressive_does_not_bypass_budget_version_or_flat_service():
    c, r, e = fixture(1000)
    for live in e[1]["active"]:
        live["computed_tokens"] = 999
    c.service = lambda rank, n, context: dict(seconds=0.05)
    cost = KVTransferCost(1024, 16, 0.18, 0, 2**24, 0)
    args = dict(transport=cost, kv_only=True, aggressive=aggressive_options())
    assert not plan_batch_rebalance(c, r, e, 0.01, **args)["selected"]
    e[0]["budget_valid"] = False
    assert plan_batch_rebalance(c, r, e, 0.01, **args)["reason"] == "target_unavailable"
    e[0]["budget_valid"] = True
    r["versions"][0] = 1
    assert plan_batch_rebalance(c, r, e, 0.01, **args)["reason"] == "incompatible_epoch"


def test_bounded_exploration_can_sample_flat_service_without_claiming_gain():
    c, r, e = fixture(1000)
    # Replicate to a busy source while keeping an idle, sufficiently large target.
    r["active"] = [
        dict(
            request_id=str(i),
            rank=1,
            version=0,
            metrics=dict(generated_tokens=0, max_new_tokens=100, prompt_tokens=1000),
        )
        for i in range(20)
    ]
    e[1]["active"] = [
        dict(request_id=str(i), tokens=0, computed_tokens=999) for i in range(20)
    ]
    e[1]["queue"]["running"] = [str(i) for i in range(20)]
    r["capacity"] = 32
    e[0]["queue"].update(width=32, free_kv_tokens=100000)
    e[1]["queue"]["width"] = 32
    c.service = lambda rank, n, context: dict(seconds=0.05)
    cost = KVTransferCost(1024, 16, 0.18, 0, 2**24, 0)
    policy = dict(aggressive_options(), explore=True)
    plan = plan_batch_rebalance(
        c, r, e, 0, transport=cost, kv_only=True, aggressive=policy
    )
    assert plan["reason"] == "bounded_exploration"
    assert 1 <= len(plan["selected"]) <= 8
    assert plan["predicted_gain_s"] <= 0
    assert plan["exploration_stage_budget_s"] <= 1.6
    policy["explore"] = False
    assert not plan_batch_rebalance(
        c, r, e, 0, transport=cost, kv_only=True, aggressive=policy
    )["selected"]


def test_exploration_still_rejects_unavailable_memory_and_wrong_version():
    c, r, e = fixture(1000)
    cost = KVTransferCost(1024, 16, 0.18, 0, 2**24, 0)
    args = dict(
        transport=cost,
        kv_only=True,
        aggressive=dict(aggressive_options(), explore=True),
    )
    e[0]["budget_valid"] = False
    assert plan_batch_rebalance(c, r, e, 0, **args)["reason"] == "target_unavailable"
    e[0]["budget_valid"] = True
    r["versions"][0] = 1
    assert plan_batch_rebalance(c, r, e, 0, **args)["reason"] == "incompatible_epoch"
