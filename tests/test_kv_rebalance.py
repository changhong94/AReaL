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
        _kv_peers={(1, 0): {"tensors": 2, "block_bytes": 128, "block_tokens": 16}},
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


def test_tp_pair_checks_geometry_before_creating_communicators(monkeypatch):
    import pytest

    calls = []

    def rpc(address, action, data):
        calls.append(action)
        return dict(tp_size=2, model_layout=address)

    monkeypatch.setattr(kv_rebalance, "rpc", rpc)
    with pytest.raises(RuntimeError, match="geometries differ"):
        kv_rebalance.initialize_pair(["source", "target"], 0, 1)
    assert calls == ["info", "info"]


def test_tp_pair_uses_distinct_matching_rendezvous_for_shards(monkeypatch):
    from areal.utils import network

    geometry = dict(tp_size=2, model_layout="same")
    inits = {}

    def rpc(address, action, data):
        if action == "init":
            inits[address] = data
        return dict(geometry)

    monkeypatch.setattr(kv_rebalance, "rpc", rpc)
    monkeypatch.setattr(
        network, "find_free_ports", lambda count: [12001, 12002][:count]
    )
    monkeypatch.setattr(network, "gethostip", lambda: "127.0.0.1")
    assert kv_rebalance.initialize_pair(["source", "target"], 0, 1) == geometry
    assert inits["source"]["init_methods"] == inits["target"]["init_methods"]
    assert len(set(inits["source"]["init_methods"])) == 2
    assert inits["source"]["rank"] == 0
    assert inits["target"]["rank"] == 1


def test_all_directed_replica_pairs_keep_unique_channels(monkeypatch):
    """Every fixed/borrowed edge gets independent IDs and local endpoint ranks."""
    from areal.utils import network

    calls = []
    geometry = dict(tp_size=2, block_tokens=128)

    def rpc(address, action, data):
        if action == "init":
            calls.append((address, data))
        return dict(geometry)

    monkeypatch.setattr(kv_rebalance, "rpc", rpc)
    monkeypatch.setattr(
        network, "find_free_ports", lambda count: list(range(12000, 12000 + count))
    )
    monkeypatch.setattr(network, "gethostip", lambda: "127.0.0.1")
    peers = kv_rebalance.initialize_pairs(["a", "b", "c", "d"], [2, 3], [0, 1])
    assert set(peers) == {(2, 0), (2, 1), (3, 0), (3, 1)}
    assert len({data["group_name"] for _, data in calls}) == 4
    assert len({data["pair_id"] for _, data in calls}) == 4
    for source, target in peers:
        edge = [
            (address, data)
            for address, data in calls
            if data["pair_id"] == kv_rebalance.pair_key(source, target)
        ]
        assert len(edge) == 2
        assert {data["rank"] for _, data in edge} == {0, 1}
        assert {address for address, _ in edge} == {"abcd"[source], "abcd"[target]}


def test_invalid_peer_sets_reject_before_rpc(monkeypatch):
    """Invalid topology must not partially create communicators."""
    import pytest

    def forbidden(*args):
        raise AssertionError("Invalid topology reached RPC")

    monkeypatch.setattr(kv_rebalance, "rpc", forbidden)
    for sources, targets in [
        ([], [0]),
        ([1], []),
        ([1, 1], [0]),
        ([1], [1]),
        ([2], [0]),
    ]:
        with pytest.raises(ValueError):
            kv_rebalance.initialize_pairs(["a", "b"], sources, targets)


def test_retained_evacuation_mixes_kv_and_version_mismatch_replay(monkeypatch):
    from areal.infra.request_router import RequestLeasePool

    pool = RequestLeasePool(["source", "target"], 4, task_affinity=False)
    pool.close(1)
    leases = [pool.acquire(str(i), str(i)) for i in range(2)]
    pool.close(0)
    pool.open(1, 0)
    pool.active["1"]["version"] = -1
    calls = []
    tickets = {}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, timeout):
            calls.append((url, json))
            if url.endswith("areal_generation_progress"):
                body = {"internal_ids": {"0": "native-0", "1": "native-1"}}
            else:
                selected = json.get("internal_ids", ["native-1"])
                for native in selected:
                    rid = native.split("-")[-1]
                    pool.release(rid, leases[int(rid)]["lease_id"])
                    continuation = pool.acquire(rid + "-continued", rid)
                    assert continuation["rank"] == 1
                    assert continuation["running_handoff"] == (rid == "0")
                body = {"aborted": selected}
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    def rpc(address, action, data):
        calls.append((action, address))
        if action == "export":
            assert data["retained_source"] and data["request_id"] == "native-0"
            return dict(bytes=16)
        if action == "register":
            tickets[data["ticket"]] = dict(outcome="import_applied")
        if action == "import_status":
            return dict(results=tickets)
        return {}

    monkeypatch.setattr(kv_rebalance.requests, "Session", Session)
    monkeypatch.setattr(kv_rebalance, "rpc", rpc)
    controller = SimpleNamespace(
        _request_pool=pool,
        borrowed_ranks=[0],
        fixed_ranks=[1],
        _kv_peers={(0, 1): {}},
        config=SimpleNamespace(request_timeout=1),
    )
    kv_rebalance.evacuate_retained(controller)
    assert not pool.active_requests(0)
    assert sum(action == "export" for action, _ in calls) == 1
    assert calls.index(("register", "target")) < next(
        i
        for i, (url, body) in enumerate(calls)
        if url.endswith("areal_evacuate_retained") and body.get("internal_ids")
    )
    assert not pool.retained_kv_pins
