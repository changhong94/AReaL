# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.engine.colocation_memory import ParkedWorkspace, TaggedMemory
from areal.trainer.colocation_memory import GIB, ColocationMemoryPolicy


def test_disposable_cache_covers_pressure_without_weight_transfer():
    """Do not copy live inference weights while disposable actor memory suffices."""
    cfg = SimpleNamespace(
        device_budget_gib=None, safety_margin_gib=2, train_workspace_gib=4
    )
    free = 3 * GIB
    calls = []

    def status():
        return dict(
            free_bytes=free,
            total_bytes=64 * GIB,
            restore_bytes=0,
            gradient_bytes=0,
            future_optimizer_bytes=0,
            supports_partial_reclaim=True,
        )

    def actor(action, **kwargs):
        nonlocal free
        calls.append(("actor", action))
        if action == "release_cache":
            free += 4 * GIB
        return status()

    def rollout(action, **kwargs):
        calls.append(("rollout", action))
        assert action != "reclaim", "No live weights should be transferred"
        return status()

    ColocationMemoryPolicy(cfg, actor, rollout, Mock()).train()
    assert calls[-1] == ("actor", "restore")
    assert ("actor", "release_cache") in calls


class Backend:
    def __init__(self):
        self.data = {
            i: SimpleNamespace(tag=tag, handle=(0, 4, i, 0), cpu_backup_tensor=None)
            for i, tag in [(1, "weights"), (2, "optimizer"), (3, "optimizer")]
        }
        self.contents = {i: bytearray([i] * 4) for i in self.data}
        self.free = 10
        self.synchronize = Mock()

    def inventory(self):
        return self.data

    def free_memory(self):
        return self.free, 100

    def backup(self, a):
        return self.contents[a.handle[2]][:]

    def unmap(self, a):
        del self.contents[a.handle[2]]
        self.free += 4

    def map(self, a):
        self.contents[a.handle[2]] = bytearray(4)
        self.free -= 4

    def restore(self, a):
        self.contents[a.handle[2]][:] = a.cpu_backup_tensor


def test_small_pressure_evicts_only_one_optimizer_block_and_preserves_weights():
    """A small deficit must not cause a whole-tag CPU migration."""
    b = Backend()
    m = TaggedMemory(b, ["weights", "optimizer"])
    before = {i: v[:] for i, v in b.contents.items()}
    s = m.reclaim(["optimizer", "weights"], 13)
    assert s["states"] == {"weights": "mapped", "optimizer": "partial"}
    assert s["restore_bytes"] == s["cpu_backup_bytes"] == 4
    assert b.contents[1] == before[1] and b.contents[3] == before[3]
    m.transition(["weights", "optimizer"], True)
    assert b.contents == before
    assert m.snapshot()["restore_bytes"] == 0


def test_zero_pressure_does_not_synchronize_or_copy():
    b = Backend()
    m = TaggedMemory(b, ["weights", "optimizer"])
    m.reclaim(["optimizer", "weights"], 10)
    b.synchronize.assert_not_called()
    assert all(a.cpu_backup_tensor is None for a in b.data.values())


def test_incremental_pressure_and_new_generation_backup_roundtrip():
    b = Backend()
    m = TaggedMemory(b, ["weights", "optimizer"])
    m.reclaim(["optimizer", "weights"], 13)
    m.reclaim(["optimizer", "weights"], 17)
    assert set(b.contents) == {1}
    m.transition(["optimizer"], True)
    b.contents[2][:] = b"new!"
    m.reclaim(["optimizer"], 13)
    m.transition(["optimizer"], True)
    assert b.contents[2] == b"new!"
    assert b.free == 10


@pytest.mark.parametrize("operation", ["backup", "unmap", "map", "restore"])
def test_partial_transfer_failure_stays_fail_closed(operation):
    b = Backend()
    m = TaggedMemory(b, ["weights", "optimizer"])
    if operation in ("map", "restore"):
        m.reclaim(["optimizer"], 13)
    setattr(b, operation, Mock(side_effect=RuntimeError("injected")))
    with pytest.raises(RuntimeError, match="injected"):
        if operation in ("map", "restore"):
            m.transition(["optimizer"], True)
        else:
            m.reclaim(["optimizer"], 13)
    with pytest.raises(RuntimeError, match="previously failed"):
        m.snapshot()


def test_inventory_replacement_while_unmapped_fails_closed():
    b = Backend()
    m = TaggedMemory(b, ["weights", "optimizer"])
    m.reclaim(["optimizer"], 13)
    b.data[2] = SimpleNamespace(tag="optimizer", handle=(0, 4, 2, 0))
    with pytest.raises(RuntimeError, match="changed"):
        m.snapshot()
    assert m.broken


def test_pressure_target_includes_existing_over_budget_debt():
    cfg = SimpleNamespace(device_budget_gib=56, safety_margin_gib=2)
    policy = ColocationMemoryPolicy(cfg, None, None, Mock())
    owner = Mock(return_value={"free_bytes": 14 * GIB, "total_bytes": 64 * GIB})
    policy._reclaim_shortfall(
        owner, {"free_bytes": 4 * GIB, "total_bytes": 64 * GIB}, 6 * GIB, "train"
    )
    owner.assert_called_once_with("reclaim", target_free_bytes=14 * GIB)


def test_insufficient_partial_reclaim_never_passes_budget_check():
    cfg = SimpleNamespace(device_budget_gib=None, safety_margin_gib=2)
    policy = ColocationMemoryPolicy(cfg, None, None, Mock())
    status = {"free_bytes": 4 * GIB, "total_bytes": 64 * GIB}
    with pytest.raises(RuntimeError, match="Insufficient memory"):
        policy._reclaim_shortfall(Mock(return_value=status), status, 6 * GIB, "train")


def test_workspace_parking_stops_at_deficit_and_can_grow_incrementally():
    b = Backend()
    for a in b.data.values():
        a.tag = "actor_workspace"
    b.segments = lambda: [
        dict(
            address=i,
            total_size=4,
            active_size=0,
            allocated_size=0,
            blocks=[dict(state="inactive", size=4)],
        )
        for i in b.data
    ]
    w = ParkedWorkspace(b)
    w.park(13)
    assert w.restore_bytes == 4 and len(b.contents) == 2
    w.park(17)
    assert w.restore_bytes == 8 and len(b.contents) == 1
    w.restore()
    assert b.free == 10 and len(b.contents) == 3


@pytest.mark.parametrize(
    "excluded",
    [
        None,
        "active",
        "pending",
        "parked",
        "unmapped",
        "foreign",
        "size",
        "backup",
        "stream",
        "missing_stream",
    ],
)
def test_workspace_credit_requires_inactive_mapped_private_segments(excluded):
    b = Backend()
    b.data = {1: b.data[1]}
    a = b.data[1]
    a.tag = "actor_workspace"
    b.torch = SimpleNamespace(
        npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(npu_stream=7))
    )
    segment = dict(
        stream=7,
        address=1,
        total_size=4,
        active_size=0,
        allocated_size=0,
        blocks=[dict(state="inactive", size=4)],
    )
    b.segments = lambda: [segment]
    w = ParkedWorkspace(b)
    if excluded == "active":
        segment["active_size"] = 4
    elif excluded == "pending":
        segment["blocks"][0]["state"] = "active_pending_free"
    elif excluded == "parked":
        w.parked = [a]
    elif excluded == "unmapped":
        b.residency = SimpleNamespace(_missing={1: (a, "discarded")})
    elif excluded == "foreign":
        a.tag = "actor_optimizer"
    elif excluded == "size":
        segment["total_size"] = 8
    elif excluded == "backup":
        a.cpu_backup_tensor = object()
    elif excluded == "stream":
        segment["stream"] = 8
    elif excluded == "missing_stream":
        segment.pop("stream")
    assert w.reusable_bytes() == (4 if excluded is None else 0)
    b.synchronize.assert_not_called()


@pytest.mark.parametrize("credit,expected", [(0, 19), (2, 17), (20, 15)])
def test_concurrent_credit_is_bounded_to_gradients(credit, expected):
    cfg = SimpleNamespace(
        device_budget_gib=None,
        safety_margin_gib=2,
        train_workspace_gib=8,
        inference_workspace_gib=3,
    )
    actor = Mock(
        return_value=dict(
            free_bytes=30 * GIB,
            restore_bytes=0,
            gradient_bytes=4 * GIB,
            future_optimizer_bytes=GIB,
            publication_workspace_bytes=GIB,
            reusable_workspace_bytes=credit * GIB,
        )
    )

    def rollout(action, **kwargs):
        if action == "status":
            return dict(
                restore_bytes=0,
                free_bytes=30 * GIB,
                total_bytes=64 * GIB,
                dynamic_kv=dict(mapped_bytes=GIB),
            )
        assert action == "kv_budget"
        assert kwargs["kv_budget_bytes"] == (31 - expected) * GIB
        return dict(free_bytes=30 * GIB, concurrent_admitted=True, capacity_blocks=100)

    assert ColocationMemoryPolicy(cfg, actor, rollout, Mock()).concurrent_train()


def test_default_pool_credit_rejects_unknown_live_private_and_foreign_segments():
    from areal.engine.colocation_memory import reusable_default_segments

    base = dict(
        segment_pool_id=(0, 0),
        is_expandable=False,
        device=0,
        stream=7,
        active_size=0,
        allocated_size=0,
        total_size=100,
        blocks=[dict(state="inactive", size=100)],
    )
    assert reusable_default_segments([base], 7, 0) == 100
    for change in [
        dict(segment_pool_id=(1, 0)),
        dict(is_expandable=True),
        dict(is_expandable=None),
        dict(segment_pool_id=()),
        dict(device=1),
        dict(stream=8),
        dict(active_size=1),
        dict(allocated_size=1),
        dict(blocks=[dict(state="active_awaiting_free", size=100)]),
        dict(blocks=[dict(state="inactive", size=99)]),
    ]:
        assert reusable_default_segments([dict(base, **change)], 7, 0) == 0


def test_resident_budget_retains_cache_but_preserves_other_reserves(monkeypatch):
    monkeypatch.setenv("AREAL_RESIDENT_BUDGET", "1")
    cfg = SimpleNamespace(
        device_budget_gib=None, safety_margin_gib=2, train_workspace_gib=16
    )
    state = dict(
        free_bytes=17 * GIB,
        total_bytes=64 * GIB,
        restore_bytes=0,
        future_optimizer_bytes=0,
        gradient_bytes=6 * GIB,
        publication_workspace_bytes=0,
        workspace_parked_bytes=0,
        reusable_workspace_bytes=3 * GIB,
        reusable_forward_bytes=10 * GIB,
        supports_partial_reclaim=True,
    )
    actor = Mock(return_value=state)
    rollout = Mock(
        side_effect=AssertionError("Resident budget must not re-enter inference")
    )
    policy = ColocationMemoryPolicy(cfg, actor, rollout, Mock())
    policy.train(resident=True)
    actor.assert_called_once_with("status")
    rollout.assert_not_called()
    # Publication/optimizer growth still must fit; reuse cannot hide these bytes.
    state["future_optimizer_bytes"] = 20 * GIB
    with pytest.raises(AssertionError, match="re-enter"):
        policy.train(resident=True)


def test_resident_budget_never_skips_restore_or_parked_state(monkeypatch):
    monkeypatch.setenv("AREAL_RESIDENT_BUDGET", "1")
    cfg = SimpleNamespace(
        device_budget_gib=None, safety_margin_gib=2, train_workspace_gib=4
    )
    state = dict(
        free_bytes=50 * GIB,
        total_bytes=64 * GIB,
        restore_bytes=1,
        future_optimizer_bytes=0,
        gradient_bytes=GIB,
        workspace_parked_bytes=1,
        reusable_workspace_bytes=GIB,
        reusable_forward_bytes=4 * GIB,
        supports_partial_reclaim=True,
    )
    actor = Mock(return_value=state)
    rollout = Mock(return_value=state)
    ColocationMemoryPolicy(cfg, actor, rollout, Mock()).train(resident=True)
    assert actor.call_args.args == ("restore",)
    rollout.assert_called_once_with("status")


def test_resident_cache_credit_cannot_hide_exceeded_device_cap(monkeypatch):
    monkeypatch.setenv("AREAL_RESIDENT_BUDGET", "1")
    cfg = SimpleNamespace(
        device_budget_gib=20, safety_margin_gib=0, train_workspace_gib=16
    )
    state = dict(
        free_bytes=17 * GIB,
        total_bytes=64 * GIB,
        restore_bytes=0,
        future_optimizer_bytes=0,
        gradient_bytes=6 * GIB,
        reusable_workspace_bytes=6 * GIB,
        reusable_forward_bytes=16 * GIB,
        supports_partial_reclaim=True,
    )
    rollout = Mock(side_effect=AssertionError("must use pressure path"))
    with pytest.raises(AssertionError, match="pressure path"):
        ColocationMemoryPolicy(cfg, Mock(return_value=state), rollout, Mock()).train(
            resident=True
        )
