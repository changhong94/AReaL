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
