# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from areal.engine.vllm_ext.dynamic_kv import SparseKVPages, pending_blocks


class Backend:
    page = 16

    def __init__(self):
        self.addresses = set()
        self.events = []
        self.free = 10000

    def synchronize(self):
        self.events.append("sync")

    def free_bytes(self):
        return self.free

    def map(self, a):
        self.events.append(("map", a))
        self.addresses.add(a)

    def unmap(self, a):
        self.events.append(("unmap", a))
        self.addresses.remove(a)


def make(**kwargs):
    b = Backend()
    ts = [
        SimpleNamespace(data_ptr=lambda: 1024),
        SimpleNamespace(data_ptr=lambda: 2048),
    ]
    p = SparseKVPages(b, ts, 4, 20, 32, **kwargs)
    return b, p


def test_reserve_only_null_then_grow_and_reclaim():
    b, p = make(keep_idle_pages=0)
    assert p.snapshot()["mapped_bytes"] == 32
    p.reconcile({1, 7, 17})
    assert p.mapped == {0, 1, 4}
    assert p.snapshot()["virtual_bytes"] == 160
    p.reconcile({17})
    assert p.mapped == {0, 4} and 1040 not in b.addresses
    p.reconcile(set(), trim=True)
    assert p.mapped == {0}


def test_no_sync_without_mapping_change():
    b, p = make()
    b.events.clear()
    p.reconcile({1, 2})
    assert b.events == []


def test_live_slab_never_freed_and_idle_cache_bounded():
    b, p = make(keep_idle_pages=1)
    p.reconcile({4, 8, 12})
    p.reconcile({8})
    assert 2 in p.mapped and len(p.mapped) == 3
    p.reconcile({8}, trim=True)
    assert p.mapped == {0, 2}


def test_bad_ids_fail_without_mutation():
    b, p = make()
    before = set(b.addresses)
    with pytest.raises(ValueError):
        p.reconcile({20})
    assert b.addresses == before


def test_budget_rejects_before_mapping_and_fails_closed():
    b, p = make()
    b.free = 32
    b.events.clear()
    with pytest.raises(MemoryError):
        p.reconcile({8})
    assert not any(isinstance(e, tuple) and e[0] == "map" for e in b.events)
    with pytest.raises(RuntimeError):
        p.reconcile({0})


def test_native_failure_is_terminal():
    b, p = make()

    def fail(a):
        raise RuntimeError("native map failed")

    b.map = fail
    with pytest.raises(RuntimeError):
        p.reconcile({8})
    assert p.broken


def test_pending_preserves_unscheduled_and_maps_zero_targets():
    old = {
        "unscheduled": SimpleNamespace(block_ids=([10, 11],)),
        "old": SimpleNamespace(block_ids=([2],)),
    }
    out = SimpleNamespace(
        kv_cache_block_copies=[],
        scheduled_new_reqs=[SimpleNamespace(block_ids=([3],))],
        scheduled_cached_reqs=SimpleNamespace(new_block_ids=[([4],), None]),
        new_block_ids_to_zero=[5],
    )
    assert pending_blocks(old, out) == {2, 3, 4, 5, 10, 11}
    out.kv_cache_block_copies = [(1, 2)]
    with pytest.raises(ValueError):
        pending_blocks(old, out)


def test_last_block_has_mapped_padding_page():
    backend, pool = make(keep_idle_pages=0)
    pool.reconcile({19})
    assert pool.mapped == {0, 4, 5, 6}
    pool.reconcile({0}, trim=True)
    assert pool.mapped == {0}
