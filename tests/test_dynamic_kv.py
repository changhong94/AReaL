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


def test_preserved_live_budget_keeps_live_pages_and_rejects_too_small_capacity():
    backend, pool = make(keep_idle_pages=0)
    pool.reconcile({8})
    pool.pause_preserved()
    before = set(backend.addresses)
    assert pool.set_budget(32) == 0
    assert backend.addresses == before
    assert pool.set_budget(96) == 12
    assert pool.live == {0, 8}
    assert pool.mapped == {0, 2}
    assert pool.set_budget(None) == 20


def test_live_unpaused_budget_is_rejected():
    _, pool = make()
    pool.reconcile({8})
    with pytest.raises(RuntimeError, match="preserved-paused"):
        pool.set_budget(96)


def test_retained_export_mixed_hbm_dram_preserves_bytes_and_mappings():
    import ctypes

    page = 16
    buffers = [ctypes.create_string_buffer(bytes(range(i, i + 80))) for i in (0, 80)]

    class CopyBackend(Backend):
        def copy_to_device(self, dst, src, size, *, host):
            self.events.append(("copy", host))
            ctypes.memmove(dst, src, size)

    backend = CopyBackend()
    # Use aligned storage, as required by the production allocator.
    storage = [ctypes.create_string_buffer(128) for _ in buffers]
    addresses = [(ctypes.addressof(b) + page - 1) // page * page for b in storage]
    for address, content in zip(addresses, buffers, strict=True):
        ctypes.memmove(address, content, 80)
    tensors = [SimpleNamespace(data_ptr=lambda a=a: a) for a in addresses]
    pool = SparseKVPages(backend, tensors, 4, 20, 0)
    pool.reconcile({1, 5, 6})
    pool.pause_preserved()
    backups = [
        ctypes.create_string_buffer(ctypes.string_at(a + page, page)) for a in addresses
    ]
    pool.swapped[1] = [
        SimpleNamespace(data_ptr=lambda b=b: ctypes.addressof(b)) for b in backups
    ]
    pool.mapped.remove(1)
    # Poison unmapped memory: reading tensor views would give incorrect bytes.
    for address in addresses:
        ctypes.memset(address + page, 255, page)
    destination = ctypes.create_string_buffer(24)
    mapped, live = set(pool.mapped), set(pool.live)
    pool.copy_retained_blocks([6, 1, 5], ctypes.addressof(destination))
    expected = b"".join(
        bytes(range(base + block * 4, base + block * 4 + 4))
        for base in (0, 80)
        for block in (6, 1, 5)
    )
    assert destination.raw == expected
    assert pool.mapped == mapped and pool.live == live and pool.paused
    assert len(pool.swapped) == 1
    assert [e for e in backend.events if isinstance(e, tuple) and e[0] == "copy"] == [
        ("copy", True),
        ("copy", False),
        ("copy", True),
    ] * 2


def test_retained_export_rejects_unpaused_or_missing_backing_before_copy():
    backend, pool = make()
    with pytest.raises(RuntimeError, match="pause"):
        pool.copy_retained_blocks([1], 0)
    pool.reconcile({5})
    pool.pause_preserved()
    pool.mapped.remove(1)
    with pytest.raises(RuntimeError, match="backing"):
        pool.copy_retained_blocks([5], 0)
    assert pool.paused


def test_reclaim_succeeds_below_reserve_without_new_mapping():
    backend, pool = make(keep_idle_pages=0)
    pool.reconcile({4, 8})
    backend.free = 0
    backend.events.clear()
    pool.reconcile({0}, trim=True)
    assert pool.mapped == {0}
    assert pool.live == {0}
    assert not pool.broken
    assert not any(isinstance(e, tuple) and e[0] == "map" for e in backend.events)
