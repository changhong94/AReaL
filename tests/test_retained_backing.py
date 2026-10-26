from threading import Event, Thread
from types import SimpleNamespace

import pytest

from areal.engine.colocation_memory import TaggedMemory


class Backend:
    def __init__(self):
        self.data = {
            1: SimpleNamespace(
                tag="weights", handle=(0, 4, 1, 0), cpu_backup_tensor=None
            ),
            2: SimpleNamespace(tag="kv", handle=(0, 4, 2, 0), cpu_backup_tensor=None),
        }
        self.contents = {1: bytearray(b"abcd"), 2: bytearray(b"1234")}
        self.copies = 0

    def inventory(self):
        return self.data

    def synchronize(self):
        pass

    def free_memory(self):
        return 100, 200

    def backup(self, a):
        self.copies += 1
        return self.contents[a.handle[2]][:]

    def map(self, a):
        self.contents[a.handle[2]] = bytearray(4)

    def unmap(self, a):
        del self.contents[a.handle[2]]

    def restore(self, a):
        self.contents[a.handle[2]][:] = a.cpu_backup_tensor

    def verify_backup(self, a, backup):
        if self.contents[a.handle[2]] != backup:
            raise RuntimeError("stale backup")


@pytest.fixture
def memory(monkeypatch):
    monkeypatch.setenv("AREAL_RETAIN_DRAM", "1")
    monkeypatch.setenv("AREAL_IDLE_FULL_VERIFY", "1")
    return TaggedMemory(Backend(), ["weights", "kv"], retained_tags=("weights",))


def test_clean_roundtrips_reuse_backup_without_recopy(memory):
    for _ in range(3):
        memory.transition(["weights"], False)
        memory.transition(["weights"], True)
        assert memory.backend.contents[1] == b"abcd"
        assert memory.backend.data[1].cpu_backup_tensor is None
        assert memory.snapshot()["cpu_backup_bytes"] == 4
    assert memory.backend.copies == 1


def test_dirty_background_refresh_restores_latest_contents(memory):
    memory.start_retained_backup()
    memory.wait_retained_backup()
    memory.mark_dirty(["weights"])
    memory.backend.contents[1][:] = b"WXYZ"
    memory.start_retained_backup()
    memory.wait_retained_backup()
    memory.transition(["weights"], False)
    memory.transition(["weights"], True)
    assert memory.backend.contents[1] == b"WXYZ"
    assert memory.backend.copies == 2


def test_unmapped_cpu_staging_becomes_retained_generation(memory):
    memory.transition(["weights"], False)
    memory.backend.data[1].cpu_backup_tensor[:] = b"next"
    memory.transition(["weights"], True)
    memory.transition(["weights"], False)
    memory.transition(["weights"], True)
    assert memory.backend.contents[1] == b"next"
    assert memory.backend.copies == 1


def test_address_reuse_cannot_reuse_old_allocation_backup(memory):
    memory.start_retained_backup()
    memory.wait_retained_backup()
    memory.backend.data[1] = SimpleNamespace(
        tag="weights", handle=(0, 4, 1, 0), cpu_backup_tensor=None
    )
    memory.backend.contents[1][:] = b"new!"
    memory.transition(["weights"], False)
    memory.transition(["weights"], True)
    assert memory.backend.contents[1] == b"new!"
    assert memory.backend.copies == 2


def test_mutation_fence_joins_incomplete_background_copy(memory):
    entered, release, marked = Event(), Event(), Event()
    original = memory.backend.backup

    def blocked(a):
        entered.set()
        assert release.wait(5)
        return original(a)

    memory.backend.backup = blocked
    memory.start_retained_backup()
    assert entered.wait(5)

    def mutate():
        memory.mark_dirty(["weights"])
        marked.set()

    worker = Thread(target=mutate)
    worker.start()
    assert not marked.wait(0.1)
    release.set()
    worker.join(5)
    assert marked.is_set()
    assert memory.retained.get(memory.backend.data[1]) is None


def test_failed_copy_poisoned_before_unmap(memory):
    def fail(a):
        raise RuntimeError("copy failure")

    memory.backend.backup = fail
    memory.start_retained_backup()
    with pytest.raises(RuntimeError, match="copy failure"):
        memory.transition(["weights"], False)
    assert 1 in memory.backend.contents
    assert memory.broken


def test_untracked_kv_is_always_copied(memory):
    for _ in range(2):
        memory.transition(["kv"], False)
        memory.transition(["kv"], True)
    assert memory.backend.copies == 2


def test_validation_detects_missing_dirty_hook(memory):
    memory.start_retained_backup()
    memory.wait_retained_backup()
    memory.backend.contents[1][:] = b"bad!"
    with pytest.raises(RuntimeError, match="stale backup"):
        memory.transition(["weights"], False)
    assert memory.broken


def test_discard_does_not_reuse_obsolete_backup(memory):
    memory.start_retained_backup()
    memory.wait_retained_backup()
    memory.transition(["weights"], False, preserve=False)
    memory.transition(["weights"], True)
    memory.backend.contents[1][:] = b"new!"
    memory.transition(["weights"], False)
    memory.transition(["weights"], True)
    assert memory.backend.contents[1] == b"new!"
    assert memory.backend.copies == 2
