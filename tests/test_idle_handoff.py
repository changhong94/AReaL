"""Per-device handoff gates, independent preparation, and failure propagation."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.engine.idle_handoff import atomic_json, device_key, handoff


def engine(index, ready=True):
    value = SimpleNamespace(
        rank=index,
        is_offload=False,
        device=SimpleNamespace(index=index),
        _adaptive_memory=object(),
        _idle_dram=SimpleNamespace(pending_import=(None, None, 1)),
        lr_scheduler=SimpleNamespace(last_epoch=2),
        set_version=Mock(),
        onload=Mock(),
        idle_training_state=Mock(return_value={}),
        _idle_stage_future=Future(),
    )
    if ready:
        value._idle_stage_future.set_result({"snapshot_generation": 3})
    return value


def config(path, devices):
    atomic_json(
        path / "transfer.json",
        dict(
            token="unique",
            direction="single_to_multi",
            version=2,
            scheduler_epoch=2,
            snapshot_generation=3,
            source_devices=devices,
        ),
    )


def test_ready_rank_restores_without_waiting_for_other_rank(tmp_path):
    first, second = engine(0), engine(1, ready=False)
    config(tmp_path, [device_key(first)])
    atomic_json(tmp_path / f"released-{device_key(first)}.json", {"token": "unique"})
    completed = threading.Event()
    first.onload.side_effect = completed.set
    with ThreadPoolExecutor(max_workers=2) as jobs:
        a = jobs.submit(handoff, first, "restore", tmp_path)
        b = jobs.submit(handoff, second, "restore", tmp_path)
        assert completed.wait(5)
        a.result(timeout=5)
        second.onload.assert_not_called()
        second._idle_stage_future.set_result({"snapshot_generation": 3})
        b.result(timeout=5)
    second.onload.assert_called_once()


@pytest.mark.parametrize("failure", ["stale_token", "peer_failure", "stale_generation"])
def test_invalid_release_or_generation_never_restores(tmp_path, failure):
    value = engine(0)
    config(tmp_path, [device_key(value)])
    if failure == "stale_token":
        atomic_json(tmp_path / f"released-{device_key(value)}.json", {"token": "old"})
    elif failure == "peer_failure":
        (tmp_path / "failed.json").touch()
    else:
        value._idle_stage_future = Future()
        value._idle_stage_future.set_result({"snapshot_generation": 2})
    with pytest.raises(RuntimeError):
        handoff(value, "restore", tmp_path)
    value.onload.assert_not_called()


def test_source_failure_does_not_authorize_device_restore(tmp_path):
    value = engine(0)
    value.colocation_memory = Mock(side_effect=RuntimeError("release failed"))
    config(tmp_path, [device_key(value)])
    with pytest.raises(RuntimeError, match="release failed"):
        handoff(value, "release", tmp_path)
    assert (tmp_path / "failed.json").exists()
    assert not (tmp_path / f"released-{device_key(value)}.json").exists()


def test_parked_source_release_does_not_finish_or_restore_training(tmp_path):
    value = engine(0)
    value.is_offload = True
    value.colocation_memory, value.offload = Mock(), Mock()
    config(tmp_path, [device_key(value)])
    handoff(value, "release", tmp_path)
    value.colocation_memory.assert_called_once_with("park_cache")
    value.onload.assert_not_called()
    assert (tmp_path / f"released-{device_key(value)}.json").exists()
