# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areal.api import WeightUpdateMeta
from areal.engine import fsdp_engine
from areal.engine.weight_snapshot import MANIFEST, open_snapshot


@pytest.fixture
def engine(monkeypatch):
    """Exercise real materialization/casting with CPU tensors and mocked transport."""
    obj = object.__new__(fsdp_engine.FSDPEngine)
    obj.config = SimpleNamespace(use_lora=False, dtype="bfloat16")
    obj._initialized = True
    obj._cpu_group = None
    obj.weight_update_master_addr = "localhost"
    obj.weight_update_master_port = 1
    obj.weight_update_group_names = ["test"]
    obj.weight_update_groups = [object()]
    monkeypatch.setattr(fsdp_engine.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(fsdp_engine.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(fsdp_engine.dist, "barrier", Mock())
    monkeypatch.setattr(
        fsdp_engine,
        "current_platform",
        SimpleNamespace(device_type="cpu", synchronize=lambda: None),
    )
    return obj


def test_fused_publication_sends_and_snapshots_identical_cast_weights(engine, tmp_path):
    """Snapshot copies overlap pending buckets and use exactly the sent values."""
    params = [
        (str(i), torch.nn.Parameter(torch.arange(16).float() + i)) for i in range(4)
    ]
    engine._get_model_name_parameters = Mock(return_value=iter(params))
    path = tmp_path / "weights"
    received = {}
    pending = []
    observed_pending = []

    def send(meta, tensors, stream=None):
        received.update({name: t.clone() for name, t in tensors})
        pending.append(tensors)
        observed_pending.append(not (path / MANIFEST).exists())
        return tensors

    def wait(bucket):
        assert bucket in pending
        pending.remove(bucket)

    engine._update_bucket_weights_from_distributed_async = send
    engine._wait_pending_weight_update_bucket = wait
    meta = WeightUpdateMeta(
        type="xccl",
        path=str(path),
        version=3,
        npu_share_fixed=True,
        npu_share_snapshot=True,
        weight_chunked_mem_mb=32 / (1024 * 1024),
    )
    engine._update_weights_from_distributed(meta)
    engine.wait_shared_snapshot(3)
    assert not pending and any(observed_pending)
    engine._get_model_name_parameters.assert_called_once()
    with open_snapshot(str(path), 3) as weights:
        actual = dict(weights)
        assert actual.keys() == received.keys()
        for name, param in params:
            torch.testing.assert_close(actual[name], received[name], rtol=0, atol=0)
            torch.testing.assert_close(
                actual[name], param.to(torch.bfloat16), rtol=0, atol=0
            )


def test_snapshot_failure_drains_already_submitted_bucket(
    engine, monkeypatch, tmp_path
):
    """A failed materialization cannot leave HCCL reading released tensors."""
    params = [(str(i), torch.nn.Parameter(torch.ones(16))) for i in range(4)]
    engine._get_model_name_parameters = lambda meta: iter(params)
    real = engine._get_full_tensor
    calls = 0

    def materialize(param):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("materialize failed")
        return real(param)

    engine._get_full_tensor = materialize
    bucket = object()
    engine._update_bucket_weights_from_distributed_async = Mock(return_value=bucket)
    engine._wait_pending_weight_update_bucket = Mock()
    path = tmp_path / "weights"
    meta = WeightUpdateMeta(
        type="xccl",
        path=str(path),
        version=1,
        npu_share_fixed=True,
        npu_share_snapshot=True,
        weight_chunked_mem_mb=32 / (1024 * 1024),
    )
    with pytest.raises(RuntimeError, match="materialize failed"):
        engine._update_weights_from_distributed(meta)
    engine._wait_pending_weight_update_bucket.assert_called_once_with(bucket)
    assert not engine._shared_snapshot._thread.is_alive()


def test_fused_snapshot_rejects_multiple_actor_ranks(engine, monkeypatch, tmp_path):
    monkeypatch.setattr(fsdp_engine.dist, "get_world_size", lambda: 2)
    engine._get_model_name_parameters = lambda meta: iter([])
    with pytest.raises(ValueError, match="one full-model actor"):
        engine._update_weights_from_distributed(
            WeightUpdateMeta(
                type="xccl",
                path=str(tmp_path / "w"),
                version=1,
                npu_share_fixed=True,
                npu_share_snapshot=True,
            )
        )


def test_final_hccl_failure_is_not_success_even_with_complete_snapshot(
    engine, tmp_path
):
    """An owned snapshot is not a receiver ACK; publication still must fail."""
    engine._get_model_name_parameters = lambda meta: iter(
        [("w", torch.nn.Parameter(torch.ones(4)))]
    )
    engine._update_bucket_weights_from_distributed_async = Mock(return_value=object())
    engine._wait_pending_weight_update_bucket = Mock(
        side_effect=RuntimeError("receiver failed")
    )
    path = tmp_path / "weights"
    with pytest.raises(RuntimeError, match="receiver failed"):
        engine._update_weights_from_distributed(
            WeightUpdateMeta(
                type="xccl",
                path=str(path),
                version=1,
                npu_share_fixed=True,
                npu_share_snapshot=True,
            )
        )
    assert not engine._shared_snapshot._thread.is_alive()


def test_fixed_ack_does_not_wait_for_snapshot(engine, monkeypatch, tmp_path):
    from threading import Event

    from areal.engine import weight_snapshot

    started, release = Event(), Event()
    real = weight_snapshot.write_snapshot

    def slow(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(weight_snapshot, "write_snapshot", slow)
    engine._get_model_name_parameters = lambda meta: iter(
        [(str(i), torch.nn.Parameter(torch.ones(4))) for i in range(10)]
    )
    engine._update_bucket_weights_from_distributed_async = Mock(return_value=object())
    engine._wait_pending_weight_update_bucket = Mock()
    meta = WeightUpdateMeta(
        type="xccl",
        path=str(tmp_path / "w"),
        version=2,
        npu_share_fixed=True,
        npu_share_snapshot=True,
    )
    try:
        engine._update_weights_from_distributed(meta)
        assert started.wait(1)
        assert not engine._shared_snapshot._future.done()
        with pytest.raises(RuntimeError, match="version mismatch"):
            engine.wait_shared_snapshot(3)
    finally:
        release.set()
        engine.wait_shared_snapshot(2)


def test_writer_failure_surfaces_at_snapshot_join(engine, monkeypatch, tmp_path):
    from areal.engine import weight_snapshot

    monkeypatch.setattr(
        weight_snapshot, "write_snapshot", Mock(side_effect=OSError("snapshot full"))
    )
    engine._get_model_name_parameters = lambda meta: iter(
        [(str(i), torch.nn.Parameter(torch.ones(4))) for i in range(10)]
    )
    engine._update_bucket_weights_from_distributed_async = Mock(return_value=object())
    engine._wait_pending_weight_update_bucket = Mock()
    engine._update_weights_from_distributed(
        WeightUpdateMeta(
            type="xccl",
            path=str(tmp_path / "w"),
            version=2,
            npu_share_fixed=True,
            npu_share_snapshot=True,
        )
    )
    with pytest.raises(OSError, match="snapshot full"):
        engine.wait_shared_snapshot(2)
    assert not engine._shared_snapshot._thread.is_alive()


def test_destroy_cleans_up_after_snapshot_failure(engine, monkeypatch):
    job = Mock()
    job.wait.side_effect = OSError("snapshot full")
    engine._shared_snapshot = job
    engine.model = Mock()
    engine.optimizer = Mock()
    engine._per_layer_optim_wrapper = None
    monkeypatch.setattr(fsdp_engine.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        fsdp_engine.current_platform, "empty_cache", Mock(), raising=False
    )
    with pytest.raises(OSError, match="snapshot full"):
        engine.destroy()
    job.wait.assert_called_once()
    assert not hasattr(engine, "model")
    assert not hasattr(engine, "optimizer")
    assert not engine._initialized
    assert engine._shared_snapshot is None


def test_memory_switch_waits_before_releasing_storage(engine):
    job = Mock()
    job.wait.side_effect = OSError("snapshot failed")
    engine._shared_snapshot = job
    engine._adaptive_memory = Mock()
    with pytest.raises(OSError, match="snapshot failed"):
        engine.colocation_memory("offload")
    engine._adaptive_memory.transition.assert_not_called()
