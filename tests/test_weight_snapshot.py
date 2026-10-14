# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
import torch

from areal.api import WeightUpdateMeta
from areal.engine.vllm_ext.generation_gate import update_weights_while_paused
from areal.engine.vllm_remote import VLLMBackend
from areal.engine.weight_snapshot import (
    DATA,
    MANIFEST,
    IndependentSnapshot,
    open_snapshot,
    validate_loaded_weights,
    write_snapshot,
)


def test_independent_snapshot_borrows_storage_and_owns_bytes_after_join(
    tmp_path, monkeypatch
):
    from threading import Event

    from areal.engine import weight_snapshot

    started, release = Event(), Event()
    real = weight_snapshot.write_snapshot
    original = torch.arange(16, dtype=torch.float32)

    def delayed(directory, version, parameters, *args):
        assert parameters[0][1].data_ptr() == original.data_ptr()
        started.set()
        assert release.wait(5)
        return real(directory, version, parameters, *args)

    monkeypatch.setattr(weight_snapshot, "write_snapshot", delayed)
    path = str(tmp_path / "independent")
    job = IndependentSnapshot(path, 4, [("w", original)], torch.bfloat16)
    try:
        assert started.wait(2)
        assert not job._future.done()
    finally:
        release.set()
        job.wait()
    assert job._parameters == []
    original.fill_(-1)
    with open_snapshot(path, 4) as weights:
        torch.testing.assert_close(
            dict(weights)["w"], torch.arange(16, dtype=torch.bfloat16), rtol=0, atol=0
        )


def test_independent_snapshot_single_rank_dtensor_roundtrip(tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group(
        "gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1,))
        original = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        param = distribute_tensor(original, mesh, [Shard(0)])
        path = str(tmp_path / "dtensor")
        job = IndependentSnapshot(path, 2, [("w", param)], torch.bfloat16)
        job.wait()
        with open_snapshot(path, 2) as weights:
            torch.testing.assert_close(
                dict(weights)["w"], original.to(torch.bfloat16), rtol=0, atol=0
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("loaded", [None, {"a"}, set()])
def test_incomplete_loader_ack_is_rejected(loaded):
    with pytest.raises(RuntimeError, match="Snapshot"):
        validate_loaded_weights({"a", "b"}, loaded)


def test_complete_loader_ack_is_accepted():
    validate_loaded_weights({"a", "b"}, {"a", "b", "alias"})


def make_snapshot(path, version=3):
    params = [
        ("weight", torch.arange(24, dtype=torch.float32).reshape(4, 6).t()),
        ("scalar", torch.tensor(1.25)),
        ("counter", torch.tensor([2, 4], dtype=torch.int64)),
    ]

    def materialize(p):
        return p.to(torch.bfloat16) if p.is_floating_point() else p

    write_snapshot(str(path), version, params, torch.bfloat16, materialize)
    return params


def test_snapshot_round_trip_is_owned_and_preserves_all_values(tmp_path):
    path = tmp_path / "snapshot"
    params = make_snapshot(path)
    expected = {
        name: p.to(torch.bfloat16).clone() if p.is_floating_point() else p.clone()
        for name, p in params
    }
    for _, p in params:
        p.zero_()
    with open_snapshot(str(path), 3) as weights:
        actual = dict(weights)
        assert actual.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        actual["weight"].zero_()
    # Private reader mappings cannot corrupt another consumer's snapshot.
    with open_snapshot(str(path), 3) as weights:
        torch.testing.assert_close(
            dict(weights)["weight"], expected["weight"], rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "fault",
    ["version", "bounds", "dtype", "shape", "duplicate", "truncated", "missing"],
)
def test_invalid_snapshot_rejected_before_exposing_weights(tmp_path, fault):
    path = tmp_path / "snapshot"
    make_snapshot(path)
    meta = json.loads((path / MANIFEST).read_text())
    if fault == "version":
        meta["version"] = 4
    elif fault == "bounds":
        meta["tensors"][1]["offset"] = 0
    elif fault == "dtype":
        meta["tensors"][0]["dtype"] = "unknown"
    elif fault == "shape":
        meta["tensors"][0]["shape"] = [-1]
    elif fault == "duplicate":
        meta["tensors"][1]["name"] = "weight"
    elif fault == "truncated":
        (path / DATA).write_bytes(b"x")
    elif fault == "missing":
        (path / DATA).unlink()
    (path / MANIFEST).write_text(json.dumps(meta))
    with pytest.raises((ValueError, FileNotFoundError)):
        with open_snapshot(str(path), 3):
            pytest.fail("invalid snapshot exposed to loader")


def test_failed_copy_never_publishes_manifest(tmp_path):
    path = tmp_path / "snapshot"

    def fail(_):
        raise RuntimeError("copy failed")

    with pytest.raises(RuntimeError, match="copy failed"):
        write_snapshot(str(path), 1, [("weight", torch.ones(2))], torch.float32, fail)
    assert not (path / MANIFEST).exists()
    assert not (path / DATA).exists()


def test_snapshot_version_is_forwarded_to_worker_without_resuming_external_pause():
    async def run():
        gate = asyncio.Event()
        engine = AsyncMock()
        engine.collective_rpc.return_value = [(True, "ok")]
        await update_weights_while_paused(engine, gate, "/snapshot", snapshot_version=7)
        engine.collective_rpc.assert_awaited_once_with(
            "areal_update_weights", args=("/snapshot", 7)
        )
        engine.resume_generation.assert_not_awaited()

    asyncio.run(run())


def test_backend_requires_version_and_marks_only_snapshot_requests():
    backend = object.__new__(VLLMBackend)
    meta = WeightUpdateMeta(
        type="disk", path="/snapshot", version=7, npu_share_snapshot=True
    )
    req = backend.build_disk_weight_update_requests(meta).requests[0]
    assert req.payload == {"model_path": "/snapshot", "snapshot_version": 7}
    meta.npu_share_snapshot = False
    assert backend.build_disk_weight_update_requests(meta).requests[0].payload == {
        "model_path": "/snapshot"
    }
    meta.npu_share_snapshot = True
    meta.version = None
    with pytest.raises(ValueError, match="version"):
        backend.build_disk_weight_update_requests(meta)
