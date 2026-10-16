# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areal.engine.megatron_utils import shared_publication as publication
from areal.engine.weight_snapshot import open_snapshot


@pytest.fixture
def setup(monkeypatch, tmp_path):
    engine = SimpleNamespace(
        config=SimpleNamespace(use_lora=False, use_merged_lora=False),
        quantization_config=None,
        tf_config=SimpleNamespace(num_moe_experts=None),
        bridge_cls="mbridge",
        parallel_strategy=SimpleNamespace(tensor_parallel_size=1),
        dtype=torch.float32,
        weight_update_master_addr="localhost",
        weight_update_master_port=1234,
        weight_update_group_name="fixed",
        rollout_engine=Mock(),
    )
    helper = publication.SharedMegatronPublication(engine)
    params = [(str(i), torch.arange(8).float() + i) for i in range(3)]
    helper._parameters = Mock(return_value=params)
    meta = SimpleNamespace(
        path=str(tmp_path / "weights"),
        version=4,
        npu_share_fixed=True,
        npu_share_snapshot=True,
        weight_chunked_mem_mb=32 / 1024 / 1024,
    )
    return engine, helper, params, meta


def test_targeted_publication_and_snapshot_match(setup):
    engine, helper, params, meta = setup
    received = {}

    def send(meta, bucket):
        received.update({name: tensor.clone() for name, tensor in bucket})
        bucket.clear()  # Match the real Megatron bucket method's ownership contract.

    engine._update_bucket_weights_from_distributed = Mock(side_effect=send)
    helper.publish_fixed(meta)
    helper.wait(meta.version)
    with open_snapshot(meta.path, meta.version) as weights:
        saved = {name: tensor.clone() for name, tensor in weights}
    assert set(saved) == set(received) == {name for name, _ in params}
    for name, tensor in params:
        torch.testing.assert_close(saved[name], tensor)
        torch.testing.assert_close(received[name], tensor)
    assert engine._update_bucket_weights_from_distributed.call_count == 3
    engine.rollout_engine.pause_generation.assert_not_called()
    engine.rollout_engine.continue_generation.assert_not_called()
    # Once joined, changing source weights cannot change the owned host snapshot.
    params[0][1].add_(100)
    with open_snapshot(meta.path, meta.version) as weights:
        assert dict(weights)["0"][0].item() == 0


def test_version_and_model_parallel_guards(setup, monkeypatch):
    _, helper, _, meta = setup
    with pytest.raises(RuntimeError, match="No shared snapshot"):
        helper.wait(4)
    helper._validate(meta)
    helper.engine.parallel_strategy.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="model-parallel"):
        helper.publish_fixed(meta)


def test_rejects_non_targeted_publication(setup):
    _, helper, _, meta = setup
    meta.npu_share_fixed = False
    with pytest.raises(ValueError, match="fixed-receiver"):
        helper.publish_fixed(meta)


def test_export_initial_snapshot(setup):
    _, helper, params, meta = setup
    manifest = helper.export(meta)
    assert manifest["version"] == 4
    with open_snapshot(meta.path, 4) as weights:
        for (name, actual), (expected_name, expected) in zip(weights, params):
            assert name == expected_name
            torch.testing.assert_close(actual, expected)


def test_transport_failure_joins_snapshot(setup, monkeypatch):
    engine, helper, _, meta = setup
    job = Mock()
    monkeypatch.setattr(publication, "IndependentSnapshot", Mock(return_value=job))
    failure = RuntimeError("receiver failed")
    engine._update_bucket_weights_from_distributed = Mock(side_effect=failure)
    with pytest.raises(RuntimeError, match="receiver failed"):
        helper.publish_fixed(meta)
    job.abort.assert_called_once_with(failure)


def test_snapshot_failure_propagates(setup, monkeypatch):
    engine, helper, _, meta = setup
    job = Mock(version=4)
    job.wait.side_effect = OSError("host storage full")
    monkeypatch.setattr(publication, "IndependentSnapshot", Mock(return_value=job))
    engine._update_bucket_weights_from_distributed = Mock()
    helper.publish_fixed(meta)
    with pytest.raises(RuntimeError, match="version mismatch"):
        helper.wait(5)
    with pytest.raises(OSError, match="host storage full"):
        helper.wait(4)
