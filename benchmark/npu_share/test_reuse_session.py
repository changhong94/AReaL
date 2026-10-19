"""Development tool tests: immutable initial weights and independent run order."""

import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "initial_weights", Path(__file__).parent / "templates/initial_weights.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_initial_snapshot_coexists_with_mutable_offload_backup():
    model = torch.nn.Linear(4, 2)
    model.register_buffer("scalar", torch.tensor(3.0))
    original = {k: v.clone() for k, v in model.state_dict().items()}
    snapshot = MODULE.InitialWeights(model)
    with torch.no_grad():
        model.weight.add_(10)
    current_backup = {k: v.clone() for k, v in model.state_dict().items()}
    current_backup["weight"].add_(20)
    result = snapshot.restore(model)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert not torch.equal(current_backup["weight"], model.weight)
    assert result["initial_sha256"] == result["restored_sha256"]
    assert snapshot.bytes == sum(
        v.numel() * v.element_size() for v in original.values()
    )


def test_corrupt_initial_snapshot_fails_before_copying():
    model = torch.nn.Linear(2, 2)
    snapshot = MODULE.InitialWeights(model)
    snapshot._tensors["weight"].add_(1)
    with pytest.raises(RuntimeError, match="Immutable"):
        snapshot.restore(model)


def test_changed_model_schema_cannot_reuse_snapshot():
    snapshot = MODULE.InitialWeights(torch.nn.Linear(2, 2))
    with pytest.raises(RuntimeError, match="schema"):
        snapshot.restore(torch.nn.Linear(3, 3))


def test_snapshot_path_maps_only_the_current_session_output():
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "templates"))
    from reuse_paths import snapshot_path

    trace = "/session/repetitions/01/latest/request-events"
    assert (
        snapshot_path("/run-output/checkpoints/v1", trace)
        == "/session/repetitions/01/latest/checkpoints/v1"
    )
    assert snapshot_path("/model", trace) == "/model"
    with pytest.raises(ValueError):
        snapshot_path("/run-output/../escape", trace)
    with pytest.raises(ValueError):
        snapshot_path("/run-output/checkpoints", "/other/request-events")
