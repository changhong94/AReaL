"""Model-only helper refresh must not overwrite optimizer state."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from areal.engine.idle_helper import stage_model
from areal.engine.idle_state import manifest


@pytest.mark.parametrize("initialized_optimizer", [False, True])
def test_helper_refresh_preserves_optimizer_and_checks_generation(
    initialized_optimizer,
):
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        root = Path(directory)
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters())
        weight = model.weight
        if initialized_optimizer:
            optimizer.state[weight]["exp_avg"] = torch.ones_like(weight) * 7
        values = {"weight/weight": torch.ones_like(weight) * 3}
        layout = manifest(values)
        (root / "layout.json").write_text(json.dumps(layout))
        data = torch.from_file(
            str(root / "state.bin"),
            shared=True,
            size=layout["bytes"],
            dtype=torch.uint8,
        )
        entry = layout["entries"]["weight/weight"]
        data.narrow(0, entry["offset"], entry["size"]).view(weight.dtype).reshape(
            weight.shape
        ).copy_(values["weight/weight"])
        (root / "ready-0.json").write_text(
            json.dumps(dict(generation=2, world=1, version=1, scheduler={}))
        )
        engine = SimpleNamespace(
            world_size=1,
            is_offload=True,
            model=model,
            optimizer=optimizer,
            set_version=lambda v: None,
        )
        stage_model(engine, str(root), 1, 2)
        torch.testing.assert_close(weight, values["weight/weight"])
        if initialized_optimizer:
            torch.testing.assert_close(
                optimizer.state[weight]["exp_avg"], torch.ones_like(weight) * 7
            )
        else:
            assert not optimizer.state
        with pytest.raises(RuntimeError, match="generation mismatch"):
            stage_model(engine, str(root), 1, 3)
