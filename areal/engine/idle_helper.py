# SPDX-License-Identifier: Apache-2.0

"""Model-only refresh of an independent backward-only helper's parked storage."""

import json
from pathlib import Path

import torch

from areal.engine.idle_state import inventory, parked_storage_view, tensor_slice
from areal.utils import request_trace


def stage_model(engine, source_path, version, generation):
    if engine.world_size != 1 or not engine.is_offload:
        raise RuntimeError("Helper refresh requires parked DP1")
    root = Path(source_path).resolve()
    if not root.is_relative_to(Path("/dev/shm")):
        raise ValueError("Helper source must be shared DRAM")
    layout = json.loads((root / "layout.json").read_text())
    meta = json.loads((root / "ready-0.json").read_text())
    if meta["generation"] != generation:
        raise RuntimeError("Helper snapshot generation mismatch")
    for rank in range(meta["world"]):
        peer = json.loads((root / f"ready-{rank}.json").read_text())
        if any(peer[k] != meta[k] for k in ("version", "generation", "scheduler")):
            raise RuntimeError("Mixed helper model generation")
    data = torch.from_file(
        str(root / "state.bin"), shared=True, size=layout["bytes"], dtype=torch.uint8
    )
    values, _ = inventory(engine)
    copies = []
    for name, value in values.items():
        if not name.startswith(("weight/", "buffer/")):
            continue
        local, shape, _, _ = tensor_slice(value, 0, 1)
        entry = layout["entries"][name]
        if list(shape) != entry["shape"] or str(local.dtype) != entry["dtype"]:
            raise RuntimeError("Helper model shape or dtype mismatch")
        src = (
            data.narrow(0, entry["offset"], entry["size"])
            .view(local.dtype)
            .reshape(shape)
        )
        dst = (
            local
            if local.device.type == "cpu"
            else parked_storage_view(
                engine._adaptive_memory,
                local.data_ptr(),
                local.numel() * local.element_size(),
                local.dtype,
                local.shape,
            )
        )
        copies.append((dst, src))
    try:
        with torch.no_grad():
            for dst, src in copies:
                dst.copy_(src)
    except BaseException:
        engine._adaptive_memory.broken = True
        raise
    engine.set_version(version)
    request_trace.emit(
        "idle_helper_model_staged",
        generation=generation,
        version=version,
        bytes=sum(d.numel() * d.element_size() for d, _ in copies),
    )
    return dict(generation=generation, version=version)
