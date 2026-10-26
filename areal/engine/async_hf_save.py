# SPDX-License-Identifier: Apache-2.0

"""Bounded asynchronous HF serialization of independently owned CPU tensors."""

import base64
import copy
import json
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import torch

from areal.utils import request_trace


@contextmanager
def save_timing(stage, **fields):
    """Record CPU wall time without synchronizing accelerator streams."""
    started = time.monotonic()
    request_trace.emit("hf_save_detail_begin", stage=stage, **fields)
    try:
        yield
    finally:
        request_trace.emit(
            "hf_save_detail_end",
            stage=stage,
            seconds=time.monotonic() - started,
            **fields,
        )


def capture_save_assets(tokenizer=None, processor=None):
    """Snapshot exact save_pretrained files without compressing or rebuilding objects.

    Capture on every save so caller-side tokenizer/processor mutations are honored.
    Plain strings cross the RPC codec without expensive tokenizer reconstruction.
    """
    with save_timing("capture_assets"):
        with tempfile.TemporaryDirectory() as directory:
            if tokenizer is not None:
                tokenizer.save_pretrained(directory)
            if processor is not None:
                processor.save_pretrained(directory)
            root = Path(directory)
            assets = {}
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    name = str(path.relative_to(root))
                    validate_asset_path(name)
                    assets[name] = base64.b64encode(path.read_bytes()).decode("ascii")
            return assets


def validate_asset_path(name):
    """Reject traversal and files reserved for the model checkpoint."""
    relative = Path(name)
    normalized = relative.as_posix()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or normalized == "."
        or normalized
        in {"config.json", "generation_config.json", "areal_save_complete.json"}
        or normalized.endswith((".safetensors", ".bin", ".safetensors.index.json"))
    ):
        raise ValueError(f"Unsafe HF auxiliary artifact path: {name}")
    return relative


def write_save_assets(directory, assets):
    """Write an immutable artifact snapshot; never allow it to replace weights."""
    root = Path(directory)
    for name, content in assets.items():
        target = root / validate_asset_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(content, validate=True))


class AsyncHfSave:
    """One pending write; never read live model tensors from the writer thread."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-save")
        self.future = None

    def submit(
        self,
        path,
        state,
        config,
        generation_config=None,
        tokenizer=None,
        processor=None,
        metadata=None,
        verify=False,
        assets=None,
    ):
        self.wait()
        if not state or any(v.device.type != "cpu" for v in state.values()):
            raise ValueError("Async HF save requires a nonempty independent CPU state")
        with save_timing("metadata_copy", path=str(path)):
            config = copy.deepcopy(config)
            generation_config = copy.deepcopy(generation_config)
            tokenizer = copy.deepcopy(tokenizer)
            processor = copy.deepcopy(processor)
        with save_timing("writer_submit", path=str(path)):
            self.future = self.executor.submit(
                self._write,
                path,
                state,
                config,
                generation_config,
                tokenizer,
                processor,
                dict(metadata or {}),
                verify,
                dict(assets or {}),
            )

    @staticmethod
    def _write(
        path,
        state,
        config,
        generation_config,
        tokenizer,
        processor,
        metadata,
        verify,
        assets,
    ):
        from huggingface_hub import save_torch_state_dict

        started = time.monotonic()
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        marker = root / "areal_save_complete.json"
        if any(root.iterdir()):
            raise FileExistsError(f"Async checkpoint destination must be empty: {root}")
        staging = root / (".areal-pending-" + uuid.uuid4().hex)
        staging.mkdir()
        request_trace.emit("hf_save_write_begin", path=str(root), **metadata)
        try:
            save_torch_state_dict(state, staging, safe_serialization=True)
            config.save_pretrained(staging)
            if generation_config is not None:
                generation_config.save_pretrained(staging)
            if tokenizer is not None:
                tokenizer.save_pretrained(staging)
            if processor is not None:
                processor.save_pretrained(staging)
            with save_timing("write_auxiliary_assets", path=str(root)):
                write_save_assets(staging, assets)
            if verify:
                from safetensors.torch import load_file

                restored = {}
                for part in staging.glob("*.safetensors"):
                    restored.update(load_file(str(part)))
                if restored.keys() != state.keys():
                    raise RuntimeError("Saved HF tensor names differ from staged state")
                for name, value in state.items():
                    if not torch.equal(value, restored[name]):
                        raise RuntimeError(f"Saved HF tensor differs: {name}")
                request_trace.emit("hf_save_verified", tensors=len(state), **metadata)
            for item in staging.iterdir():
                os.replace(item, root / item.name)
            temporary = root / "areal_save_complete.json.tmp"
            temporary.write_text(json.dumps(dict(metadata, tensors=len(state))) + "\n")
            os.replace(temporary, marker)
            request_trace.emit(
                "hf_save_write_end",
                path=str(root),
                seconds=time.monotonic() - started,
                **metadata,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def wait(self):
        if self.future is not None:
            self.future.result()
            self.future = None

    def close(self):
        try:
            self.wait()
        finally:
            self.executor.shutdown(wait=True)


def independent_cpu_state(state, dtype):
    """Break aliases to both live tensors and reusable DRAM generations."""
    return {
        name: value.detach()
        .to(
            device="cpu",
            dtype=dtype if value.is_floating_point() else value.dtype,
            copy=True,
        )
        .contiguous()
        for name, value in state.items()
    }


def dram_model_state(engine):
    """Resolve persistent state keys, including tied aliases, from a full CPU map."""
    from areal.engine.idle_state import model_snapshot_parameters

    dram = engine._idle_dram
    values = dict(model_snapshot_parameters(dram.data, dram.layout))
    for name, entry in dram.layout["entries"].items():
        if name.startswith("buffer/"):
            dtype = getattr(torch, entry["dtype"].removeprefix("torch."))
            values[name[7:]] = (
                dram.data.narrow(0, entry["offset"], entry["size"])
                .view(dtype)
                .reshape(entry["shape"])
            )
    aliases = {}
    canonical = {}
    for name, value in list(
        engine.model.named_parameters(remove_duplicate=False)
    ) + list(engine.model.named_buffers(remove_duplicate=False)):
        canonical.setdefault(id(value), name)
        aliases[name] = canonical[id(value)]
    result = {}
    for name in engine.model.state_dict():
        source = name if name in values else aliases.get(name)
        if source not in values:
            raise RuntimeError(f"DRAM checkpoint is missing persistent tensor: {name}")
        result[name] = values[source]
    return result
