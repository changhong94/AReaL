# SPDX-License-Identifier: Apache-2.0

"""Versioned, process-shared DRAM shards for preinitialized FSDP actors.

The mmap lives on tmpfs, never in the checkpoint directory. Device allocations
remain owned by the existing CaMem pools. Only contiguous one-dimensional DP
Shard(0) and replicated tensors are supported.
"""

import copy
import hashlib
import json
import math
import os
import time
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from threading import Thread

import torch
import torch.distributed as dist


def tensor_slice(value, rank, world):
    """Return the local storage and its global leading-axis interval."""
    local = getattr(value, "_local_tensor", value).detach()
    shape = tuple(value.shape)
    offset = 0
    replicated = True
    placements = getattr(value, "placements", None)
    if placements is not None:
        if len(placements) != 1:
            raise ValueError("Idle DRAM requires a one-dimensional DP mesh")
        placement = placements[0]
        if placement.is_shard():
            if placement.dim != 0:
                raise ValueError("Idle DRAM supports only Shard(0)")
            replicated = False
            offset = min(rank * math.ceil(shape[0] / world), shape[0])
        elif not placement.is_replicate():
            raise ValueError("Unsupported idle-state placement")
    if not local.is_contiguous():
        raise ValueError("Idle DRAM requires contiguous local tensors")
    return local, shape, offset, replicated


def inventory(engine):
    result = {}
    names = {}
    for name, value in engine.model.named_parameters():
        names[id(value)] = name
        result["weight/" + name] = value
        for key, state in engine.optimizer.state.get(value, {}).items():
            if not isinstance(state, torch.Tensor):
                raise ValueError("Non-tensor optimizer state is not supported")
            result["optimizer/" + name + "/" + key] = state
    # Include buffers, including replicated scalars and model-specific buffers.
    for name, value in engine.model.named_buffers():
        result["buffer/" + name] = value
    groups = []
    for group in engine.optimizer.param_groups:
        item = copy.deepcopy({k: v for k, v in group.items() if k != "params"})
        item["params"] = [names[id(p)] for p in group["params"]]
        groups.append(item)
    return result, groups


def manifest(values):
    entries = {}
    position = 0
    for name, value in sorted(values.items()):
        position = (position + 63) // 64 * 64
        size = value.numel() * value.element_size()
        entries[name] = dict(
            offset=position, shape=list(value.shape), dtype=str(value.dtype), size=size
        )
        position += size
    return dict(entries=entries, bytes=position)


def parked_storage_view(memory, address, size, dtype, shape):
    """Resolve a tensor byte interval into a preserved, unmapped pool backup."""
    if memory.broken:
        raise RuntimeError("Cannot stage into a broken residency pool")
    matches = [
        a
        for a in memory.inventory().values()
        if a.tag in ("actor_weights", "actor_optimizer")
        and a.handle[2] <= address
        and address + size <= a.handle[2] + a.handle[1]
    ]
    if len(matches) != 1:
        raise RuntimeError("Training tensor has no unique backed allocation")
    allocation = matches[0]
    missing = memory._missing.get(allocation.handle[2])
    backup = allocation.cpu_backup_tensor
    if missing is None or missing[0] is not allocation or missing[1] != "cpu":
        raise RuntimeError("Training allocation is not parked with preserved state")
    if (
        backup is None
        or backup.device.type != "cpu"
        or backup.dtype != torch.uint8
        or not backup.is_contiguous()
        or backup.numel() < allocation.handle[1]
    ):
        raise RuntimeError("Invalid parked training backup")
    return (
        backup.narrow(0, address - allocation.handle[2], size)
        .view(dtype)
        .reshape(shape)
    )


class DramState:
    def __init__(self, engine, path):
        self.engine = engine
        self.path = Path(path)
        if not self.path.resolve().is_relative_to(Path("/dev/shm")):
            raise ValueError("Idle state must use /dev/shm, not checkpoint storage")
        self.rank = engine.rank
        self.world = engine.world_size
        values, _ = inventory(engine)
        for value in values.values():
            tensor_slice(value, self.rank, self.world)
        self.layout = manifest(values)
        if self.rank == 0:
            self.path.mkdir(parents=True, exist_ok=False)
            available = os.statvfs(self.path)
            if self.layout["bytes"] > available.f_bavail * available.f_frsize:
                raise RuntimeError("Insufficient shared DRAM capacity")
            fd = os.open(self.path / "state.bin", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                # Reserve tmpfs capacity now instead of risking SIGBUS mid-copy.
                os.posix_fallocate(fd, 0, self.layout["bytes"])
            finally:
                os.close(fd)
            (self.path / "layout.json").write_text(json.dumps(self.layout))
        dist.barrier(group=engine.cpu_group)
        if json.loads((self.path / "layout.json").read_text()) != self.layout:
            raise RuntimeError("Training ranks disagree on the state layout")
        self.data = torch.from_file(
            str(self.path / "state.bin"),
            shared=True,
            size=self.layout["bytes"],
            dtype=torch.uint8,
        )
        self.future = None
        self.generation = 0
        self.source_maps = {}

    @staticmethod
    def view(data, entry, value, rank, world):
        local, shape, offset, replicated = tensor_slice(value, rank, world)
        raw = data.narrow(0, entry["offset"], entry["size"])
        whole = raw.view(local.dtype).reshape(shape)
        target = whole if replicated else whole.narrow(0, offset, local.shape[0])
        if target.shape != local.shape:
            raise RuntimeError("Local state shape does not match the shard manifest")
        return local, target, replicated

    def wait(self):
        return self.future.result() if self.future is not None else None

    def poll(self):
        """Inspect completion without occupying the engine RPC thread waiting."""
        if self.future is None:
            raise RuntimeError("No idle snapshot has been started")
        return {
            "generation": self.generation,
            "done": self.future.done(),
            "snapshot": self.future.result() if self.future.done() else None,
        }

    def start(self):
        self.wait()
        values, groups = inventory(self.engine)
        if manifest(values) != self.layout:
            raise RuntimeError("State layout changed after initialization")
        # Capture metadata and tensor views before launching; the engine's
        # mutation/unmapping fences keep borrowed device storage immutable.
        meta = dict(
            generation=self.generation + 1,
            version=self.engine.get_version(),
            scheduler=copy.deepcopy(self.engine.lr_scheduler.state_dict()),
            groups=groups,
            world=self.world,
        )
        views = [
            (
                name,
                *self.view(
                    self.data,
                    self.layout["entries"][name],
                    value,
                    self.rank,
                    self.world,
                ),
            )
            for name, value in values.items()
        ]
        from areal.infra.platforms import current_platform

        current_platform.synchronize()
        memory = getattr(self.engine, "_adaptive_memory", None)
        if memory is not None:
            memory.start_retained_backup()
        device = current_platform.current_device()
        self.generation += 1
        self.future = Future()
        device_api = getattr(
            torch, getattr(current_platform, "device_type", "cpu"), None
        )
        copy_stream = (
            device_api.Stream(device=device)
            if getattr(current_platform, "device_type", "cpu") == "npu"
            else None
        )

        def run():
            started = time.monotonic()
            try:
                current_platform.set_device(device)
                # No distributed collectives or FSDP materialization here.
                copied = 0
                # start() fenced producer work before creating this private stream.
                # Read-only publication may run concurrently on the engine stream.
                context = (
                    device_api.stream(copy_stream)
                    if copy_stream is not None
                    else nullcontext()
                )
                with context:
                    for _, local, target, replicated in views:
                        if replicated and self.rank != 0:
                            continue
                        target.copy_(local, non_blocking=False)
                        copied += local.numel() * local.element_size()
                # Do not wait for unrelated publication work on the default stream.
                if copy_stream is not None:
                    copy_stream.synchronize()
                else:
                    current_platform.synchronize()
                meta["copy_seconds"] = time.monotonic() - started
                meta["copied_bytes"] = copied
                temp = self.path / f"ready-{self.rank}.tmp"
                temp.write_text(json.dumps(meta))
                temp.replace(self.path / f"ready-{self.rank}.json")
                self.future.set_result(meta)
            except BaseException as exc:
                self.future.set_exception(exc)

        Thread(target=run, name="idle-state-dram-copy").start()
        return {
            "started": True,
            "version": meta["version"],
            "generation": self.generation,
        }

    def stage(self, source_path, version):
        if getattr(self, "pending_import", None) is not None:
            raise RuntimeError("Previous staged import has not completed")
        self.wait()
        source = Path(source_path)
        if not source.resolve().is_relative_to(Path("/dev/shm")):
            raise ValueError("Idle state must be RAM-backed")
        layout = json.loads((source / "layout.json").read_text())
        if layout != self.layout:
            raise RuntimeError("Source and destination tensor manifests differ")
        meta = json.loads((source / "ready-0.json").read_text())
        for rank in range(meta["world"]):
            other = json.loads((source / f"ready-{rank}.json").read_text())
            for key in ("version", "scheduler", "groups", "generation"):
                if other[key] != meta[key]:
                    raise RuntimeError("Source ranks published mixed state generations")
        cache_key = str(source.resolve())
        if cache_key not in self.source_maps:
            self.source_maps[cache_key] = torch.from_file(
                str(source / "state.bin"),
                shared=True,
                size=layout["bytes"],
                dtype=torch.uint8,
            )
        data = self.source_maps[cache_key]
        values, groups = inventory(self.engine)
        if [g["params"] for g in groups] != [g["params"] for g in meta["groups"]]:
            raise RuntimeError("Optimizer parameter ordering changed")
        # Validate every target before mutating any backup or CPU optimizer state.
        memory = getattr(self.engine, "_adaptive_memory", None)
        if memory is not None and not self.engine.is_offload:
            raise RuntimeError("Destination training engine must be offloaded")
        copies = []
        for name, value in values.items():
            local, target, _ = self.view(
                data, layout["entries"][name], value, self.rank, self.world
            )
            if local.numel() == 0:
                continue
            if local.device.type == "cpu":
                destination = local
            elif memory is not None:
                destination = parked_storage_view(
                    memory,
                    local.data_ptr(),
                    local.numel() * local.element_size(),
                    local.dtype,
                    local.shape,
                )
            else:
                raise RuntimeError(
                    "Device training state requires a parked adaptive pool"
                )
            copies.append((destination, target))
        started = time.monotonic()
        try:
            with torch.no_grad():
                for destination, target in copies:
                    destination.copy_(target)
        except BaseException:
            if memory is not None:
                memory.broken = True
            raise
        self.pending_import = (data, layout, version)
        copied = sum(dst.numel() * dst.element_size() for dst, _ in copies)
        elapsed = time.monotonic() - started
        for group, saved in zip(
            self.engine.optimizer.param_groups, meta["groups"], strict=True
        ):
            group.update({k: v for k, v in saved.items() if k != "params"})
        self.engine.lr_scheduler.load_state_dict(meta["scheduler"])
        self.engine.set_version(version)
        return dict(staged_bytes=copied, stage_seconds=elapsed, version=version)

    def verify(self, full=True):
        if getattr(self.engine, "is_offload", False):
            raise RuntimeError("Restore staged training state before verification")
        if getattr(self, "pending_import", None) is None:
            raise RuntimeError("No staged training import to verify")
        data, layout, version = self.pending_import
        timings = dict(verify_d2h_seconds=0.0, equality_seconds=0.0, hash_seconds=0.0)
        checked = 0
        digest = hashlib.sha256()
        if full:
            values, _ = inventory(self.engine)
            for name, value in values.items():
                local, target, _ = self.view(
                    data, layout["entries"][name], value, self.rank, self.world
                )
                started = time.monotonic()
                actual = local.cpu()
                timings["verify_d2h_seconds"] += time.monotonic() - started
                started = time.monotonic()
                if not torch.equal(
                    actual.reshape(-1).view(torch.uint8),
                    target.reshape(-1).view(torch.uint8),
                ):
                    raise RuntimeError(f"Imported training state differs: {name}")
                timings["equality_seconds"] += time.monotonic() - started
                started = time.monotonic()
                digest.update(
                    memoryview(
                        actual.contiguous().reshape(-1).view(torch.uint8).numpy()
                    )
                )
                timings["hash_seconds"] += time.monotonic() - started
                checked += actual.numel() * actual.element_size()
        self.pending_import = None
        return dict(
            **timings,
            full_verification=full,
            verified_bytes=checked,
            shard_sha256=digest.hexdigest() if full else None,
            version=version,
            scheduler_epoch=self.engine.lr_scheduler.last_epoch,
        )

    def load(self, source_path, version):
        """CPU test convenience; device engines must stage before onload."""
        self.stage(source_path, version)
        return self.verify()


def model_snapshot_parameters(data, layout):
    """Expose full model parameter views from an immutable gathered CPU layout."""
    parameters = []
    for name, entry in layout["entries"].items():
        if not name.startswith("weight/"):
            continue
        dtype = getattr(torch, entry["dtype"].removeprefix("torch."))
        value = (
            data.narrow(0, entry["offset"], entry["size"])
            .view(dtype)
            .reshape(entry["shape"])
        )
        parameters.append((name.removeprefix("weight/"), value))
    if not parameters:
        raise RuntimeError("DRAM snapshot contains no model parameters")
    return parameters
