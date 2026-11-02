# SPDX-License-Identifier: Apache-2.0
"""Canonical parameter-relative state for pure-DP Megatron handoffs.

These operations deliberately do not use optimizer buffer offsets: bucket
padding and ownership change with DP size. Source bytes must remain immutable
until the complete transfer has been acknowledged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShardRange:
    start: int
    end: int
    total: int

    def validate(self):
        if not 0 <= self.start <= self.end <= self.total:
            raise ValueError("Invalid parameter-relative shard range")
        return self


def validate_partition(ranges: list[ShardRange], total: int):
    """Reject holes, overlaps, and different global tensor sizes."""
    cursor = 0
    for shard in sorted(ranges, key=lambda value: (value.start, value.end)):
        shard.validate()
        if shard.total != total:
            raise ValueError("Global parameter size changed")
        if shard.start == shard.end:
            continue
        if shard.start != cursor:
            raise ValueError("Optimizer shards overlap or leave a gap")
        cursor = shard.end
    if cursor != total:
        raise ValueError("Optimizer shards do not cover the parameter")


def intersections(source: list[ShardRange], target: ShardRange):
    """Return (source index, source offset, target offset, length)."""
    target.validate()
    validate_partition(source, target.total)
    copies = []
    for index, shard in enumerate(source):
        begin, end = max(shard.start, target.start), min(shard.end, target.end)
        if begin < end:
            copies.append(
                (index, begin - shard.start, begin - target.start, end - begin)
            )
    if sum(item[3] for item in copies) != target.end - target.start:
        raise ValueError("Destination shard is not fully covered")
    return copies


def optimizer_shards(engine):
    """Expose genuine master weights and Adam moments, without copying them."""
    from areal.engine.megatron_utils.shared_memory import optimizer_leaves

    names = {}
    for chunk, model in enumerate(engine.model):
        for name, parameter in model.named_parameters():
            key = f"{chunk}/{name}"
            if id(parameter) in names:
                raise ValueError(
                    "Aliased model parameters need an explicit canonical map"
                )
            names[id(parameter)] = key
    result = {}
    for index, optimizer in enumerate(optimizer_leaves(engine.optimizer)):
        required = (
            "model_param_gbuf_map",
            "_get_model_param_range_map",
            "_get_main_param_and_optimizer_states",
        )
        if not all(hasattr(optimizer, name) for name in required):
            raise ValueError(
                "Idle handoff requires a Megatron distributed Adam optimizer"
            )
        for parameter in optimizer.model_param_gbuf_map:
            key = names[id(parameter)]
            interval = optimizer._get_model_param_range_map(parameter)["param"]
            shard = ShardRange(
                interval.start, interval.end, parameter.numel()
            ).validate()
            tensors = optimizer._get_main_param_and_optimizer_states(parameter)
            if not {"param", "exp_avg", "exp_avg_sq"} <= tensors.keys():
                raise ValueError("Adam states must be initialized before handoff")
            if key in result:
                raise ValueError("Parameter belongs to multiple optimizer leaves")
            for field in ("param", "exp_avg", "exp_avg_sq"):
                tensor = tensors[field]
                if tensor.numel() != shard.end - shard.start:
                    raise ValueError(f"Optimizer shard shape mismatch: {key}/{field}")
                if str(tensor.dtype) != "torch.float32":
                    raise ValueError(
                        "Initial idle handoff requires FP32 masters and moments"
                    )
            result[key] = dict(optimizer=index, shard=shard, tensors=tensors)
    return result


def optimizer_metadata(optimizer):
    """Match MCore TE metadata without touching parked moment tensors."""
    import copy

    if optimizer.grad_scaler is not None:
        raise ValueError("Initial Megatron handoff requires BF16 without a grad scaler")
    groups = optimizer.optimizer.param_groups
    steps = {
        int(group["step"]) for group in groups if group["params"] and "step" in group
    }
    if len(steps) > 1:
        raise RuntimeError("Adam parameter groups have different steps")
    result = [
        copy.deepcopy({k: v for k, v in group.items() if k != "params"})
        for group in groups
    ]
    if steps:
        step = next(iter(steps))
        for group in result:
            group["step"] = step
    return {"optimizer": {"param_groups": result}}


def restore_optimizer_metadata(optimizer, metadata, parked_engine=None):
    """Restore counters in place without native checkpoint moment clones."""
    import copy

    if optimizer.grad_scaler is not None:
        raise ValueError("Handoff requires no grad scaler")
    inner = optimizer.optimizer
    for name in (
        "_state_quantizers",
        "_fp8_adam_metadata_cache",
        "_remainder_metadata_cache",
    ):
        if getattr(inner, name, None):
            raise ValueError(f"Unsupported optimizer cache: {name}")
    keys = ("wd_mult", "lr_mult", "is_expert_parallel", "is_decoupled_lr")

    def identity(group):
        return tuple(
            group[key] if key in group else group["pre_" + key] for key in keys
        )

    sources = metadata["optimizer"]["param_groups"]
    targets = inner.param_groups
    source_map = {identity(group): group for group in sources}
    target_ids = [identity(group) for group in targets]
    if (
        len(source_map) != len(sources)
        or len(set(target_ids)) != len(targets)
        or set(target_ids) != source_map.keys()
    ):
        raise ValueError("Optimizer parameter-group identities differ")
    updates = []
    for group in targets:
        update = copy.deepcopy(source_map[identity(group)])
        if "params" in update or type(update.get("step")) is not int:
            raise ValueError("Expected scalar Adam step metadata without parameters")
        updates.append(update)
    for group, update in zip(targets, updates, strict=True):
        params = group["params"]
        group.clear()
        group.update(update)
        group["params"] = params
        if params:
            mirror = (
                _prepared_step_mirror(inner, group, parked_engine)
                if parked_engine is not None
                else inner._get_group_step_tensor(group)
            )
            mirror.fill_(group["step"])


def save_transport(payload, path):
    """Serialize metadata and optionally stream leased CPU bytes without clones."""
    import os
    import pickle
    import sys
    import time
    import uuid
    from pathlib import Path

    import torch

    started = time.monotonic()
    raw = os.getenv("AREAL_IDLE_RAW_TRANSPORT", "0") == "1"
    total = 0
    timings = {}
    if raw:
        data_path = Path(path).parent / ("tensor-" + uuid.uuid4().hex + ".bin")
        temporary = data_path.with_suffix(".partial")
        try:
            tensors = []

            def encode(value):
                nonlocal total
                if isinstance(value, torch.Tensor):
                    if value.device.type != "cpu" or not value.is_contiguous():
                        raise ValueError(
                            "Raw transport requires contiguous CPU tensors"
                        )
                    total += (-total) % 64
                    offset = total
                    size = value.numel() * value.element_size()
                    total += size
                    tensors.append((offset, value))
                    return {
                        "__idle_tensor__": (
                            offset,
                            size,
                            str(value.dtype),
                            tuple(value.shape),
                        )
                    }
                if isinstance(value, dict):
                    return {k: encode(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [encode(v) for v in value]
                if isinstance(value, tuple):
                    return tuple(encode(v) for v in value)
                return value

            metadata = encode(payload)
            timings["encode_seconds"] = time.monotonic() - started
            from concurrent.futures import ThreadPoolExecutor

            # Disjoint mmap ranges avoid the shared inode write lock and extra clones.
            # The file is private until every writer succeeds and it is renamed.
            allocate_started = time.monotonic()
            recycled = False
            capacity = total
            pool = Path(path).parent.parent / "recycled-raw"
            # Only the coordinator releases fully acknowledged, unused generations.
            if pool.is_dir():
                candidates = []
                for candidate in pool.glob("tensor-*.bin"):
                    try:
                        candidates.append((candidate.stat().st_size, candidate))
                    except FileNotFoundError:
                        continue
                for candidate_size, candidate in sorted(candidates):
                    try:
                        if candidate_size >= total:
                            candidate.rename(temporary)
                            capacity = candidate_size
                            recycled = True
                            break
                    except FileNotFoundError:
                        continue  # Another rank claimed this allocation first.
            with temporary.open("r+b" if recycled else "x+b", buffering=0) as stream:
                if not recycled:
                    if total:
                        os.posix_fallocate(stream.fileno(), 0, total)
                    else:
                        stream.truncate(0)
                timings["recycled"] = recycled
                timings["allocate_seconds"] = time.monotonic() - allocate_started
                raw_copy_started = time.monotonic()
                import mmap

                import numpy as np

                mapping = (
                    mmap.mmap(stream.fileno(), total, access=mmap.ACCESS_WRITE)
                    if total
                    else None
                )
                try:

                    def write_tensor(item):
                        offset, value = item
                        start = 0
                        size = value.numel() * value.element_size()
                        if size:
                            target = np.frombuffer(
                                mapping,
                                dtype=np.uint8,
                                count=size,
                                offset=offset + start,
                            )
                            try:
                                source = (
                                    value.detach().reshape(-1).view(torch.uint8).numpy()
                                )
                                np.copyto(target, source[start : start + size])
                            finally:
                                del target

                    workers = int(os.getenv("AREAL_IDLE_RAW_WRITERS", "4"))
                    if not 1 <= workers <= 8:
                        raise ValueError("Raw state writers must be between 1 and 8")
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        for _ in executor.map(write_tensor, tensors):
                            pass
                finally:
                    if mapping is not None:
                        mapping.close()
            timings["raw_copy_seconds"] = time.monotonic() - raw_copy_started
            temporary.replace(data_path)
            payload = dict(
                raw_transport_version=1,
                data_file=data_path.name,
                data_bytes=total,
                storage_bytes=capacity,
                byteorder=sys.byteorder,
                payload=metadata,
            )
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    metadata_started = time.monotonic()
    writer = torch._C.PyTorchFileWriter(str(path), False, 64)
    torch.serialization._save(payload, writer, pickle, 2, False)
    writer.write_end_of_file()
    timings["metadata_save_seconds"] = time.monotonic() - metadata_started
    return dict(
        format="raw" if raw else "torch",
        bytes=total,
        **timings,
        write_seconds=time.monotonic() - started,
    )


def load_transport(path):
    """Read only published metadata; mmap owns its storage through tensor views."""
    import math
    import sys
    from pathlib import Path

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or "raw_transport_version" not in payload:
        return payload
    if payload["raw_transport_version"] != 1:
        raise ValueError("Unknown raw transport version")
    if payload.get("byteorder") != sys.byteorder:
        raise ValueError("Raw transport byte order mismatch")
    name = payload["data_file"]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Raw state file must be a sibling")
    data_path = Path(path).parent / name
    size = payload["data_bytes"]
    capacity = payload.get("storage_bytes", size)
    if (
        type(size) is not int
        or type(capacity) is not int
        or size < 0
        or capacity < size
        or data_path.stat().st_size != capacity
    ):
        raise ValueError("Truncated raw state transport")
    storage = torch.from_file(
        str(data_path), shared=False, size=size, dtype=torch.uint8
    )

    def decode(value):
        if isinstance(value, dict) and set(value) == {"__idle_tensor__"}:
            offset, length, dtype_name, shape = value["__idle_tensor__"]
            dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError("Invalid raw tensor dtype")
            element = torch.empty(0, dtype=dtype).element_size()
            if (
                type(offset) is not int
                or type(length) is not int
                or offset < 0
                or length < 0
                or offset + length > size
                or offset % element
                or any(type(d) is not int or d < 0 for d in shape)
                or math.prod(shape) * element != length
            ):
                raise ValueError("Invalid raw tensor extent")
            return storage[offset : offset + length].view(dtype).reshape(shape)
        if isinstance(value, dict):
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        if isinstance(value, tuple):
            return tuple(decode(v) for v in value)
        return value

    result = decode(payload["payload"])
    if result.get("async_export"):
        import json

        ready = json.loads(
            (Path(path).parent / f"ready-{result['rank']}.json").read_text()
        )
        if "error" in ready or any(
            ready.get(k) != result[k] for k in ("rank", "world", "version")
        ):
            raise RuntimeError("Uncommitted or mixed raw generation")
    return result


def restore_scheduler(scheduler, state):
    """MCore checkpoint load increments num_steps; handoff must replace it."""
    if not hasattr(scheduler, "num_steps") or "num_steps" not in state:
        raise ValueError("Unsupported Megatron scheduler progress format")
    scheduler.num_steps = 0
    scheduler.load_state_dict(state)
    if not _metadata_equal(scheduler.state_dict(), state):
        raise RuntimeError("Scheduler did not restore exact training progress")


def _prepared_step_mirror(inner, group, engine):
    device = group["params"][0].device
    key = (id(group), device.type, device.index)
    mirror = inner._step_tensors.get(key)
    if mirror is None:
        raise RuntimeError("CPU staging requires a preallocated Adam step mirror")
    return _parked_tensor(engine, mirror)


def _parked_tensor(engine, tensor):
    if not tensor.is_contiguous():
        raise ValueError("Parked state requires contiguous storage")
    if tensor.device.type == "cpu":
        return tensor
    from areal.engine.idle_state import parked_storage_view

    return parked_storage_view(
        engine._adaptive_memory,
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        tensor.dtype,
        tensor.shape,
    )


def _cpu_tensor(engine, tensor):
    """Copy live storage or read its preserved CPU backing without waking it."""
    from areal.engine.idle_state import parked_storage_view

    if not tensor.is_contiguous():
        raise ValueError("Megatron state transport requires contiguous tensor storage")
    if tensor.device.type == "cpu":
        return tensor.detach().clone()
    if engine.is_offload:
        value = parked_storage_view(
            engine._adaptive_memory,
            tensor.data_ptr(),
            tensor.numel() * tensor.element_size(),
            tensor.dtype,
            tensor.shape,
        )
        return value.clone()
    return tensor.detach().cpu()


def _model_tensors(engine):
    for chunk, model in enumerate(engine.model):
        for name, value in model.named_parameters():
            yield f"{chunk}/parameter/{name}", value
        for name, value in model.named_buffers():
            yield f"{chunk}/buffer/{name}", value


def _metadata_equal(left, right):
    """Compare optimizer metadata recursively, including scalar Adam steps."""
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _metadata_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _metadata_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def state_operation(engine, action, path=None):
    """Synchronous reference transport; include all exposure in switch costs."""
    import copy
    import os
    import time
    from pathlib import Path

    import torch
    import torch.distributed as dist

    from areal.engine.megatron_utils.shared_memory import optimizer_leaves
    from areal.infra.platforms import current_platform

    started = time.monotonic()
    if action == "save_wait":
        pending = getattr(engine, "_idle_export_future", None)
        return pending.result() if pending is not None else {"joined": True}
    engine.wait_shared_snapshot()
    leaves = list(optimizer_leaves(engine.optimizer))
    if action == "prepare":
        if any(
            getattr(engine.tf_config, name, 0.0) != 0.0
            for name in ("hidden_dropout", "attention_dropout")
        ):
            raise ValueError("Initial DP handoff validation requires dropout disabled")
        if getattr(engine.mcore_config, "swap_optimizer", False):
            raise ValueError(
                "Megatron idle handoff does not support optimizer swapping"
            )
        if engine.is_offload:
            raise RuntimeError("Initialize Adam states before parking the engine")
        with engine._shared_scope("actor_optimizer"):
            for leaf in leaves:
                inner = leaf.optimizer
                if (
                    type(inner).__name__ != "FusedAdam"
                    or not type(inner).__module__.startswith("transformer_engine.")
                    or not callable(getattr(inner, "initialize_state", None))
                ):
                    raise ValueError(
                        "Initial Megatron handoff requires TE FusedAdam initialize_state"
                    )
                for group in inner.param_groups:
                    for parameter in group["params"]:
                        inner.initialize_state(parameter)
                    if group["params"]:
                        inner._get_group_step_tensor(group)
        optimizer_shards(engine)
        for leaf in leaves:
            if not _metadata_equal(optimizer_metadata(leaf), leaf.state_dict()):
                raise RuntimeError(
                    "Optimizer metadata adapter differs from native Megatron"
                )
        return {"prepared": True}
    if action not in ("save", "save_start", "load", "stage", "verify"):
        raise ValueError("Unknown Megatron idle state operation")
    root = Path(path)
    if not root.resolve().is_relative_to(Path("/dev/shm")):
        raise ValueError("Megatron handoff storage must be RAM-backed")
    rank, world = dist.get_rank(), dist.get_world_size()
    shards = optimizer_shards(engine)
    if action == "save_start":
        # Lease the parked CPU allocations, never live HBM. Restore only reads
        # them; subsequent dirty/offload creates new backups. Strong views keep
        # this generation alive independently of allocator cache ownership.
        if not engine.is_offload:
            raise RuntimeError("Asynchronous export requires parked CPU backing")
        import json
        from concurrent.futures import Future
        from threading import Thread

        pending = getattr(engine, "_idle_export_future", None)
        if pending is not None:
            if not pending.done():
                raise RuntimeError("Previous CPU exporter is still using its lease")
            pending.result()
        root.mkdir(parents=True, exist_ok=False) if rank == 0 else None
        dist.barrier(group=engine.cpu_group)

        def lease(value):
            # Native CPU buffers may mutate in place; only allocator backing is leased.
            view = _parked_tensor(engine, value)
            return view.clone() if value.device.type == "cpu" else view

        payload = {
            "world": world,
            "rank": rank,
            "version": engine.get_version(),
            "scheduler": copy.deepcopy(engine.lr_scheduler.state_dict()),
            "optimizers": [optimizer_metadata(o) for o in leaves],
            "model": {k: lease(v) for k, v in _model_tensors(engine)}
            if rank == 0
            else {},
            "shards": {},
        }
        for key, item in shards.items():
            r = item["shard"]
            payload["shards"][key] = dict(
                range=(r.start, r.end, r.total),
                optimizer=item["optimizer"],
                tensors={k: lease(v) for k, v in item["tensors"].items()},
            )
        payload["async_export"] = True
        version = payload["version"]
        future = engine._idle_export_future = Future()

        def write():
            try:
                # Compact each narrow allocation view in the background. No
                # process groups, engine state or NPU operations in this thread.
                clone_started = time.monotonic()
                if os.getenv("AREAL_IDLE_RAW_TRANSPORT", "0") != "1":
                    payload["model"] = {
                        k: v.clone() for k, v in payload["model"].items()
                    }
                    for item in payload["shards"].values():
                        item["tensors"] = {
                            k: v.clone() for k, v in item["tensors"].items()
                        }
                clone_seconds = time.monotonic() - clone_started
                temporary = root / f"rank-{rank}.tmp"
                transport = save_transport(payload, temporary)
                temporary.replace(root / f"rank-{rank}.pt")
                result = dict(
                    version=version,
                    rank=rank,
                    world=world,
                    seconds=time.monotonic() - started,
                    clone_seconds=clone_seconds,
                    transport=transport,
                )
                ready = root / f"ready-{rank}.tmp"
                ready.write_text(json.dumps(result))
                ready.replace(root / f"ready-{rank}.json")
                future.set_result(result)
            except BaseException as exc:
                try:
                    ready = root / f"ready-{rank}.tmp"
                    ready.write_text(
                        json.dumps(dict(error=repr(exc), rank=rank, version=version))
                    )
                    ready.replace(root / f"ready-{rank}.json")
                finally:
                    future.set_exception(exc)

        try:
            Thread(target=write, name="megatron-cpu-export", daemon=False).start()
        except BaseException as exc:
            future.set_exception(exc)
            raise
        return dict(version=version, world=world, captured=True)
    if action == "save":
        if not engine.is_offload:
            current_platform.synchronize()
        payload = {
            "world": world,
            "rank": rank,
            "version": engine.get_version(),
            "scheduler": copy.deepcopy(engine.lr_scheduler.state_dict()),
            "optimizers": [optimizer_metadata(o) for o in leaves],
            "shards": {},
            "model": {k: _cpu_tensor(engine, v) for k, v in _model_tensors(engine)}
            if rank == 0
            else {},
        }
        for key, item in shards.items():
            r = item["shard"]
            payload["shards"][key] = {
                "range": (r.start, r.end, r.total),
                "optimizer": item["optimizer"],
                "tensors": {
                    k: _cpu_tensor(engine, v) for k, v in item["tensors"].items()
                },
            }
        root.mkdir(parents=True, exist_ok=True)
        temporary = root / f"rank-{rank}.tmp"
        save_transport(payload, temporary)
        temporary.replace(root / f"rank-{rank}.pt")
        dist.barrier(group=engine.cpu_group)
        if rank == 0:
            (root / "world.txt").write_text(str(world))
        dist.barrier(group=engine.cpu_group)
        return {"seconds": time.monotonic() - started, "version": engine.get_version()}
    if action == "stage":
        if not engine.is_offload:
            raise RuntimeError("CPU staging requires parked destination")
        engine._idle_stage_valid = False
    elif engine.is_offload:
        raise RuntimeError("Restore destination allocations before import/verify")
    source_world = int((root / "world.txt").read_text())
    load_started = time.monotonic()
    sources = [load_transport(root / f"rank-{i}.pt") for i in range(source_world)]
    load_seconds = time.monotonic() - load_started
    head = sources[0]
    if any(
        s["world"] != source_world
        or s["version"] != head["version"]
        or not _metadata_equal(s["scheduler"], head["scheduler"])
        or not _metadata_equal(s["optimizers"], head["optimizers"])
        for s in sources
    ):
        raise RuntimeError("Source snapshot mixes training generations")
    # Resolve every target before any write; staging never touches mapped NPU bytes.
    staged = {}
    if action == "stage":
        for leaf in leaves:
            for group in leaf.optimizer.param_groups:
                if group["params"]:
                    _prepared_step_mirror(leaf.optimizer, group, engine)
        values = list(_model_tensors(engine))
        values += [
            (key + "/" + field, value)
            for key, item in shards.items()
            for field, value in item["tensors"].items()
        ]
        for name, value in values:
            staged[id(value)] = _parked_tensor(engine, value)
    writes = []

    def copy_or_check(destination, source, name):
        if destination.shape != source.shape or destination.dtype != source.dtype:
            raise RuntimeError(f"State shape/dtype mismatch: {name}")
        if action in ("load", "stage"):
            writes.append((destination, source))
        elif not torch.equal(
            destination.detach().cpu().reshape(-1).view(torch.uint8),
            source.contiguous().reshape(-1).view(torch.uint8),
        ):
            raise RuntimeError(f"State verification failed: {name}")

    with torch.no_grad():
        for key, tensor in _model_tensors(engine):
            copy_or_check(
                staged[id(tensor)] if action == "stage" else tensor,
                head["model"][key],
                key,
            )
        for key, item in shards.items():
            parts = [s["shards"][key] for s in sources if key in s["shards"]]
            ranges = [ShardRange(*s["range"]) for s in parts]
            copies = intersections(ranges, item["shard"])
            for field, target in item["tensors"].items():
                if action == "stage":
                    target = staged[id(target)]
                if field == "step" and target.numel() == 1:
                    reference = parts[0]["tensors"][field]
                    if any(
                        not torch.equal(reference, s["tensors"][field]) for s in parts
                    ):
                        raise RuntimeError("Adam step differs across source ranks")
                    copy_or_check(target, reference, key + "/" + field)
                    continue
                for i, src, dst, count in copies:
                    copy_or_check(
                        target.reshape(-1)[dst : dst + count],
                        parts[i]["tensors"][field].reshape(-1)[src : src + count],
                        key + "/" + field,
                    )
    if action in ("load", "stage"):
        copy_started = time.monotonic()
        copy_mode = "serial"
        with torch.no_grad():
            for destination, source in writes:
                destination.copy_(source)
        copy_seconds = time.monotonic() - copy_started
        from contextlib import nullcontext

        with (
            nullcontext()
            if action == "stage"
            else engine._shared_scope("actor_optimizer")
        ):
            for opt, metadata in zip(leaves, head["optimizers"], strict=True):
                restore_optimizer_metadata(
                    opt, metadata, parked_engine=engine if action == "stage" else None
                )
        restore_scheduler(engine.lr_scheduler, head["scheduler"])
        engine.set_version(head["version"])
    elif (
        engine.get_version() != head["version"]
        or engine.lr_scheduler.state_dict() != head["scheduler"]
    ):
        raise RuntimeError("Training progress changed during handoff")
    if action == "verify" and not _metadata_equal(
        [optimizer_metadata(o) for o in leaves], head["optimizers"]
    ):
        raise RuntimeError("Optimizer metadata changed during handoff")
    if action != "stage":
        current_platform.synchronize()
    record = {
        "rank": rank,
        "seconds": time.monotonic() - started,
        "version": engine.get_version(),
        "source_world": source_world,
        "target_world": world,
        "verified": action == "verify",
        "scheduler_steps": engine.lr_scheduler.num_steps,
        "load_seconds": load_seconds,
        "copy_seconds": copy_seconds if action in ("load", "stage") else 0.0,
        "copy_mode": copy_mode if action in ("load", "stage") else "verify",
    }
    records = [None] * world
    dist.all_gather_object(records, record, group=engine.cpu_group)
    if action == "stage":
        engine._idle_stage_valid = True
    return {**record, "ranks": records}
