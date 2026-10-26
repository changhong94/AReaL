# SPDX-License-Identifier: Apache-2.0

"""Preserve an accumulated DP1 gradient prefix across a RAM-only DP handoff."""

import json
import math
import os
import time
from pathlib import Path

import torch

from areal.engine.idle_state import manifest, tensor_slice
from areal.utils import request_trace


def export_prefix_gradients(engine, path, token):
    if engine.world_size != 1:
        raise ValueError("Gradient prefix export currently requires DP1")
    root = Path(path)
    if not root.resolve().is_relative_to(Path("/dev/shm")):
        raise ValueError("Gradient handoff must use shared DRAM")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    values = {
        n: p.grad for n, p in engine.model.named_parameters() if p.grad is not None
    }
    if os.getenv("AREAL_IDLE_FULL_VERIFY", "1") == "1" and hasattr(
        engine, "_idle_recorded_segments"
    ):
        replay_segments(engine, engine._idle_replay_segments, token)
        torch.save(engine._idle_recorded_segments, root / "packed-prefix.pt")
    values = {
        n: p.grad for n, p in engine.model.named_parameters() if p.grad is not None
    }
    layout = manifest(values)
    available = os.statvfs(root)
    if layout["bytes"] > available.f_bavail * available.f_frsize:
        raise RuntimeError("Insufficient DRAM for accumulated gradients")
    allocation_started = time.monotonic()
    fd = os.open(root / "gradients.bin", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if layout["bytes"]:
            os.posix_fallocate(fd, 0, layout["bytes"])
    finally:
        os.close(fd)
    data = torch.from_file(
        str(root / "gradients.bin"),
        shared=True,
        size=layout["bytes"],
        dtype=torch.uint8,
    )
    allocation_seconds = time.monotonic() - allocation_started
    copy_started = time.monotonic()
    for name, value in values.items():
        local, shape, _, _ = tensor_slice(value, 0, 1)
        entry = layout["entries"][name]
        target = (
            data.narrow(0, entry["offset"], entry["size"])
            .view(local.dtype)
            .reshape(shape)
        )
        target.copy_(local, non_blocking=False)
    meta = {"token": token, "version": engine.get_version(), "layout": layout}
    (root / "meta.json").write_text(json.dumps(meta))
    request_trace.emit(
        "idle_gradient_exported",
        token=token,
        bytes=layout["bytes"],
        parameters=len(values),
        allocation_seconds=allocation_seconds,
        copy_seconds=time.monotonic() - copy_started,
    )
    return {"bytes": layout["bytes"]}


@torch.no_grad()
def add_prefix_gradients(engine, path, token):
    """Add already normalized prefix AFTER suffix reduction, never average twice."""
    started = time.monotonic()
    root = Path(path)
    meta = json.loads((root / "meta.json").read_text())
    if meta["token"] != token or meta["version"] != engine.get_version():
        raise RuntimeError("Gradient prefix generation mismatch")
    if getattr(engine, "_idle_added_gradient_token", None) == token:
        raise RuntimeError("Gradient prefix has already been added")
    layout = meta["layout"]
    params = dict(engine.model.named_parameters())
    if not set(layout["entries"]).issubset(params):
        raise RuntimeError("Gradient prefix contains unknown parameters")
    data = torch.from_file(
        str(root / "gradients.bin"),
        shared=True,
        size=layout["bytes"],
        dtype=torch.uint8,
    )
    # Validate all shapes and dtypes before mutating any gradient.
    views = []
    for name, entry in layout["entries"].items():
        param = params[name]
        local, shape, offset, replicated = tensor_slice(
            param, engine.rank, engine.world_size
        )
        if list(shape) != entry["shape"]:
            raise RuntimeError("Gradient prefix parameter shape mismatch")
        dtype = getattr(torch, entry["dtype"].split(".")[-1])
        raw = data.narrow(0, entry["offset"], entry["size"]).view(dtype).reshape(shape)
        shard = raw if replicated else raw.narrow(0, offset, local.shape[0])
        if param.grad is not None:
            grad_local, grad_shape, grad_offset, grad_replicated = tensor_slice(
                param.grad, engine.rank, engine.world_size
            )
            if (
                grad_shape != shape
                or grad_offset != offset
                or grad_replicated != replicated
                or grad_local.shape != shard.shape
            ):
                raise RuntimeError("Gradient prefix shard layout mismatch")
        views.append((param, shard))
    for param, shard in views:
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        grad = getattr(param.grad, "_local_tensor", param.grad)
        grad.add_(shard.to(device=grad.device, dtype=grad.dtype))
    engine._idle_added_gradient_token = token
    request_trace.emit(
        "idle_gradient_prefix_added",
        seconds=time.monotonic() - started,
        rank=engine.rank,
        token=token,
        bytes=layout["bytes"],
        parameters=len(views),
    )


@torch.no_grad()
def verify_accumulated_gradients(engine, replay, token):
    """Replay the unsegmented batch at unchanged weights before the Adam step."""
    saved = {
        n: getattr(p.grad, "_local_tensor", p.grad).detach().cpu().clone()
        for n, p in engine.model.named_parameters()
        if p.grad is not None
    }
    engine.optimizer_zero_grad()
    with torch.enable_grad():
        replay()
    compare_gradients(engine, saved, "full_batch_reference", token)
    square_error = 0.0
    square_reference = 0.0
    max_error = 0.0
    actual = {n for n, p in engine.model.named_parameters() if p.grad is not None}
    if actual != set(saved):
        raise RuntimeError("Accumulated/reference gradient parameter coverage differs")
    for name, p in engine.model.named_parameters():
        if name not in saved:
            continue
        local = getattr(p.grad, "_local_tensor", p.grad)
        reference = local.detach().cpu().float()
        error = saved[name].float() - reference
        square_error += error.square().sum().item()
        square_reference += reference.square().sum().item()
        max_error = max(max_error, error.abs().max().item() if error.numel() else 0.0)
        local.copy_(saved[name])
    import torch.distributed as dist

    metrics = torch.tensor([square_error, square_reference], dtype=torch.float64)
    dist.all_reduce(metrics, group=engine.cpu_group)
    relative_l2 = (metrics[0] / metrics[1].clamp_min(1e-30)).sqrt().item()
    if (
        not math.isfinite(relative_l2)
        or not torch.isfinite(metrics).all().item()
        or relative_l2 > 0.02
    ):
        raise RuntimeError(
            f"Accumulated gradient differs from full replay: {relative_l2}"
        )
    request_trace.emit(
        "idle_gradient_replay_verified",
        rank=engine.rank,
        token=token,
        relative_l2=relative_l2,
        max_absolute_error=max_error,
    )


@torch.no_grad()
def gradient_snapshot(engine):
    return {
        n: getattr(p.grad, "_local_tensor", p.grad).detach().cpu().clone()
        for n, p in engine.model.named_parameters()
        if p.grad is not None
    }


@torch.no_grad()
def compare_gradients(engine, expected, label, token):
    import torch.distributed as dist

    rows = []
    for name, p in engine.model.named_parameters():
        if name not in expected:
            continue
        if p.grad is None:
            raise RuntimeError("Diagnostic gradient coverage mismatch")
        actual = getattr(p.grad, "_local_tensor", p.grad).detach().cpu().float()
        reference = expected[name].float()
        delta = actual - reference
        rows.append(
            dict(
                name=name,
                error=delta.square().sum().item(),
                reference=reference.square().sum().item(),
                actual=actual.square().sum().item(),
                dot=(actual * reference).sum().item(),
            )
        )
    totals = torch.tensor(
        [sum(r[k] for r in rows) for k in ("error", "reference", "actual", "dot")],
        dtype=torch.float64,
    )
    dist.all_reduce(totals, group=engine.cpu_group)
    e, r, a, d = totals.tolist()
    request_trace.emit(
        "idle_gradient_diagnostic",
        label=label,
        token=token,
        rank=engine.rank,
        relative_l2=math.sqrt(e / max(r, 1e-30)),
        norm_ratio=math.sqrt(a / max(r, 1e-30)),
        cosine=d / max(math.sqrt(a * r), 1e-30),
        best_fit_scale=d / max(r, 1e-30),
        largest_errors=sorted(rows, key=lambda x: x["error"], reverse=True)[:10],
    )


@torch.no_grad()
def replay_segments(engine, replay, token, prefix_path=None):
    """Diagnostic only: isolate identical packing and cross-DP prefix replay."""
    saved = gradient_snapshot(engine)
    engine.optimizer_zero_grad()
    with torch.enable_grad():
        replay(engine._idle_recorded_segments, engine.parallel_helper.dp_size)
    compare_gradients(
        engine,
        saved,
        "same_layout_suffix" if prefix_path else "same_layout_source",
        token,
    )
    if prefix_path:
        engine.optimizer_zero_grad()
        add_prefix_gradients(engine, prefix_path, token)
        expected_prefix = gradient_snapshot(engine)
        engine._idle_added_gradient_token = None
        engine.optimizer_zero_grad()
        chunks = torch.load(Path(prefix_path) / "packed-prefix.pt", weights_only=False)
        with torch.enable_grad():
            # Every DP rank repeats the DP1 packed prefix; FSDP averaging then
            # gives the DP1 objective without an extra DP multiplier.
            replay(chunks, 1)
        compare_gradients(engine, expected_prefix, "cross_dp_identical_prefix", token)
    restore_gradient_snapshot(engine, saved)


@torch.no_grad()
def restore_gradient_snapshot(engine, saved):
    """Restore exact gradient coverage, including unused parameters."""
    for name, p in engine.model.named_parameters():
        if name not in saved:
            p.grad = None
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        local = getattr(p.grad, "_local_tensor", p.grad)
        local.copy_(saved[name])
