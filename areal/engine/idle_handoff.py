# SPDX-License-Identifier: Apache-2.0

"""Rank-local preparation and release gates for local adaptive FSDP handoffs."""

import hashlib
import json
import os
import time
from concurrent.futures import Future
from pathlib import Path
from threading import Thread

from areal.utils import request_trace


def device_key(engine):
    """Identify a device in the common visibility namespace, not its DP rank."""
    mask = os.getenv(
        "ASCEND_RT_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "all")
    )
    return hashlib.sha256(f"{mask}:{engine.device.index}".encode()).hexdigest()[:20]


def stage_async(engine, source, version, generation):
    previous = getattr(engine, "_idle_stage_future", None)
    if previous is not None:
        raise RuntimeError("Previous standby preparation has not been consumed")
    future = engine._idle_stage_future = Future()

    def prepare():
        started = time.monotonic()
        try:
            metadata = json.loads((Path(source) / "ready-0.json").read_text())
            if metadata["generation"] != generation:
                raise RuntimeError("Standby snapshot generation changed before staging")
            result = engine._idle_dram.stage(source, version)
            result["snapshot_generation"] = generation
            request_trace.emit(
                "idle_dram_backup_staged",
                rank=engine.rank,
                preparation_seconds=time.monotonic() - started,
                **result,
            )
            future.set_result(result)
        except BaseException as exc:
            future.set_exception(exc)

    Thread(target=prepare, name="idle-target-shard").start()
    return {"started": True}


def wait_stage(engine):
    future = getattr(engine, "_idle_stage_future", None)
    return future.result(timeout=600) if future is not None else None


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def handoff(engine, action, path=None):
    if getattr(engine, "_adaptive_memory", None) is None:
        raise RuntimeError("Independent handoff requires rank-local adaptive memory")
    key = device_key(engine)
    if action == "identity":
        return {"rank": engine.rank, "device": key, "local_device": engine.device.index}
    root = Path(path)
    config = json.loads((root / "transfer.json").read_text())
    token = config["token"]

    def phase(operation, fn):
        started = time.monotonic()
        value = fn()
        request_trace.emit(
            "idle_rank_transfer_step",
            rank=engine.rank,
            device=key,
            token=token,
            direction=config["direction"],
            operation=operation,
            seconds=time.monotonic() - started,
        )
        return value

    try:
        if action == "release":
            if not engine.is_offload:
                phase("source_finish", lambda: engine.colocation_memory("finish_train"))
            phase("source_park_cache", lambda: engine.colocation_memory("park_cache"))
            phase("source_offload", engine.offload)
            atomic_json(root / f"released-{key}.json", {"token": token})
            return {"device": key, "released": True}
        if action != "restore":
            raise ValueError(f"Unknown handoff action: {action}")
        staged = phase("destination_stage_wait", lambda: wait_stage(engine))
        if (
            staged is None
            or staged["snapshot_generation"] != config["snapshot_generation"]
        ):
            raise RuntimeError(
                "Prepared snapshot generation does not match the transfer"
            )
        if engine._idle_dram.pending_import is None:
            raise RuntimeError("Destination has no prepared state")
        if engine.lr_scheduler.last_epoch != config["scheduler_epoch"]:
            raise RuntimeError(
                "Prepared state is not the expected optimizer generation"
            )

        def wait_release():
            deadline = time.monotonic() + 600
            release = root / f"released-{key}.json"
            while True:
                if (root / "failed.json").exists():
                    raise RuntimeError("Another handoff participant failed")
                if key not in config["source_devices"]:
                    # The controller already joined inference offload on all devices.
                    return
                if release.exists():
                    if json.loads(release.read_text())["token"] != token:
                        raise RuntimeError("Stale release authorization")
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError("Source device release timed out")
                time.sleep(0.002)

        phase("device_release_wait", wait_release)
        data, layout, _ = engine._idle_dram.pending_import
        engine._idle_dram.pending_import = (data, layout, config["version"])
        engine.set_version(config["version"])
        phase("destination_restore", engine.onload)
        result = phase(
            "destination_verify", lambda: engine.idle_training_state("dram_verify")
        )
        engine._idle_stage_future = None
        request_trace.emit(
            "idle_rank_ready",
            rank=engine.rank,
            device=key,
            token=token,
            direction=config["direction"],
        )
        return result
    except BaseException:
        # Unique per-rank files avoid concurrent replacement of the same temporary file.
        (root / "failed.json").touch()
        raise
