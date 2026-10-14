# SPDX-License-Identifier: Apache-2.0
"""Single-host, immutable weight snapshots for staged NPU sharing.

The configured exchange directory should be on tmpfs for best performance.
Readers never unlink snapshots: the controller owns cleanup after all ACKs.
"""

import json
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

MANIFEST = "areal_weights.json"
DATA = "areal_weights.bin"
FORMAT = "areal-flat-weights-v1"
DTYPES = {
    name: getattr(torch, name)
    for name in (
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "int32",
        "int64",
        "bool",
    )
}


def validate_loaded_weights(expected: set[str], loaded: set[str] | None) -> None:
    if loaded is None:
        raise RuntimeError("Snapshot loader did not acknowledge parameter names")
    missing = expected - set(loaded)
    if missing:
        raise RuntimeError(f"Snapshot omitted weights: {sorted(missing)}")


def write_snapshot(
    directory: str,
    version: int,
    parameters: list[tuple[str, torch.Tensor]],
    compute_dtype: torch.dtype,
    materialize: Callable[[torch.Tensor], torch.Tensor],
    on_tensor: Callable[[str, torch.Tensor], None] | None = None,
) -> dict:
    """Cast on device, then synchronously copy into independently owned host bytes."""
    if version < 0 or not parameters:
        raise ValueError("Snapshot requires a version and nonempty parameters")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    entries, names, total = [], set(), 0
    for name, param in parameters:
        if not name or name in names:
            raise ValueError("Snapshot parameter names must be unique")
        names.add(name)
        dtype = compute_dtype if param.is_floating_point() else param.dtype
        dtype_name = str(dtype).removeprefix("torch.")
        if dtype_name not in DTYPES:
            raise ValueError(f"Unsupported snapshot dtype: {dtype}")
        size = param.numel() * torch.empty((), dtype=dtype).element_size()
        offset = (total + 63) // 64 * 64
        entries.append(
            dict(
                name=name,
                shape=list(param.shape),
                dtype=dtype_name,
                offset=offset,
                nbytes=size,
            )
        )
        total = offset + size
    if total == 0:
        raise ValueError("Snapshot has no tensor data")
    temporary = root / (MANIFEST + ".tmp")
    try:
        # Reserve backing storage up front: a sparse file could SIGBUS on write
        # when tmpfs is full, bypassing Python's normal error handling.
        with (root / DATA).open("xb") as f:
            os.posix_fallocate(f.fileno(), 0, total)
        storage = torch.from_file(
            str(root / DATA), shared=True, size=total, dtype=torch.uint8
        )
        with torch.no_grad():
            for entry, (_, param) in zip(entries, parameters):
                tensor = materialize(param).detach()
                if (
                    list(tensor.shape) != entry["shape"]
                    or tensor.dtype != DTYPES[entry["dtype"]]
                ):
                    raise ValueError(
                        "Materialized tensor does not match snapshot layout"
                    )
                if on_tensor is not None:
                    on_tensor(entry["name"], tensor)
                target = (
                    storage.narrow(0, entry["offset"], entry["nbytes"])
                    .view(tensor.dtype)
                    .view(tensor.shape)
                )
                target.copy_(tensor, non_blocking=False)
                del tensor, target
        del storage
        manifest = dict(format=FORMAT, version=version, nbytes=total, tensors=entries)
        temporary.write_text(json.dumps(manifest), encoding="utf-8")
        os.replace(temporary, root / MANIFEST)
        return manifest
    except BaseException:
        temporary.unlink(missing_ok=True)
        (root / DATA).unlink(missing_ok=True)
        raise


@contextmanager
def open_snapshot(
    directory: str, version: int
) -> Iterator[Iterator[tuple[str, torch.Tensor]]]:
    """Validate the entire layout before exposing any tensors to a model loader."""
    root = Path(directory)
    meta = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT or meta.get("version") != version:
        raise ValueError("Snapshot format/version mismatch")
    entries = meta.get("tensors")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Snapshot has no tensor metadata")
    total, names = 0, set()
    for entry in entries:
        name, shape, dtype_name = entry["name"], entry["shape"], entry["dtype"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Invalid snapshot parameter name")
        names.add(name)
        if (
            dtype_name not in DTYPES
            or not isinstance(shape, list)
            or any(type(d) is not int or d < 0 for d in shape)
        ):
            raise ValueError("Invalid snapshot dtype/shape")
        expected = (
            math.prod(shape) * torch.empty((), dtype=DTYPES[dtype_name]).element_size()
        )
        if entry["offset"] != (total + 63) // 64 * 64 or entry["nbytes"] != expected:
            raise ValueError("Invalid snapshot tensor bounds")
        total = entry["offset"] + expected
    if (
        total <= 0
        or total != meta.get("nbytes")
        or (root / DATA).stat().st_size != total
    ):
        raise ValueError("Snapshot storage size mismatch")
    # Private mapping: a loader cannot mutate another consumer's snapshot.
    storage = torch.from_file(
        str(root / DATA), shared=False, size=total, dtype=torch.uint8
    )

    def tensors():
        for entry in entries:
            yield (
                entry["name"],
                storage.narrow(0, entry["offset"], entry["nbytes"])
                .view(DTYPES[entry["dtype"]])
                .view(entry["shape"]),
            )

    weights = tensors()
    try:
        yield weights
    finally:
        weights.close()


class IndependentSnapshot:
    """Borrow stable single-rank weights; cast and copy without a producer queue.

    The caller must join before mutating/offloading the source storage. Only plain
    local tensors enter the writer: no FSDP collectives run on its private stream.
    At most one extra compute-dtype tensor is held by the snapshot writer.
    """

    def __init__(self, directory, version, parameters, compute_dtype):
        from concurrent.futures import Future
        from threading import Event, Thread

        from torch.distributed.tensor import DTensor

        if not parameters:
            raise ValueError("Snapshot requires nonempty parameters")
        sources = []
        for name, param in parameters:
            tensor = param.detach()
            if isinstance(tensor, DTensor):
                if tensor.device_mesh.size() != 1:
                    raise ValueError("Independent snapshot requires a single-rank mesh")
                tensor = tensor.to_local()
            if tuple(tensor.shape) != tuple(param.shape):
                raise ValueError("Independent snapshot requires complete local weights")
            sources.append((name, tensor))
        self.version = version
        self._parameters = sources
        self._device = sources[0][1].device
        if any(t.device != self._device for _, t in sources):
            raise ValueError("Snapshot sources must share a device")
        self._backend = (
            getattr(torch, self._device.type) if self._device.type != "cpu" else None
        )
        self._ready = None
        if self._backend is not None:
            self._ready = self._backend.Event()
            self._ready.record(self._backend.current_stream(self._device))
        self._future = Future()
        self._cancelled = Event()
        self._thread = Thread(
            target=self._run,
            args=(directory, compute_dtype),
            name=f"independent-snapshot-{version}",
        )
        self._thread.start()

    def _run(self, directory, compute_dtype):
        from contextlib import nullcontext

        from areal.utils import request_trace

        stream = None
        try:
            if self._backend is not None:
                self._backend.set_device(self._device)
                stream = self._backend.Stream(device=self._device)
                stream.wait_event(self._ready)
                context = self._backend.stream(stream)
            else:
                context = nullcontext()

            def materialize(tensor):
                if self._cancelled.is_set():
                    raise RuntimeError("Snapshot publication cancelled")
                return (
                    tensor.to(compute_dtype) if tensor.is_floating_point() else tensor
                )

            request_trace.emit("shared_snapshot_export_begin", version=self.version)
            with context:
                result = write_snapshot(
                    directory,
                    self.version,
                    self._parameters,
                    compute_dtype,
                    materialize,
                )
            request_trace.emit("shared_snapshot_export_end", version=self.version)
            self._future.set_result(result)
        except BaseException as exc:
            try:
                # A failed copy/layout check may leave a cast reading sources.
                if stream is not None:
                    stream.synchronize()
            finally:
                self._future.set_exception(exc)
        finally:
            # Release borrowed storage before a successful join permits offload.
            self._parameters = []

    def wait(self):
        self._thread.join()
        return self._future.result()

    def abort(self, error):
        self._cancelled.set()
        try:
            self.wait()
        except BaseException:
            pass


class BackgroundSnapshot:
    """Bounded, single-producer snapshot; no model collectives in the writer.

    At most two queued tensors plus the current copy are retained. A large
    individual parameter is never split. Producer events fence dtype casts;
    synchronous D2H on a private stream owns the bytes before releasing sources.
    """

    def __init__(self, directory, version, parameters, compute_dtype):
        from concurrent.futures import Future
        from queue import Queue
        from threading import Thread

        self.version = version
        self._queue = Queue(maxsize=2)
        self._future = Future()
        self._submitted = 0
        self.submit_seconds = 0.0
        self._parameters = parameters
        self._device = parameters[0][1].device
        self._backend = (
            getattr(torch, self._device.type) if self._device.type != "cpu" else None
        )
        self._thread = Thread(
            target=self._run,
            args=(directory, compute_dtype),
            name=f"snapshot-{version}",
        )
        self._thread.start()

    def _run(self, directory, compute_dtype):
        from contextlib import nullcontext

        from areal.utils import request_trace

        try:
            if self._backend is not None:
                self._backend.set_device(self._device)
                stream = self._backend.Stream(device=self._device)
                context = self._backend.stream(stream)
            else:
                stream, context = None, nullcontext()

            def receive(_):
                item = self._queue.get()
                if isinstance(item, BaseException):
                    raise item
                name, tensor, ready = item
                expected = self._parameters[self._received][0]
                if name != expected:
                    raise ValueError("Snapshot tensor order mismatch")
                self._received += 1
                if stream is not None:
                    stream.wait_event(ready)
                return tensor

            self._received = 0
            request_trace.emit("shared_snapshot_export_begin", version=self.version)
            with context:
                result = write_snapshot(
                    directory, self.version, self._parameters, compute_dtype, receive
                )
            request_trace.emit("shared_snapshot_export_end", version=self.version)
            self._future.set_result(result)
        except BaseException as exc:
            self._future.set_exception(exc)

    def _put(self, item):
        from queue import Full

        while True:
            if self._future.done():
                self._future.result()
                raise RuntimeError("Snapshot writer already finished")
            try:
                self._queue.put(item, timeout=0.05)
                return
            except Full:
                continue

    def submit(self, name, tensor):
        import time

        start = time.perf_counter()
        ready = None
        if self._backend is not None:
            ready = self._backend.Event()
            ready.record(self._backend.current_stream(self._device))
        self._put((name, tensor.detach(), ready))
        self._submitted += 1
        self.submit_seconds += time.perf_counter() - start

    def wait(self):
        self._thread.join()
        # Clear queued sources on failure; the writer no longer reads them.
        while not self._queue.empty():
            self._queue.get_nowait()
        return self._future.result()

    def abort(self, error):
        if self._submitted < len(self._parameters) and not self._future.done():
            try:
                self._put(error)
            except BaseException:
                pass
        try:
            self.wait()
        except BaseException:
            pass
