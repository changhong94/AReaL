# SPDX-License-Identifier: Apache-2.0
"""Opt-in append-only request diagnostics, including unfinished requests.

AREAL_REQUEST_TRACE_DIR enables request tracing. AREAL_NPU_DEVICE_TRACE=1 also
enables synchronized device windows (off by default), which change performance.
"""

import asyncio
import functools
import json
import os
import threading
import time
from pathlib import Path

_lock = threading.Lock()


def enabled() -> bool:
    return bool(os.environ.get("AREAL_REQUEST_TRACE_DIR"))


def emit(event: str, **fields) -> None:
    directory = os.environ.get("AREAL_REQUEST_TRACE_DIR")
    if not directory:
        return
    record = dict(event=event, time=time.time(), pid=os.getpid(), **fields)
    with _lock:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"events-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")


def identity(request_id: str) -> dict:
    from areal.utils import perf_tracer

    tracer = perf_tracer.get_session_tracer()
    return dict(
        request_id=request_id,
        task_id=perf_tracer.get_task_id(),
        session_id=perf_tracer.get_session_id(),
        rank=getattr(tracer, "_rank", None),
        role=getattr(tracer, "_role", None),
    )


def trace_errors(fn):
    @functools.wraps(fn)
    async def wrapped(self, req, *args, **kwargs):
        try:
            return await fn(self, req, *args, **kwargs)
        except BaseException as exc:
            if enabled():
                emit(
                    "answer_cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "answer_error",
                    error=type(exc).__name__,
                    **identity(req.rid),
                )
            raise

    return wrapped


def install_request_id_trace() -> None:
    if not enabled():
        return
    from vllm.v1.engine.input_processor import InputProcessor

    original = InputProcessor.assign_request_id
    if getattr(original, "_areal_traced", False):
        return

    @functools.wraps(original)
    def assign(request):
        result = original(request)
        emit(
            "native_id",
            request_id=request.external_req_id,
            internal_id=request.request_id,
        )
        return result

    assign._areal_traced = True
    InputProcessor.assign_request_id = staticmethod(assign)
    install_token_trace()


def install_token_trace() -> None:
    from vllm.v1.engine.output_processor import OutputProcessor

    original_process = OutputProcessor.process_outputs
    if getattr(original_process, "_areal_traced", False):
        return
    original_abort = OutputProcessor.abort_requests

    def flush(self):
        observed = getattr(self, "_areal_token_observations", {})
        for rid in list(observed):
            if rid not in self.request_states:
                emit("token_summary", internal_id=rid, **observed.pop(rid))

    @functools.wraps(original_process)
    def process(self, outputs, *args, **kwargs):
        observed = getattr(self, "_areal_token_observations", None)
        if observed is None:
            observed = self._areal_token_observations = {}
        for output in outputs:
            rid = output.request_id
            if rid not in self.request_states or not output.new_token_ids:
                continue
            now = time.time()
            if rid not in observed:
                observed[rid] = dict(
                    first_token_time=now, last_token_time=now, tokens=0
                )
                emit("token_first", internal_id=rid, token_time=now)
            observed[rid]["last_token_time"] = now
            observed[rid]["tokens"] += len(output.new_token_ids)
        result = original_process(self, outputs, *args, **kwargs)
        flush(self)
        return result

    @functools.wraps(original_abort)
    def abort(self, *args, **kwargs):
        result = original_abort(self, *args, **kwargs)
        flush(self)
        return result

    process._areal_traced = True
    OutputProcessor.process_outputs = process
    OutputProcessor.abort_requests = abort


def install_scheduler_trace() -> None:
    """Record queue admission in the process that owns the scheduler."""
    if not enabled():
        return
    from vllm.v1.core.sched.scheduler import Scheduler

    original_add = Scheduler.add_request
    if not getattr(original_add, "_areal_traced", False):

        @functools.wraps(original_add)
        def add(self, request):
            result = original_add(self, request)
            emit("native_queue", internal_id=request.request_id)
            return result

        add._areal_traced = True
        Scheduler.add_request = add


def install_device_trace() -> None:
    if not enabled():
        return
    install_scheduler_trace()
    if os.environ.get("AREAL_NPU_DEVICE_TRACE") != "1":
        return

    import torch
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original = NPUModelRunner.execute_model
    if getattr(original, "_areal_traced", False):
        return

    @functools.wraps(original)
    def execute(self, scheduler_output, *args, **kwargs):
        if not enabled() or os.environ.get("AREAL_NPU_DEVICE_TRACE") != "1":
            return original(self, scheduler_output, *args, **kwargs)
        from vllm.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return original(self, scheduler_output, *args, **kwargs)
        ids = list(scheduler_output.num_scheduled_tokens)
        if not ids:
            return original(self, scheduler_output, *args, **kwargs)
        # These are synchronized model-step envelopes, not exclusive per-request
        # kernel durations. All scheduled requests share the same device window.
        torch.npu.synchronize()
        step_id = getattr(self, "_areal_trace_step", 0) + 1
        self._areal_trace_step = step_id
        emit(
            "device_step_begin",
            step_id=step_id,
            internal_ids=ids,
            scheduled_tokens=dict(scheduler_output.num_scheduled_tokens),
            logical_device=torch.npu.current_device(),
        )
        try:
            result = original(self, scheduler_output, *args, **kwargs)
            torch.npu.synchronize()
        except BaseException:
            emit("device_step_error", step_id=step_id, internal_ids=ids)
            raise
        emit("device_step_end", step_id=step_id, internal_ids=ids)
        return result

    execute._areal_traced = True
    NPUModelRunner.execute_model = execute
