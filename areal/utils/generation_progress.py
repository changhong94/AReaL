# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU-only generation progress; independent of synchronized tracing."""

import functools
import os
import time
from collections import deque
from threading import RLock


class ProgressRegistry:
    def __init__(self):
        self.lock = RLock()
        self.active = {}
        self.history = deque(maxlen=256)
        self.profile_enabled = os.getenv("AREAL_BATCH_DRAIN_METRICS", "0") == "1"
        self.decode_bins = {}
        self.first_samples = []
        self.previous_frame = None
        self.window_started = time.monotonic()

    def add(self, internal_id, request_id, prompt_tokens):
        with self.lock:
            self.active[internal_id] = dict(
                request_id=request_id.removeprefix("generate-tokens-"),
                prompt_tokens=prompt_tokens,
                queued=time.monotonic(),
                first=None,
                last=None,
                tokens=0,
                seen=False,
            )

    def advance(self, internal_id, count):
        if not count:
            return
        with self.lock:
            row = self.active.get(internal_id)
            if row is not None:
                row["seen"] = True
                now = time.monotonic()
                if row["first"] is None:
                    row["first"] = now
                    if self.profile_enabled:
                        self.first_samples.append(
                            dict(
                                seconds=now - row["queued"],
                                prompt_tokens=row["prompt_tokens"],
                            )
                        )
                row["last"] = now
                row["tokens"] += count

    def retire_missing(self, live_ids):
        with self.lock:
            for rid in list(self.active):
                if rid in live_ids:
                    self.active[rid]["seen"] = True
                elif self.active[rid]["seen"]:
                    row = self.active.pop(rid)
                    if row["tokens"] > 1 and row["last"] > row["first"]:
                        self.history.append(row)

    def break_service_interval(self):
        with self.lock:
            self.previous_frame = None

    def observe_frame(self, outputs):
        if not self.profile_enabled:
            return
        with self.lock:
            now = time.monotonic()
            rows = [self.active.get(o.request_id) for o in outputs]
            pure = bool(rows) and all(
                r is not None and r["tokens"] > 0 and len(o.new_token_ids) == 1
                for r, o in zip(rows, outputs)
            )
            if pure:
                concurrency = 2 ** ((len(rows) - 1).bit_length())
                context = int(
                    sum(r["prompt_tokens"] + r["tokens"] for r in rows) / len(rows)
                )
                context = 2 ** max(0, (max(1, context) - 1).bit_length())
                previous = self.previous_frame
                if previous is not None:
                    key = (concurrency, context)
                    row = self.decode_bins.setdefault(
                        key,
                        dict(
                            concurrency=concurrency,
                            context=context,
                            seconds=0.0,
                            count=0,
                            max_concurrency=0,
                        ),
                    )
                    row["seconds"] += now - previous
                    row["count"] += 1
                    row["max_concurrency"] = max(row["max_concurrency"], len(rows))
                self.previous_frame = now
            else:
                # Prefill or untracked output must not become a decode sample.
                self.previous_frame = None

    def snapshot(self, rotate=False):
        with self.lock:
            now = time.monotonic()
            result = dict(
                time=now,
                active=[dict(r) for r in self.active.values()],
                history=[dict(r) for r in self.history],
            )
            if rotate:
                result["window"] = dict(
                    start=self.window_started,
                    cutoff=now,
                    decode=[dict(r) for r in self.decode_bins.values()],
                    first=list(self.first_samples),
                )
                self.decode_bins.clear()
                self.first_samples.clear()
                self.previous_frame = None
                self.window_started = now
            return result

    def discard_external(self, request_id):
        """Remove final routed requests, including failed pre-engine submission."""
        with self.lock:
            for rid in list(self.active):
                if self.active[rid]["request_id"] == request_id:
                    self.active.pop(rid)


registry = ProgressRegistry()


def install():
    if os.getenv("AREAL_DRAIN_METRICS", "0") != "1":
        return
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm.v1.engine.output_processor import OutputProcessor

    original_assign = InputProcessor.assign_request_id
    if getattr(original_assign, "_areal_progress", False):
        return
    original_process = OutputProcessor.process_outputs
    original_abort = OutputProcessor.abort_requests

    @functools.wraps(original_assign)
    def assign(request):
        result = original_assign(request)
        registry.add(
            request.request_id,
            request.external_req_id,
            len(request.prompt_token_ids or []),
        )
        return result

    @functools.wraps(original_process)
    def process(self, outputs, *args, **kwargs):
        registry.observe_frame(outputs)
        for output in outputs:
            if output.request_id in self.request_states:
                registry.advance(output.request_id, len(output.new_token_ids))
        result = original_process(self, outputs, *args, **kwargs)
        registry.retire_missing(self.request_states)
        return result

    @functools.wraps(original_abort)
    def abort(self, *args, **kwargs):
        registry.retire_missing(self.request_states)
        result = original_abort(self, *args, **kwargs)
        registry.retire_missing(self.request_states)
        return result

    assign._areal_progress = True
    InputProcessor.assign_request_id = staticmethod(assign)
    OutputProcessor.process_outputs = process
    OutputProcessor.abort_requests = abort
