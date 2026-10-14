# SPDX-License-Identifier: Apache-2.0
"""Fence routed native admission against pause before KV/weight transitions."""

from contextvars import ContextVar
from functools import wraps

routed_request = ContextVar("areal_routed_request", default=False)


class RoutingPaused(ValueError):
    pass


def install_admission_guard(lock, run_event):
    from vllm.v1.engine.async_llm import AsyncLLM

    original = AsyncLLM._add_request
    if getattr(original, "_areal_admission_guard", False):
        return

    @wraps(original)
    async def add(self, *args, **kwargs):
        if not routed_request.get():
            return await original(self, *args, **kwargs)
        while True:
            async with lock:
                if run_event.is_set():
                    return await original(self, *args, **kwargs)
                if not getattr(self, "_areal_preserve_pause", False):
                    raise RoutingPaused("Replica no longer accepts this segment")
            # A lease granted immediately before pause remains on this replica.
            # Never hold the transition lock while waiting for resume.
            await run_event.wait()

    add._areal_admission_guard = True
    AsyncLLM._add_request = add
