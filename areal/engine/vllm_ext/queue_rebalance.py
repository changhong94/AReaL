# SPDX-License-Identifier: Apache-2.0
"""Engine-loop atomic withdrawal of requests that have never been scheduled."""

import os
from functools import wraps


def eligible(scheduler, request, waiting_status):
    return (
        request.status == waiting_status
        and request.request_id not in getattr(scheduler, "_areal_ever_scheduled", set())
        and request.num_computed_tokens == 0
        and not request.output_token_ids
        and request not in scheduler.running
    )


def queue_operation(core, action, ids, waiting_status, aborted_status):
    """Called only by EngineCore's serialized utility dispatcher, never an API thread."""
    scheduler = core.scheduler
    if action not in ("snapshot", "abort", "abort_running"):
        raise ValueError("Unknown queue rebalance operation")
    available = [r for r in scheduler.waiting if eligible(scheduler, r, waiting_status)]
    if action == "snapshot":
        return dict(
            queued=[r.request_id for r in available],
            waiting=[r.request_id for r in scheduler.waiting],
            running=[r.request_id for r in scheduler.running],
            width=scheduler.max_num_running_reqs,
            computed_tokens={
                r.request_id: r.num_computed_tokens
                - getattr(r, "num_in_flight_tokens", 0)
                for r in scheduler.running
            }
            if os.getenv("AREAL_RUNNING_REBALANCE", "off") == "kv"
            else {},
            free_kv_tokens=(
                scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
                * scheduler.block_size
            )
            if hasattr(scheduler, "kv_cache_manager")
            else None,
        )
    if action == "abort_running":
        if os.getenv("AREAL_RUNNING_REBALANCE", "off") not in {"active", "kv"}:
            raise ValueError("Running rebalance is not enabled")
        available = list(scheduler.running)
    selected = set(ids)
    safe = [r.request_id for r in available if r.request_id in selected]
    # Eligibility is checked again here, after the planning round trip. No
    # schedule() can interleave between this check and removal from the queue.
    removed = scheduler.finish_requests(safe, aborted_status) if safe else []
    core._send_abort_outputs(removed)
    return dict(aborted=[r.request_id for r in removed])


def install_queue_rebalance():
    if (
        os.getenv("AREAL_QUEUE_REBALANCE", "1") != "1"
        or os.getenv("AREAL_DRAIN_METRICS", "0") != "1"
    ):
        return
    from packaging.version import Version
    from vllm import __version__

    if Version(__version__).release != (0, 26, 0):
        raise RuntimeError("Queue rebalance adapter requires vLLM 0.26.0")
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.request import RequestStatus

    original = Scheduler.schedule
    if getattr(original, "_areal_queue_rebalance", False):
        return

    @wraps(original)
    def schedule(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        seen = getattr(self, "_areal_ever_scheduled", set())
        seen.intersection_update(self.requests)
        seen.update(output.num_scheduled_tokens)
        self._areal_ever_scheduled = seen
        return output

    def operate(self, action: str, ids: list[str]):
        return queue_operation(
            self, action, ids, RequestStatus.WAITING, RequestStatus.FINISHED_ABORTED
        )

    schedule._areal_queue_rebalance = True
    Scheduler.schedule = schedule
    EngineCore.areal_queue_rebalance = operate
