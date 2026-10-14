# SPDX-License-Identifier: Apache-2.0
"""CPU contract test against the pinned vLLM queue and abort implementation."""

from queue import Queue
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm")


def test_real_vllm_finish_and_abort_notification(monkeypatch):
    from vllm import SamplingParams
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine import FinishReason
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.request import Request

    from areal.engine.vllm_ext.queue_rebalance import install_queue_rebalance

    monkeypatch.setenv("AREAL_DRAIN_METRICS", "1")
    monkeypatch.setenv("AREAL_QUEUE_REBALANCE", "1")
    install_queue_rebalance()
    installed = Scheduler.schedule
    install_queue_rebalance()
    assert Scheduler.schedule is installed
    req = Request("native-attempt-1", [1, 2], SamplingParams(max_tokens=8), None)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.requests = {req.request_id: req}
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue([req])
    scheduler.skipped_waiting = FCFSRequestQueue()
    scheduler.max_num_running_reqs = 1
    scheduler._free_request = Mock(
        side_effect=lambda r, **k: scheduler.requests.pop(r.request_id)
    )
    core = EngineCoreProc.__new__(EngineCoreProc)
    core.scheduler = scheduler
    core.output_queue = Queue()
    assert core.areal_queue_rebalance("snapshot", [])["queued"] == [req.request_id]
    assert core.areal_queue_rebalance("abort", [req.request_id])["aborted"] == [
        req.request_id
    ]
    client, output = core.output_queue.get_nowait()
    assert client == 0 and output.finished_requests == [req.request_id]
    assert output.outputs[0].finish_reason == FinishReason.ABORT
    assert output.outputs[0].new_token_ids == []
    assert not scheduler.waiting
    assert core.areal_queue_rebalance("abort", [req.request_id])["aborted"] == []
    assert core.output_queue.empty()
