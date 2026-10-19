# SPDX-License-Identifier: Apache-2.0
"""CPU contract test against the pinned vLLM queue and abort implementation."""

from queue import Queue
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm")


@pytest.mark.parametrize("running", [False, True])
def test_real_vllm_finish_and_abort_notification(monkeypatch, running):
    from vllm import SamplingParams
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine import FinishReason
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.request import Request, RequestStatus

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
    if running:
        monkeypatch.setenv("AREAL_RUNNING_REBALANCE", "active")
        req.status = RequestStatus.RUNNING
        req.append_output_token_ids([3, 4])
        scheduler.running = [req]
        scheduler.waiting = FCFSRequestQueue()
    scheduler._free_request = Mock(
        side_effect=lambda r, **k: scheduler.requests.pop(r.request_id)
    )
    core = EngineCoreProc.__new__(EngineCoreProc)
    core.scheduler = scheduler
    core.output_queue = Queue()
    assert core.areal_queue_rebalance("snapshot", [])[
        "running" if running else "queued"
    ] == [req.request_id]
    action = "abort_running" if running else "abort"
    assert core.areal_queue_rebalance(action, [req.request_id])["aborted"] == [
        req.request_id
    ]
    client, output = core.output_queue.get_nowait()
    assert client == 0 and output.finished_requests == [req.request_id]
    assert output.outputs[0].finish_reason == FinishReason.ABORT
    assert output.outputs[0].new_token_ids == []
    assert not scheduler.waiting
    assert core.areal_queue_rebalance(action, [req.request_id])["aborted"] == []
    assert core.output_queue.empty()


def test_final_only_abort_output_preserves_accumulated_prefix_and_logprobs():
    from types import SimpleNamespace

    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine import FinishReason
    from vllm.v1.engine.output_processor import RequestState

    # Real output assembly must use accumulated state even when the engine
    # cancellation notification carries no new tokens.
    state = RequestState.__new__(RequestState)
    state.output_kind = RequestOutputKind.FINAL_ONLY
    state.detokenizer = SimpleNamespace(
        output_token_ids=[3, 4], get_next_output_text=lambda *_: "prefix"
    )
    logs = [{3: -0.1}, {4: -0.2}]
    state.logprobs_processor = SimpleNamespace(logprobs=logs, cumulative_logprob=-0.3)
    state.routed_experts_chunks = []
    state.request_index = 0
    output = state._new_completion_output([], FinishReason.ABORT, None)
    assert output.token_ids == [3, 4]
    assert output.logprobs == logs
    assert output.finish_reason == "abort"
