# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from areal.infra.request_router import RequestLeasePool


def test_four_samples_keep_group_owner_and_balance_other_groups():
    pool = RequestLeasePool(["shared", "fixed"], 64)
    a = pool.acquire("a0", "a0", "rollout:0", 4)
    b = pool.acquire("b0", "b0", "rollout:1", 4)
    assert a["rank"] != b["rank"]
    pool.release("a0", a["lease_id"])
    for i in range(1, 4):
        lease = pool.acquire(f"a{i}", f"a{i}", "rollout:0", 4)
        assert lease["rank"] == a["rank"]
        pool.release(lease["ticket"], lease["lease_id"])
    pool.finish_group("rollout:0")
    assert "rollout:0" not in pool.groups
    with pytest.raises(RuntimeError, match="outstanding"):
        pool.finish_group("rollout:1")


def test_full_owner_does_not_block_other_card():
    pool = RequestLeasePool(["shared", "fixed"], 1)
    a = pool.acquire("a0", "a0", "a", 4)
    assert pool.acquire("a1", "a1", "a", 4) is None
    b = pool.acquire("b0", "b0", "b", 4)
    assert b["rank"] != a["rank"]
    pool.release("a0", a["lease_id"])
    assert pool.acquire("a1", "a1", "a", 4)["rank"] == a["rank"]


def test_group_withdrawal_migrates_remaining_samples_and_does_not_steal_back():
    pool = RequestLeasePool(["shared", "fixed"], 4)
    leases = [pool.acquire(f"a{i}", f"a{i}", "a", 4) for i in range(4)]
    assert {x["rank"] for x in leases} == {0}
    for lease in leases[:2]:  # Finished answers never need another lease.
        pool.release(lease["ticket"], lease["lease_id"])
    pool.close(0)
    pool.release("a2", leases[2]["lease_id"])  # Explicit abort ACK.
    resumed = pool.acquire("a2:2", "a2", "a", 4)
    assert resumed["rank"] == 1
    assert pool.active["a3"]["rank"] == 0  # Owner move does not mutate live lease.
    pool.release("a3", leases[3]["lease_id"])
    pool.wait_empty(0, timeout=0)
    pool.open(0, 1)
    assert pool.acquire("a3:2", "a3", "a", 4)["rank"] == 1
    assert pool.groups["a"]["rank"] == 1


def test_group_cleanup_preserves_unknown_execution_and_validates_metadata():
    pool = RequestLeasePool(["shared", "fixed"], 4)
    a = pool.acquire("a", "a", "g", 4)
    with pytest.raises(ValueError, match="metadata"):
        pool.acquire("a", "a", "different", 4)
    with pytest.raises(ValueError, match="size"):
        pool.acquire("b", "b", "g", 3)
    pool.cancel_wait("b")
    pool.release("a", a["lease_id"], uncertain=True)
    with pytest.raises(RuntimeError, match="outstanding"):
        pool.finish_group("g")
    assert "a" in pool.active and "g" in pool.groups and pool.error


def test_control_mode_ignores_group_affinity_and_namespace_is_trace_independent():
    from areal.infra.request_router import workflow_group_id

    pool = RequestLeasePool(["shared", "fixed"], 4, task_affinity=False)
    a = pool.acquire("a", "a", "g", 4)
    b = pool.acquire("b", "b", "g", 4)
    assert a["rank"] != b["rank"] and not pool.groups
    assert workflow_group_id(0, False) == "rollout:0"
    assert workflow_group_id(0, True) == "eval:0"
    assert workflow_group_id(None, False) is None


@pytest.mark.asyncio
async def test_vllm_public_worker_forwards_group_cleanup():
    from areal.engine.vllm_remote import RemotevLLMEngine

    engine = RemotevLLMEngine.__new__(RemotevLLMEngine)
    engine._engine = SimpleNamespace(_finish_request_group=AsyncMock())
    await engine._finish_request_group(3, False)
    engine._engine._finish_request_group.assert_awaited_once_with(3, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_workflow_finishes_routing_group_after_all_children_drain(failure):
    import torch

    from areal.api import RolloutWorkflow
    from areal.infra import workflow_context
    from areal.infra.remote_inf_engine import GroupedRolloutWorkflow

    seen = set()
    finished = []

    class Workflow(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            i = workflow_context.get().sample_idx
            try:
                await asyncio.sleep(0)
                if failure and i == 0:
                    raise ValueError("sample failed")
                return {"rewards": torch.tensor([float(i)])}
            finally:
                await asyncio.sleep(0)
                seen.add(i)

    async def finish(task_id, is_eval):
        assert seen == {0, 1, 2, 3}
        finished.append((task_id, is_eval))

    parent = workflow_context.get()
    workflow_context.set(workflow_context.WorkflowContext(task_id=9, is_eval=False))
    try:
        group = GroupedRolloutWorkflow(Workflow(), 4, Mock())
        engine = SimpleNamespace(_finish_request_group=finish)
        if failure:
            with pytest.raises(ValueError, match="sample failed"):
                await group.arun_episode(engine, {})
        else:
            await group.arun_episode(engine, {})
        assert finished == [(9, False)]
    finally:
        workflow_context.set(parent)


def test_vllm_public_worker_forwards_router_configuration():
    from areal.engine.vllm_remote import RemotevLLMEngine

    engine = RemotevLLMEngine.__new__(RemotevLLMEngine)
    engine._engine = Mock()
    engine.configure_request_router("controller:1234")
    engine._engine.configure_request_router.assert_called_once_with("controller:1234")


def test_fifo_capacity_and_cancelled_waiter():
    pool = RequestLeasePool(["shared", "fixed"], 1)
    a = pool.acquire("a", "a")
    b = pool.acquire("b", "b")
    assert {a["rank"], b["rank"]} == {0, 1}
    assert pool.acquire("c", "c") is None
    assert pool.acquire("d", "d") is None
    pool.release("a", a["lease_id"])
    assert pool.acquire("d", "d") is None
    pool.cancel_wait("c")
    assert pool.acquire("d", "d")["rank"] == a["rank"]


def test_withdraw_resume_keeps_active_requests_and_versions():
    pool = RequestLeasePool(["shared", "fixed"], 4)
    first = pool.acquire("answer:1", "answer")
    assert first["rank"] == 0
    pool.close(0)
    with pytest.raises(TimeoutError):
        pool.wait_empty(0, timeout=0)
    pool.release(first["ticket"], first["lease_id"])
    second = pool.acquire("answer:2", "answer")
    assert second["rank"] == 1 and second["version"] == 0
    pool.open(0, 1)
    assert pool.acquire("answer:2", "answer") == second
    other = pool.acquire("other", "other")
    assert other["rank"] == 0 and other["version"] == 1
    pool.release(first["ticket"], first["lease_id"])
    assert second["ticket"] in pool.active


def test_both_closed_queue_and_uncertain_transport_fail_closed():
    pool = RequestLeasePool(["shared", "fixed"], 1)
    pool.close(0)
    pool.close(1)
    assert pool.acquire("a", "a") is None
    pool.open(1, 3)
    lease = pool.acquire("a", "a")
    pool.release("a", lease["lease_id"], uncertain=True)
    with pytest.raises(RuntimeError, match="unknown"):
        pool.acquire("a:2", "a")
    with pytest.raises(RuntimeError):
        pool.wait_empty(1)


def test_granted_but_lost_acquire_ack_cancellation_fails_closed():
    pool = RequestLeasePool(["shared", "fixed"], 1)
    pool.acquire("a", "a")
    pool.cancel_wait("a")
    with pytest.raises(RuntimeError):
        pool.acquire("a:2", "a")


def test_request_cannot_hold_two_leases_and_stale_release_cannot_free_new_one():
    pool = RequestLeasePool(["shared", "fixed"], 1)
    a = pool.acquire("a", "r")
    assert pool.acquire("b", "r") is None
    pool.release("a", a["lease_id"])
    b = pool.acquire("b", "r")
    pool.release("b", a["lease_id"])
    assert pool.active["b"] == b
    with pytest.raises(RuntimeError):
        pool.set_version(b["rank"], 8)


@pytest.mark.asyncio
async def test_late_native_admission_returns_explicit_409_and_stream_is_rejected(
    monkeypatch,
):
    import inspect

    from fastapi import HTTPException

    from areal.engine.vllm_ext import areal_vllm_server as server
    from areal.engine.vllm_ext.request_admission import RoutingPaused

    monkeypatch.setattr(
        server, "original_generate", AsyncMock(side_effect=RoutingPaused("late"))
    )
    req = SimpleNamespace(request_id="r", stream=False)
    raw = SimpleNamespace(
        headers={"X-Areal-Request-Routing": "1", "X-Areal-Route-Rank": "1"}
    )
    response = await inspect.unwrap(server.generate)(req, raw)
    assert response.status_code == 409
    req.stream = True
    with pytest.raises(HTTPException) as error:
        await inspect.unwrap(server.generate)(req, raw)
    assert error.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", [[], [11, 12]])
async def test_cross_card_continuation_preserves_prefix_versions_and_budget(
    monkeypatch, prefix
):
    from areal.api import ModelRequest
    from areal.api.cli_args import GenerationHyperparameters
    from areal.infra import remote_inf_engine as module

    requests = []
    replies = [
        dict(
            output_tokens=prefix,
            output_logprobs=[-0.1] * len(prefix),
            stop_reason="abort",
            routed_experts=None,
        ),
        dict(
            output_tokens=[13],
            output_logprobs=[-0.2],
            stop_reason="stop",
            routed_experts=None,
        ),
    ]

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def json(self):
            return replies.pop(0)

        def raise_for_status(self):
            pass

    def request(method, url, **kwargs):
        requests.append((url, kwargs["json"]))
        return Response()

    session = SimpleNamespace(request=request)
    monkeypatch.setattr(
        module.workflow_context, "get_aiohttp_session", AsyncMock(return_value=session)
    )
    leases = [
        dict(ticket="r:1", lease_id=1, server="shared", rank=0, version=7),
        dict(ticket="r:2", lease_id=2, server="fixed", rank=1, version=8),
    ]
    monkeypatch.setattr(module, "acquire_route", AsyncMock(side_effect=leases))
    release = AsyncMock()
    monkeypatch.setattr(module, "route_call", release)
    engine = SimpleNamespace(
        config=SimpleNamespace(
            return_routed_experts=False, use_lora=False, request_timeout=10
        ),
        rid_to_address={},
        rid_queue=[],
        choose_server=lambda: "shared",
        _request_router_addr="router",
        workflow_executor=SimpleNamespace(is_paused=lambda: True),
        backend=SimpleNamespace(
            build_generation_request=lambda req, **kw: SimpleNamespace(
                method="POST",
                endpoint="/generate",
                payload={
                    "input": req.input_ids.copy(),
                    "budget": req.gconfig.max_new_tokens,
                },
            ),
            parse_generation_response=lambda x: SimpleNamespace(**x),
        ),
    )
    req = ModelRequest(
        rid="r",
        input_ids=[1, 2],
        gconfig=GenerationHyperparameters(n_samples=1, max_new_tokens=5, max_tokens=20),
    )
    result = await module.RemoteInfEngine.agenerate(engine, req)
    assert result.output_tokens == prefix + [13]
    assert result.output_versions == [7] * len(prefix) + [8]
    assert result.output_logprobs == [-0.1] * len(prefix) + [-0.2]
    assert requests[1] == (
        "http://fixed/generate",
        {"input": [1, 2] + prefix, "budget": 5 - len(prefix)},
    )
    assert release.await_count == 2
    assert all(not call.kwargs["uncertain"] for call in release.await_args_list)


@pytest.mark.asyncio
async def test_native_admission_and_pause_share_lock(monkeypatch):
    from vllm.v1.engine.async_llm import AsyncLLM

    from areal.engine.vllm_ext.request_admission import (
        RoutingPaused,
        install_admission_guard,
        routed_request,
    )

    lock = asyncio.Lock()
    running = asyncio.Event()
    running.set()
    entered = asyncio.Event()
    proceed = asyncio.Event()

    async def original(*args, **kwargs):
        entered.set()
        await proceed.wait()

    monkeypatch.setattr(AsyncLLM, "_add_request", original)
    install_admission_guard(lock, running)
    token = routed_request.set(True)
    try:
        task = asyncio.create_task(AsyncLLM._add_request(None))
        await entered.wait()
        assert lock.locked()
        proceed.set()
        await task
        async with lock:
            running.clear()
        with pytest.raises(RoutingPaused):
            await AsyncLLM._add_request(None)
    finally:
        routed_request.reset(token)


def test_token_observations_include_abort_and_ignore_discarded_output(monkeypatch):
    from vllm.v1.engine.output_processor import OutputProcessor

    from areal.utils import request_trace

    def process(self, outputs, *args):
        return None

    def abort(self, ids, *args):
        for rid in ids:
            self.request_states.pop(rid, None)

    monkeypatch.setattr(OutputProcessor, "process_outputs", process)
    monkeypatch.setattr(OutputProcessor, "abort_requests", abort)
    emit = Mock()
    monkeypatch.setattr(request_trace, "emit", emit)
    request_trace.install_token_trace()
    obj = SimpleNamespace(request_states={"r": object()})
    OutputProcessor.process_outputs(
        obj, [SimpleNamespace(request_id="r", new_token_ids=[1, 2])]
    )
    OutputProcessor.abort_requests(obj, ["r"])
    OutputProcessor.process_outputs(
        obj, [SimpleNamespace(request_id="r", new_token_ids=[3])]
    )
    assert [c.args[0] for c in emit.call_args_list] == ["token_first", "token_summary"]
    assert emit.call_args_list[-1].kwargs["tokens"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["fenced", "timeout"])
async def test_transport_only_retries_explicit_fenced_rejection(monkeypatch, failure):
    from areal.api import ModelRequest
    from areal.api.cli_args import GenerationHyperparameters
    from areal.infra import remote_inf_engine as module

    calls = []

    class Response:
        def __init__(self, index):
            self.status = 409 if index == 0 else 200

        async def __aenter__(self):
            if failure == "timeout":
                raise TimeoutError("unknown")
            return self

        async def __aexit__(self, *args):
            return None

        async def json(self):
            if self.status == 409:
                return {"areal_route_rejected": True}
            return dict(
                output_tokens=[4],
                output_logprobs=[-0.2],
                stop_reason="stop",
                routed_experts=None,
            )

        def raise_for_status(self):
            pass

    def request(*args, **kwargs):
        response = Response(len(calls))
        calls.append(kwargs["json"])
        return response

    session = SimpleNamespace(request=request)
    monkeypatch.setattr(
        module.workflow_context, "get_aiohttp_session", AsyncMock(return_value=session)
    )
    acquire = AsyncMock(
        side_effect=[
            dict(ticket="r:1", lease_id=1, server="old", rank=0, version=1),
            dict(ticket="r:2", lease_id=2, server="new", rank=1, version=2),
        ]
    )
    release = AsyncMock()
    monkeypatch.setattr(module, "acquire_route", acquire)
    monkeypatch.setattr(module, "route_call", release)
    engine = SimpleNamespace(
        config=SimpleNamespace(
            return_routed_experts=False, use_lora=False, request_timeout=10
        ),
        rid_to_address={},
        rid_queue=[],
        choose_server=lambda: "old",
        _request_router_addr="router",
        workflow_executor=SimpleNamespace(is_paused=lambda: True),
        backend=SimpleNamespace(
            build_generation_request=lambda req, **kw: SimpleNamespace(
                method="POST", endpoint="/generate", payload=req.input_ids.copy()
            ),
            parse_generation_response=lambda x: SimpleNamespace(**x),
        ),
    )
    req = ModelRequest(
        rid="r",
        input_ids=[1, 2],
        gconfig=GenerationHyperparameters(n_samples=1, max_new_tokens=5, max_tokens=20),
    )
    if failure == "timeout":
        with pytest.raises(TimeoutError):
            await module.RemoteInfEngine.agenerate(engine, req)
        assert acquire.await_count == 1
        assert release.await_args.kwargs["uncertain"]
    else:
        result = await module.RemoteInfEngine.agenerate(engine, req)
        assert calls == [[1, 2], [1, 2]]
        assert result.output_tokens == [4] and result.output_versions == [2]
        assert acquire.await_count == 2
        assert all(not c.kwargs["uncertain"] for c in release.await_args_list)


def test_retained_kv_waits_for_busy_target_and_pins_version():
    pool = RequestLeasePool(["shared", "fixed"], 1, task_affinity=False)
    source = pool.acquire("a", "a")
    other = pool.acquire("b", "b")
    assert source["rank"] == 0 and other["rank"] == 1
    pool.running_moved.add("a")
    pool.close(0)
    assert pool.force_prefix_evacuation(0, [1]) == [source]
    pool.pin_retained_kv(source, 1)
    with pytest.raises(RuntimeError, match="pins replica epoch"):
        pool.open(1, 1)
    assert pool.acquire("a2", "a") is None
    pool.release("a", source["lease_id"])
    assert pool.acquire("a2", "a") is None
    pool.release("b", other["lease_id"])
    resumed = pool.acquire("a2", "a")
    assert resumed["rank"] == 1 and resumed["version"] == 0
    assert resumed["running_handoff"] and pool.retained_kv_pins
    with pytest.raises(RuntimeError, match="pins replica epoch"):
        pool.open(1, 1)
    pool.complete_retained_kv(source)
    assert not pool.retained_kv_pins


def test_retained_prefix_allows_changed_version_without_stale_kv_pin():
    pool = RequestLeasePool(["shared", "fixed"], 2)
    source = pool.acquire("a", "a", "g", 6)
    pool.close(0)
    pool.open(1, 2)
    pool.force_prefix_evacuation(0, [1])
    with pytest.raises(RuntimeError, match="changed"):
        pool.pin_retained_kv(source, 1)
    pool.release("a", source["lease_id"])
    resumed = pool.acquire("a2", "a", "g", 6)
    assert resumed["version"] == 2 and not resumed["running_handoff"]


def test_retained_kv_fallback_unpins_destination_without_releasing_source():
    pool = RequestLeasePool(["shared", "fixed"], 2)
    source = pool.acquire("a", "a")
    pool.close(0)
    pool.force_prefix_evacuation(0, [1])
    pool.pin_retained_kv(source, 1)
    pool.cancel_retained_kv(source, [1])
    assert pool.active["a"] == source
    assert not pool.retained_kv_pins and not pool.running_versions
    pool.open(1, 3)


def test_retained_import_can_enter_closing_target_until_native_ack():
    pool = RequestLeasePool(["shared", "fixed"], 1, task_affinity=False)
    source = pool.acquire("a", "a")
    pool.close(0)
    pool.force_prefix_evacuation(0, [1])
    pool.pin_retained_kv(source, 1)
    pool.closing.add(1)
    pool.release("a", source["lease_id"])
    assert pool.acquire("ordinary", "ordinary") is None
    assert pool.acquire("a2", "a")["rank"] == 1
    assert pool.retained_kv_pins
    pool.complete_retained_kv(source)
    assert not pool.retained_kv_pins
