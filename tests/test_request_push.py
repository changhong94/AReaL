# SPDX-License-Identifier: Apache-2.0
import asyncio

import aiohttp
import pytest

from areal.infra.request_push import PushRouteClient, PushRouteServer
from areal.infra.request_router import RequestLeasePool


def payload(ticket):
    return dict(ticket=ticket, request_id=ticket)


@pytest.fixture
def service():
    pool = RequestLeasePool(["engine"], 1)
    server = PushRouteServer(pool)
    server.start()
    yield pool, server
    server.stop()


async def receive(ws):
    return await asyncio.wait_for(ws.receive_json(), 3)


@pytest.mark.asyncio
async def test_resource_event_releases_queued_request(service):
    pool, server = service
    async with aiohttp.ClientSession() as session:
        client = PushRouteClient(session, server.url)
        try:
            first = await asyncio.wait_for(client.acquire(payload("a")), 3)
            waiting = asyncio.create_task(client.acquire(payload("b")))
            await asyncio.sleep(0.05)
            assert not waiting.done()
            pool.release("a", first["lease_id"])
            second = await asyncio.wait_for(waiting, 3)
            assert second["ticket"] == "b"
            pool.release("b", second["lease_id"])
            assert not pool.active
        finally:
            await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["before_offer", "before_ack", "before_ready"])
async def test_disconnect_replays_same_lease(service, stage):
    pool, server = service
    if stage == "before_offer":
        pool.close(0)
    async with aiohttp.ClientSession() as session:
        ws = await session.ws_connect(server.url, params={"worker": "worker"})
        await ws.send_json(dict(kind="register", ticket="a", payload=payload("a")))
        lease_id = None
        if stage != "before_offer":
            offer = await receive(ws)
            lease_id = offer["lease"]["lease_id"]
            if stage == "before_ready":
                await ws.send_json(dict(kind="ack", ticket="a", lease_id=lease_id))
                # Receive, but deliberately discard the acknowledgement response.
                assert (await receive(ws))["kind"] == "ready"
        await ws.close()
        if stage == "before_offer":
            pool.open(0, 0)
        ws = await session.ws_connect(server.url, params={"worker": "worker"})
        await ws.send_json(dict(kind="register", ticket="a", payload=payload("a")))
        response = await receive(ws)
        assert response["kind"] in ("offer", "ready")
        actual = response["lease"]["lease_id"]
        assert lease_id is None or actual == lease_id
        await ws.send_json(dict(kind="ack", ticket="a", lease_id=actual))
        assert (await receive(ws))["kind"] == "ready"
        assert len(pool.active) == 1
        pool.release("a", actual)
        await ws.send_json(dict(kind="register", ticket="a", payload=payload("a")))
        assert (await receive(ws))["kind"] == "error"
        assert not pool.active
        await ws.close()


@pytest.mark.asyncio
async def test_duplicate_ack_and_notification_never_allocate_again(service):
    pool, server = service
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(server.url, params={"worker": "worker"}) as ws:
            register = dict(kind="register", ticket="a", payload=payload("a"))
            await ws.send_json(register)
            first = (await receive(ws))["lease"]
            for _ in range(3):
                await ws.send_json(register)
                assert (await receive(ws))["lease"] == first
                await ws.send_json(
                    dict(kind="ack", ticket="a", lease_id=first["lease_id"])
                )
                assert (await receive(ws))["lease"] == first
            assert len(pool.active) == 1
            pool.release("a", first["lease_id"])


@pytest.mark.asyncio
async def test_cancelled_ticket_cannot_be_registered_late(service):
    pool, server = service
    pool.cancel_wait("a")
    async with aiohttp.ClientSession() as session:
        client = PushRouteClient(session, server.url)
        try:
            with pytest.raises(RuntimeError, match="retired"):
                await asyncio.wait_for(client.acquire(payload("a")), 3)
            assert not pool.active
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_epoch_change_fails_without_reexecution(service):
    pool, server = service
    async with aiohttp.ClientSession() as session:
        client = PushRouteClient(session, server.url.replace(server.epoch, "obsolete"))
        try:
            with pytest.raises(RuntimeError, match="identity"):
                await asyncio.wait_for(client.acquire(payload("a")), 3)
            assert not pool.active
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_many_requests_execute_once(service):
    pool, server = service
    seen = set()
    async with aiohttp.ClientSession() as session:
        client = PushRouteClient(session, server.url)

        async def run(i):
            ticket = str(i)
            lease = await client.acquire(payload(ticket))
            assert ticket not in seen
            seen.add(ticket)
            await asyncio.sleep(0.001)
            pool.release(ticket, lease["lease_id"])

        try:
            await asyncio.wait_for(asyncio.gather(*(run(i) for i in range(64))), 10)
            assert len(seen) == 64 and not pool.active
        finally:
            await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("dropped_kind", ["offer", "ready"])
async def test_client_recovers_lost_message_once(service, dropped_kind):
    pool, server = service
    original = server._send
    dropped = []

    async def send(owner, message):
        if message["kind"] == dropped_kind and not dropped:
            dropped.append(message["lease"]["lease_id"])
            await server.connections[owner].close()
            return
        await original(owner, message)

    server._send = send
    async with aiohttp.ClientSession() as session:
        client = PushRouteClient(session, server.url)
        try:
            lease = await asyncio.wait_for(client.acquire(payload("a")), 5)
            assert dropped == [lease["lease_id"]]
            assert pool.serial == 1 and len(pool.active) == 1
            pool.release("a", lease["lease_id"])
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_slow_receiver_does_not_block_other_worker(service):
    pool, server = service
    pool.capacity = 2
    original = server._send

    async def send(owner, message):
        if message.get("ticket") == "slow":
            await asyncio.sleep(1)
        await original(owner, message)

    server._send = send
    async with aiohttp.ClientSession() as session:
        slow = PushRouteClient(session, server.url)
        fast = PushRouteClient(session, server.url)
        task = asyncio.create_task(slow.acquire(payload("slow")))
        try:
            await asyncio.sleep(0.05)
            lease = await asyncio.wait_for(fast.acquire(payload("fast")), 0.5)
            pool.release("fast", lease["lease_id"])
            other = await asyncio.wait_for(task, 4)
            pool.release("slow", other["lease_id"])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await slow.close()
            await fast.close()


@pytest.mark.asyncio
async def test_disconnect_after_ready_does_not_replay_running_work(service):
    pool, server = service
    async with aiohttp.ClientSession() as session:
        client = PushRouteClient(session, server.url)
        try:
            first = await asyncio.wait_for(client.acquire(payload("a")), 3)
            await client.ws.close()
            waiting = asyncio.create_task(client.acquire(payload("b")))
            await asyncio.sleep(0.2)
            assert pool.serial == 1 and not waiting.done()
            assert pool.active["a"] == first
            pool.release("a", first["lease_id"])
            second = await asyncio.wait_for(waiting, 3)
            assert pool.serial == 2
            pool.release("b", second["lease_id"])
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_queued_cancellation_uses_existing_cleanup(service, monkeypatch):
    from areal.infra import request_router as router

    pool, server = service
    pool.close(0)
    monkeypatch.delenv("AREAL_ROUTE_TRANSPORT", raising=False)

    async def call(session, address, action, **kwargs):
        if action == "push_endpoint":
            return server.url
        assert action == "cancel"
        pool.cancel_wait(**kwargs)

    monkeypatch.setattr(router, "route_call", call)
    async with aiohttp.ClientSession() as session:
        task = asyncio.create_task(
            router.acquire_route(session, "controller", "a", "a")
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        pool.open(0, 0)
        await asyncio.sleep(0.05)
        assert "a" in pool.push_cancelled and not pool.active
        await session._areal_push_clients["controller"].close()


@pytest.mark.asyncio
async def test_explicit_disabled_endpoint_falls_back_before_registration(monkeypatch):
    from areal.infra import request_router as router

    calls = []
    monkeypatch.setenv("AREAL_ROUTE_TRANSPORT", "push_or_poll")

    async def call(session, address, action, **kwargs):
        calls.append(action)
        if action == "push_endpoint":
            return None
        assert action == "acquire"
        return {"ticket": kwargs["ticket"], "lease_id": 1}

    monkeypatch.setattr(router, "route_call", call)
    async with aiohttp.ClientSession() as session:
        assert (await router.acquire_route(session, "controller", "a", "a"))[
            "lease_id"
        ] == 1
    assert calls == ["push_endpoint", "acquire"]


@pytest.mark.asyncio
async def test_expensive_admission_does_not_starve_ack(service):
    import time

    pool, server = service
    pool.capacity = 2
    original = pool.acquire

    def acquire(**kwargs):
        if kwargs["ticket"] == "slow_scan":
            time.sleep(0.8)
        return original(**kwargs)

    pool.acquire = acquire
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(server.url, params={"worker": "worker"}) as ws:
            await ws.send_json(dict(kind="register", ticket="a", payload=payload("a")))
            first = (await receive(ws))["lease"]
            await ws.send_json(
                dict(kind="register", ticket="slow_scan", payload=payload("slow_scan"))
            )
            await asyncio.sleep(0.05)
            await ws.send_json(dict(kind="ack", ticket="a", lease_id=first["lease_id"]))
            message = await asyncio.wait_for(ws.receive_json(), 0.3)
            assert message["kind"] == "ready" and message["ticket"] == "a"
            pool.release("a", first["lease_id"])
            second = (await receive(ws))["lease"]
            pool.release("slow_scan", second["lease_id"])
