# SPDX-License-Identifier: Apache-2.0
"""Real HTTP fault injection for idempotent request-route callbacks."""

import asyncio
import socket
import struct
from collections import Counter
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web

from areal.infra import request_router as router


@asynccontextmanager
async def callback(handler):
    app = web.Application()
    app.router.add_post("/callback/request_route", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    address = "127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])
    try:
        async with aiohttp.ClientSession() as session:
            yield session, address
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_lost_acquire_and_release_ack_do_not_duplicate_lease(monkeypatch):
    monkeypatch.setenv("AREAL_ROUTE_TRANSPORT", "poll")
    """Drop replies after committing operations, then recover the exact lease."""
    pool = router.RequestLeasePool(["worker"], 1)
    counts = Counter()

    async def handler(request):
        payload = await request.json()
        action = payload.pop("action")
        counts[action] += 1
        if action == "acquire":
            result = pool.acquire(**payload)
        elif action == "cancel":
            pool.cancel_wait(**payload)
            result = True
        else:
            pool.release(**payload)
            result = True
        if counts[action] == 1:
            request.transport.get_extra_info("socket").setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            request.transport.abort()
        return web.json_response(result)

    async with callback(handler) as (session, address):
        lease = await router.acquire_route(session, address, "ticket", "request")
        assert lease["lease_id"] == pool.serial == 1
        assert len(pool.active) == 1
        await router.route_call(
            session, address, "release", ticket="ticket", lease_id=lease["lease_id"]
        )
    assert counts == {"acquire": 2, "release": 2}
    assert not pool.active
    assert pool.error is None


@pytest.mark.asyncio
async def test_semantic_http_error_is_not_retried():
    """A server-side invariant error must not be hidden by retries."""
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.json_response({"error": "version mismatch"}, status=500)

    async with callback(handler) as (session, address):
        with pytest.raises(aiohttp.ClientResponseError, match="500"):
            await router.route_call(session, address, "acquire", ticket="t")
    assert calls == 1


@pytest.mark.asyncio
async def test_permanent_disconnect_has_bounded_attempts():
    """A broken callback service eventually fails instead of retrying forever."""
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        request.transport.abort()
        return web.json_response(None)

    async with callback(handler) as (session, address):
        with pytest.raises(aiohttp.ClientConnectionError):
            await router.route_call(session, address, "acquire", ticket="t")
    assert calls == router._ROUTE_ATTEMPTS


@pytest.mark.asyncio
async def test_control_lane_progresses_while_acquire_lane_is_saturated():
    """Bound acquire requests without starving release or cancellation."""
    gate = asyncio.Event()
    saturated = asyncio.Event()
    active = peak = 0

    async def handler(request):
        nonlocal active, peak
        data = await request.json()
        if data["action"] == "acquire":
            active += 1
            peak = max(peak, active)
            if active == 16:
                saturated.set()
            await gate.wait()
            active -= 1
            return web.json_response(None)
        return web.json_response(True)

    async with callback(handler) as (session, address):
        tasks = [
            asyncio.create_task(
                router.route_call(session, address, "acquire", ticket=str(i))
            )
            for i in range(128)
        ]
        try:
            await asyncio.wait_for(saturated.wait(), 3)
            assert await asyncio.wait_for(
                router.route_call(
                    session, address, "release", ticket="old", lease_id=1
                ),
                1,
            )
            assert await asyncio.wait_for(
                router.route_call(session, address, "cancel", ticket="waiting"), 1
            )
        finally:
            gate.set()
            await asyncio.gather(*tasks)
    assert peak == 16


@pytest.mark.asyncio
async def test_local_queue_is_included_in_callback_deadline(monkeypatch):
    """Waiting for a local permit must not bypass the transport deadline."""
    monkeypatch.setattr(router, "_ROUTE_TIMEOUT", 0.02)
    address = "unused"
    lane = router._route_lane(address, "acquire")
    for _ in range(16):
        await lane.acquire()
    try:
        with pytest.raises(TimeoutError):
            await router.route_call(None, address, "acquire", ticket="t")
    finally:
        for _ in range(16):
            lane.release()


@pytest.mark.asyncio
async def test_many_lost_acknowledgements_preserve_unique_leases(monkeypatch):
    monkeypatch.setenv("AREAL_ROUTE_TRANSPORT", "poll")
    """Concurrent resets recover every ticket exactly once without generation replay."""
    pool = router.RequestLeasePool(["worker"], 64)
    attempts = Counter()

    async def handler(request):
        payload = await request.json()
        action = payload.pop("action")
        key = (action, payload["ticket"])
        attempts[key] += 1
        if action == "acquire":
            result = pool.acquire(**payload)
        elif action == "cancel":
            pool.cancel_wait(**payload)
            result = True
        else:
            pool.release(**payload)
            result = True
        if attempts[key] == 1:
            request.transport.get_extra_info("socket").setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            request.transport.abort()
        return web.json_response(result)

    async with callback(handler) as (session, address):

        async def one(i):
            ticket = str(i)
            lease = await router.acquire_route(session, address, ticket, ticket)
            await router.route_call(
                session, address, "release", ticket=ticket, lease_id=lease["lease_id"]
            )
            return lease["lease_id"]

        leases = await asyncio.gather(*(one(i) for i in range(64)))
    assert len(set(leases)) == pool.serial == 64
    assert not pool.active and not pool.waiting
    assert pool.error is None
    assert set(attempts.values()) == {2}


@pytest.mark.asyncio
async def test_cancelled_local_waiter_does_not_leak_permits():
    """Cancellation in the client queue cannot consume a routing permit."""
    lane = router._route_lane("unused-cancel", "acquire")
    for _ in range(16):
        await lane.acquire()
    task = asyncio.create_task(
        router.route_call(None, "unused-cancel", "acquire", ticket="t")
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(16):
        lane.release()
    for _ in range(16):
        await asyncio.wait_for(lane.acquire(), 1)
    assert lane.locked()
    for _ in range(16):
        lane.release()


@pytest.mark.asyncio
async def test_unknown_action_is_not_retried_on_disconnect():
    """Future non-idempotent actions must opt in explicitly to transport retries."""
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        request.transport.abort()
        return web.json_response(True)

    async with callback(handler) as (session, address):
        with pytest.raises(aiohttp.ClientConnectionError):
            await router.route_call(session, address, "future_action")
    assert calls == 1
