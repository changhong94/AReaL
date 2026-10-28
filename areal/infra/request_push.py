# SPDX-License-Identifier: Apache-2.0
"""Event-driven lease admission over one reconnectable worker WebSocket.

The pool remains the authority. Offers reserve capacity, ACK confirms ownership,
READY permits execution. Lost messages replay the same ticket/lease, never a new
execution. Disconnect does not release potentially executing work.
"""

import asyncio
import contextlib
import threading
import time
import uuid

import aiohttp
from aiohttp import web

from areal.utils import request_trace


class PushRouteServer:
    def __init__(self, pool, host="127.0.0.1"):
        self.pool = pool
        self.host = host
        self.url = None
        self.epoch = uuid.uuid4().hex
        self.loop = None
        self.ready = threading.Event()
        self.start_error = None
        self.records = {}
        self.retired = set()
        self.connections = {}
        self.sends = set()
        self.pool.push_cancelled = set()

    def start(self):
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()
        if not self.ready.wait(10):
            raise TimeoutError("Push route server startup timed out")
        if self.start_error:
            raise self.start_error

    def _thread_main(self):
        try:
            asyncio.run(self._serve())
        except Exception as exc:
            self.start_error = exc
            self.ready.set()

    def notify(self):
        if self.loop and not self.loop.is_closed():
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self.changed.set)

    async def _serve(self):
        self.loop = asyncio.get_running_loop()
        self.changed = asyncio.Event()
        self.stopping = asyncio.Event()
        app = web.Application()
        app.router.add_get("/leases", self._connection)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        host = f"[{self.host}]" if ":" in self.host else self.host
        self.url = f"http://{host}:{port}/leases?epoch={self.epoch}"
        self.pool.change_listeners.append(self.notify)
        dispatch = asyncio.create_task(self._dispatch())
        self.ready.set()
        try:
            stopping = asyncio.create_task(self.stopping.wait())
            done, _ = await asyncio.wait(
                [stopping, dispatch], return_when=asyncio.FIRST_COMPLETED
            )
            if dispatch in done:
                with self.pool.lock:
                    self.pool.error = "Push dispatcher stopped unexpectedly"
                request_trace.emit("route_push_fatal", error=self.pool.error)
                await asyncio.gather(
                    *(
                        self._send(
                            e["owner"],
                            dict(kind="error", ticket=t, error=self.pool.error),
                        )
                        for t, e in self.records.items()
                    )
                )
            stopping.cancel()
            await asyncio.gather(stopping, return_exceptions=True)
        finally:
            self.pool.change_listeners.remove(self.notify)
            dispatch.cancel()
            await asyncio.gather(dispatch, return_exceptions=True)
            for task in list(self.sends):
                task.cancel()
            await asyncio.gather(*self.sends, return_exceptions=True)
            for ws in list(self.connections.values()):
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(ws.close(), 1)
            await runner.cleanup()

    def stop(self):
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.stopping.set)
        self.thread.join(10)
        if self.thread.is_alive():
            raise TimeoutError("Push route server did not stop")

    async def _send(self, owner, message):
        ws = self.connections.get(owner)
        if ws is not None and not ws.closed:
            try:
                await asyncio.wait_for(ws.send_json(message), 1)
                request_trace.emit(
                    "route_push_sent",
                    kind=message["kind"],
                    ticket=message.get("ticket"),
                )
            except (ConnectionError, RuntimeError, TimeoutError):
                with contextlib.suppress(ConnectionError, RuntimeError, TimeoutError):
                    await asyncio.wait_for(ws.close(), 1)

    def _offer(self, owner, message):
        task = asyncio.create_task(self._send(owner, message))
        self.sends.add(task)
        task.add_done_callback(self.sends.discard)

    async def _dispatch(self):
        while True:
            await self.changed.wait()
            self.changed.clear()
            for ticket, entry in list(self.records.items()):
                try:
                    with self.pool.lock:
                        cancelled = ticket in self.pool.push_cancelled
                        active = ticket in self.pool.active
                    if cancelled or (entry.get("lease") is not None and not active):
                        self.retired.add(ticket)
                        del self.records[ticket]
                        if cancelled:
                            self._offer(
                                entry["owner"],
                                dict(
                                    kind="error",
                                    ticket=ticket,
                                    error="Ticket cancelled",
                                ),
                            )
                        continue
                    if entry.get("lease") is not None:
                        continue
                    # Pool admission scans queued and active requests under a
                    # synchronous lock. Never monopolize the WebSocket loop:
                    # workers must receive offers and ACK before the next scan.
                    lease = await asyncio.to_thread(
                        self.pool.acquire, **entry["payload"]
                    )
                    if lease is not None:
                        entry["lease"] = lease
                        self._offer(
                            entry["owner"],
                            dict(kind="offer", ticket=ticket, lease=lease),
                        )
                except Exception as exc:
                    self._offer(
                        entry["owner"],
                        dict(kind="error", ticket=ticket, error=str(exc)),
                    )

    async def _connection(self, request):
        if request.query.get("epoch") != self.epoch:
            raise web.HTTPConflict(text="Controller epoch changed")
        owner = request.query.get("worker")
        if not owner:
            raise web.HTTPBadRequest(text="Worker identity required")
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        previous = self.connections.get(owner)
        self.connections[owner] = ws
        if previous is not None:
            await previous.close()
        try:
            async for message in ws:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = message.json()
                ticket = data.get("ticket")
                request_trace.emit(
                    "route_push_received", kind=data.get("kind"), ticket=ticket
                )
                try:
                    if self.connections.get(owner) is not ws:
                        raise RuntimeError("Connection superseded")
                    if data["kind"] == "register":
                        payload = data["payload"]
                        if ticket != payload.get("ticket"):
                            raise ValueError("Ticket mismatch")
                        if ticket in self.retired or ticket in self.pool.push_cancelled:
                            raise RuntimeError("Ticket already retired")
                        entry = self.records.get(ticket)
                        if entry is None:
                            entry = dict(
                                owner=owner, payload=payload, lease=None, acked=False
                            )
                            self.records[ticket] = entry
                        elif entry["owner"] != owner or entry["payload"] != payload:
                            raise RuntimeError("Ticket ownership or metadata changed")
                        if entry["lease"] is not None:
                            with self.pool.lock:
                                if self.pool.active.get(ticket) != entry["lease"]:
                                    raise RuntimeError("Lease already released")
                            await self._send(
                                owner,
                                dict(
                                    kind="ready" if entry["acked"] else "offer",
                                    ticket=ticket,
                                    lease=entry["lease"],
                                ),
                            )
                        self.changed.set()
                    elif data["kind"] == "ack":
                        entry = self.records[ticket]
                        if (
                            entry["owner"] != owner
                            or entry["lease"]["lease_id"] != data["lease_id"]
                        ):
                            raise RuntimeError("Lease acknowledgement mismatch")
                        with self.pool.lock:
                            current = self.pool.active.get(ticket)
                            if (
                                current != entry["lease"]
                                or ticket in self.pool.push_cancelled
                            ):
                                raise RuntimeError("Lease is no longer active")
                        entry["acked"] = True
                        await self._send(
                            owner,
                            dict(kind="ready", ticket=ticket, lease=entry["lease"]),
                        )
                    else:
                        raise ValueError("Unknown push message")
                except Exception as exc:
                    await self._send(
                        owner, dict(kind="error", ticket=ticket, error=str(exc))
                    )
        finally:
            if self.connections.get(owner) is ws:
                self.connections.pop(owner, None)
        return ws


class PushRouteClient:
    def __init__(self, session, url):
        self.session = session
        self.url = url
        self.owner = uuid.uuid4().hex
        self.pending = {}
        self.ws = None
        self.task = None
        self.error = None

    async def acquire(self, payload):
        ticket = payload["ticket"]
        if self.error:
            raise self.error
        if ticket in self.pending:
            raise RuntimeError("Concurrent duplicate ticket on worker")
        future = asyncio.get_running_loop().create_future()
        self.pending[ticket] = (payload, future)
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())
        elif self.ws is not None and not self.ws.closed:
            try:
                await self._register(ticket, payload)
            except (aiohttp.ClientError, ConnectionError):
                # The reader reconnects and replays pending registrations.
                pass
        try:
            return await future
        finally:
            self.pending.pop(ticket, None)

    async def _register(self, ticket, payload):
        await self.ws.send_json(dict(kind="register", ticket=ticket, payload=payload))

    async def _run(self):
        outage = None
        try:
            while not self.session.closed:
                try:
                    outage = outage or time.monotonic()
                    remaining = 60 - (time.monotonic() - outage)
                    if remaining <= 0:
                        raise RuntimeError("Push route reconnect deadline exceeded")
                    connection = await asyncio.wait_for(
                        self.session.ws_connect(
                            self.url,
                            params={"worker": self.owner},
                            heartbeat=10,
                            timeout=aiohttp.ClientWSTimeout(ws_close=3),
                        ),
                        min(5, remaining),
                    )
                    async with connection as ws:
                        self.ws = ws
                        for ticket, (payload, _) in list(self.pending.items()):
                            await self._register(ticket, payload)
                        async for message in ws:
                            if message.type != aiohttp.WSMsgType.TEXT:
                                continue
                            data = message.json()
                            outage = None
                            pair = self.pending.get(data.get("ticket"))
                            if pair is None or pair[1].done():
                                continue
                            future = pair[1]
                            if data["kind"] == "offer":
                                await ws.send_json(
                                    dict(
                                        kind="ack",
                                        ticket=data["ticket"],
                                        lease_id=data["lease"]["lease_id"],
                                    )
                                )
                            elif data["kind"] == "ready":
                                future.set_result(data["lease"])
                            elif data["kind"] == "error":
                                future.set_exception(RuntimeError(data["error"]))
                        raise ConnectionError("Push route stream closed")
                except aiohttp.WSServerHandshakeError as exc:
                    if exc.status in (400, 403, 404, 409):
                        raise RuntimeError(
                            "Push controller identity or endpoint changed"
                        ) from exc
                    raise
                except (aiohttp.ClientError, ConnectionError, TimeoutError):
                    self.ws = None
                    if self.session.closed:
                        break
                    outage = outage or time.monotonic()
                    if time.monotonic() - outage >= 60:
                        raise TimeoutError("Push route reconnect deadline exceeded")
                    request_trace.emit(
                        "route_push_reconnect", pending=len(self.pending)
                    )
                    await asyncio.sleep(0.1)
            raise RuntimeError("Push route session closed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = exc
            for _, future in self.pending.values():
                if not future.done():
                    future.set_exception(exc)
        finally:
            self.ws = None

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


async def push_acquire(session, address, mode, **payload):
    from areal.infra.request_router import route_call

    clients = getattr(session, "_areal_push_clients", None)
    if clients is None:
        clients = {}
        session._areal_push_clients = clients
    if address not in clients:
        # Serialize discovery to create exactly one stream per worker/session.
        clients[address] = asyncio.create_task(
            route_call(session, address, "push_endpoint")
        )
    client = clients[address]
    if isinstance(client, asyncio.Task):
        url = await asyncio.shield(client)
        if url is None:
            if mode == "push_or_poll":
                request_trace.emit("route_push_fallback", reason="server_disabled")
                return None
            raise RuntimeError("Push routing is not enabled on controller")
        # No await between checking and replacing: concurrent waiters reuse it.
        if isinstance(clients[address], asyncio.Task):
            clients[address] = PushRouteClient(session, url)
        client = clients[address]
    return await client.acquire(payload)
