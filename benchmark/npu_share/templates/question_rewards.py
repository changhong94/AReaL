"""Work-conserving, cross-process question-priority scoring (R029).

Only metadata uses SQLite. Code, tests and callbacks remain in their owning
rollout process. Numeric flock leases enforce the aggregate scoring limit.
"""

import asyncio
import fcntl
import os
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_REWARD_CONCURRENCY = 16


class _Admission:
    """One arm-local registry, operated outside the asyncio event loop."""

    def __init__(self, directory, owner):
        self.directory = directory
        self.owner = owner
        directory.mkdir(parents=True, exist_ok=True)
        self.life = (directory / (owner + ".owner")).open("a")
        fcntl.flock(self.life, fcntl.LOCK_EX)
        self.db = sqlite3.connect(directory / "question-priority.sqlite", timeout=30)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS owners (id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS groups (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE,
                owner TEXT, started INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY, group_id TEXT, owner TEXT,
                sample INTEGER, state TEXT);
            CREATE INDEX IF NOT EXISTS pending_requests ON requests(state);
        """)
        self.db.execute("INSERT INTO owners VALUES (?)", (owner,))
        self.db.commit()

    def tick(self, new, cancelled):
        leases = []
        self.db.execute("BEGIN IMMEDIATE")
        try:
            # Kernel locks survive task cancellation and disappear on process
            # death. Remove dead owners before choosing a global priority head.
            for (owner,) in self.db.execute("SELECT id FROM owners").fetchall():
                if owner == self.owner:
                    continue
                with (self.directory / (owner + ".owner")).open("a") as life:
                    try:
                        fcntl.flock(life, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    self.db.execute("DELETE FROM requests WHERE owner=?", (owner,))
                    self.db.execute("DELETE FROM groups WHERE owner=?", (owner,))
                    self.db.execute("DELETE FROM owners WHERE id=?", (owner,))
            for ticket, group, sample in new:
                self.db.execute(
                    "INSERT OR IGNORE INTO groups(id,owner) VALUES (?,?)",
                    (group, self.owner),
                )
                self.db.execute(
                    "INSERT OR IGNORE INTO requests VALUES (?,?,?,?, 'pending')",
                    (ticket, group, self.owner, sample),
                )
            self.db.executemany(
                "DELETE FROM requests WHERE id=? AND state='pending'", cancelled
            )
            for number in range(DEFAULT_REWARD_CONCURRENCY):
                stream = (self.directory / str(number)).open("a")
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                else:
                    leases.append(stream)
            selected = self.db.execute(
                """SELECT r.id,r.owner,r.group_id FROM requests r
                JOIN groups g ON g.id=r.group_id WHERE r.state='pending'
                ORDER BY g.started DESC,g.seq,r.sample LIMIT ?""",
                (len(leases),),
            ).fetchall()
            claimed = []
            for ticket, owner, group in selected:
                if owner != self.owner:
                    continue
                stream = leases.pop()
                claimed.append((ticket, stream))
                self.db.execute(
                    "UPDATE requests SET state='running' WHERE id=?", (ticket,)
                )
                self.db.execute("UPDATE groups SET started=1 WHERE id=?", (group,))
            self.db.commit()
            return claimed
        except BaseException:
            self.db.rollback()
            for _, stream in locals().get("claimed", []):
                stream.close()
            raise
        finally:
            for stream in leases:
                stream.close()

    def finish(self, ticket):
        self.db.execute("DELETE FROM requests WHERE id=?", (ticket,))
        self.db.commit()

    def close(self):
        self.db.execute("DELETE FROM requests WHERE owner=?", (self.owner,))
        self.db.execute("DELETE FROM groups WHERE owner=?", (self.owner,))
        self.db.execute("DELETE FROM owners WHERE id=?", (self.owner,))
        self.db.commit()
        self.db.close()
        self.life.close()


class QuestionRewardQueue:
    """Prefer started questions, but fill free slots with any ready answers.

    submit receives a callback that ALREADY owns its sandbox slot (grade_in_slot).
    Each answer starts without a complete-generation or complete-scoring barrier.
    The registry directory must be shared by all rollout workers in this arm.
    """

    def __init__(self):
        self.owner = uuid.uuid4().hex
        self.directory = Path(
            os.environ.get("AREAL_CODE_SLOT_DIR", "/run-output/sandbox-slots")
        )
        self.groups = {}
        self.pending = {}
        self.jobs = set()
        self.executor = ThreadPoolExecutor(
            max_workers=DEFAULT_REWARD_CONCURRENCY, thread_name_prefix="question-reward"
        )
        self.metadata = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="reward-admission"
        )
        self.admission = None
        self.dispatcher = None
        self.closed = False

    async def submit(self, key, sample_idx, group_size, call):
        if self.closed:
            raise RuntimeError("Reward queue is closed")
        if group_size <= 0 or not 0 <= sample_idx < group_size:
            raise ValueError("Invalid reward group sample index")
        group = self.groups.setdefault(
            key, {"id": uuid.uuid4().hex, "size": group_size, "seen": set()}
        )
        if group["size"] != group_size or sample_idx in group["seen"]:
            raise ValueError("Inconsistent or duplicate reward group sample")
        group["seen"].add(sample_idx)
        ticket = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        job = dict(
            group=group["id"],
            sample=sample_idx,
            future=future,
            call=call,
            registered=False,
            running=False,
        )
        self.pending[ticket] = job
        if self.dispatcher is None or self.dispatcher.done():
            self.dispatcher = asyncio.create_task(self._dispatch())
        return await future

    async def _meta(self, function, *args):
        future = asyncio.get_running_loop().run_in_executor(
            self.metadata, function, *args
        )
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            result = await future
            if isinstance(result, _Admission):
                await asyncio.get_running_loop().run_in_executor(
                    self.metadata, result.close
                )
            elif self.admission is not None and function == self.admission.tick:
                for _, stream in result:
                    stream.close()
            raise

    async def _dispatch(self):
        try:
            if self.admission is None:
                self.admission = await self._meta(
                    _Admission, self.directory, self.owner
                )
            while self.pending:
                new = []
                cancelled = []
                for ticket, job in list(self.pending.items()):
                    if job["running"]:
                        continue
                    if job["future"].cancelled():
                        cancelled.append((ticket,))
                    elif not job["registered"]:
                        new.append((ticket, job["group"], job["sample"]))
                        job["registered"] = True
                claimed = await self._meta(self.admission.tick, new, cancelled)
                for (ticket,) in cancelled:
                    self.pending.pop(ticket, None)
                for ticket, stream in claimed:
                    job = self.pending[ticket]
                    job["running"] = True
                    task = asyncio.create_task(self._score(ticket, job, stream))
                    self.jobs.add(task)
                    task.add_done_callback(self.jobs.discard)
                await asyncio.sleep(0.01)
        except BaseException as exc:
            self.closed = True
            waiters = [job["future"] for job in self.pending.values()]
            # A live but failed dispatcher must not leave unreachable priority
            # heads. Drain callbacks before releasing its liveness lease.
            if self.jobs:
                await asyncio.gather(*self.jobs, return_exceptions=True)
            if self.admission is not None:
                admission, self.admission = self.admission, None
                try:
                    await self._meta(admission.close)
                finally:
                    admission.life.close()
            self.pending.clear()
            for future in waiters:
                if not future.done():
                    if isinstance(exc, asyncio.CancelledError):
                        future.cancel()
                    else:
                        future.set_exception(exc)

    async def _score(self, ticket, job, stream):
        def execute():
            try:
                return job["call"]()
            finally:
                stream.close()

        result = error = None
        try:
            if job["future"].cancelled():
                stream.close()
            else:
                execution = asyncio.get_running_loop().run_in_executor(
                    self.executor, execute
                )
                try:
                    result = await asyncio.shield(execution)
                except asyncio.CancelledError:
                    # Cancelling the asyncio task cannot stop a sandbox call.
                    # Drain it before unregistering or allowing queue shutdown.
                    await execution
                    job["future"].cancel()
        except Exception as exc:
            error = exc
        finally:
            try:
                await self._meta(self.admission.finish, ticket)
            except Exception as exc:
                error = exc
            self.pending.pop(ticket, None)
        # Publish only after cleanup: consumers may immediately cancel siblings
        # or clear exception traceback frames when handling a scoring failure.
        if not job["future"].done():
            if error is not None:
                job["future"].set_exception(error)
            else:
                job["future"].set_result(result)

    async def close(self):
        """Cancel queued work; drain started operations before releasing leases."""
        self.closed = True
        for job in self.pending.values():
            job["future"].cancel()
        if self.dispatcher is not None:
            await self.dispatcher
        if self.jobs:
            await asyncio.gather(*self.jobs)
        if self.admission is not None:
            await self._meta(self.admission.close)
        self.executor.shutdown(wait=True)
        self.metadata.shutdown(wait=True)
