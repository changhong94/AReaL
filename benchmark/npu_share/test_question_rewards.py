"""CPU regressions for work-conserving cross-process scoring admission."""

import asyncio
import importlib.util
import multiprocessing
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "question_rewards", Path(__file__).parent / "templates/question_rewards.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def process_scores(directory, active, peak, lock):
    os.environ["AREAL_CODE_SLOT_DIR"] = directory

    def score():
        with lock:
            active.value += 1
            peak.value = max(peak.value, active.value)
        time.sleep(0.04)
        with lock:
            active.value -= 1
        return 1

    async def run():
        queue = module.QuestionRewardQueue()
        try:
            result = await asyncio.gather(
                *(queue.submit("q", i, 12, score) for i in range(12))
            )
            assert result == [1] * 12
        finally:
            await queue.close()

    asyncio.run(run())


def dead_owner(directory):
    admission = module._Admission(Path(directory), "dead-owner")
    claims = admission.tick([("dead-ticket", "dead-group", 0)], [])
    assert len(claims) == 1
    # Simulate process death while a claimed slot and registry metadata exist.
    os._exit(0)


class QuestionRewardsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"AREAL_CODE_SLOT_DIR": self.temp.name})
        self.env.start()
        self.queue = module.QuestionRewardQueue()

    async def asyncTearDown(self):
        await self.queue.close()
        self.env.stop()
        self.temp.cleanup()

    async def test_partial_question_starts_without_waiting_for_siblings(self):
        self.assertEqual(
            await asyncio.wait_for(self.queue.submit("q", 0, 8, lambda: 7), 2), 7
        )

    async def test_slow_answer_does_not_block_other_question(self):
        release = threading.Event()
        started = threading.Event()

        def slow():
            started.set()
            release.wait(5)
            return "slow"

        slow_task = asyncio.create_task(self.queue.submit("q1", 0, 8, slow))
        try:
            await asyncio.wait_for(asyncio.to_thread(started.wait), 2)
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(self.queue.submit("q2", i, 3, lambda i=i: i) for i in range(3))
                ),
                2,
            )
            self.assertEqual(results, [0, 1, 2])
            self.assertFalse(slow_task.done())
        finally:
            release.set()
            await slow_task

    async def test_started_question_precedes_new_question(self):
        self.assertEqual(await self.queue.submit("q1", 0, 8, lambda: 0), 0)
        order = []

        def score(name):
            order.append(name)
            time.sleep(0.01)
            return name

        # Register Q2 first in this turn, but Q1 already has scoring progress.
        tasks = [
            asyncio.create_task(self.queue.submit("q2", i, 4, lambda: score("q2")))
            for i in range(4)
        ]
        tasks += [
            asyncio.create_task(self.queue.submit("q1", i, 8, lambda: score("q1")))
            for i in range(1, 8)
        ]
        await asyncio.gather(*tasks)
        self.assertEqual(order[:4], ["q1"] * 4)

    async def test_pending_cancel_never_executes_and_running_cancel_drains(self):
        release = threading.Event()
        count = []

        def slow():
            count.append(1)
            release.wait(5)
            return 1

        tasks = [
            asyncio.create_task(self.queue.submit("q1", i, 16, slow)) for i in range(16)
        ]
        while len(count) != 16:
            await asyncio.sleep(0.01)
        waiting = asyncio.create_task(
            self.queue.submit("q2", 0, 1, lambda: count.append(99))
        )
        await asyncio.sleep(0.03)
        waiting.cancel()
        tasks[0].cancel()
        await asyncio.sleep(0.03)
        # Cancellation of a waiter cannot make its running slot available.
        following = asyncio.create_task(
            self.queue.submit("q3", 0, 1, lambda: count.append(3))
        )
        await asyncio.sleep(0.05)
        self.assertEqual(count, [1] * 16)
        release.set()
        await asyncio.gather(*tasks, waiting, following, return_exceptions=True)
        self.assertNotIn(99, count)
        self.assertIn(3, count)

    async def test_failure_propagates_and_releases_capacity(self):
        def fail():
            raise RuntimeError("sandbox unavailable")

        with self.assertRaisesRegex(RuntimeError, "sandbox unavailable"):
            await self.queue.submit("q", 0, 2, fail)
        self.assertEqual(await self.queue.submit("q", 1, 2, lambda: 1), 1)

    async def test_duplicate_sample_is_rejected(self):
        await self.queue.submit("q", 0, 2, lambda: 1)
        with self.assertRaises(ValueError):
            await self.queue.submit("q", 0, 2, lambda: 1)

    async def test_dispatcher_failure_releases_owner_for_peer(self):
        original = module._Admission.tick
        calls = []

        def broken(admission, new, cancelled):
            claims = original(admission, new, cancelled)
            for _, stream in claims:
                stream.close()
            calls.append(1)
            raise RuntimeError("metadata failure")

        with patch.object(module._Admission, "tick", broken):
            with self.assertRaisesRegex(RuntimeError, "metadata failure"):
                await self.queue.submit("bad", 0, 1, lambda: 1)
            await self.queue.dispatcher
        peer = module.QuestionRewardQueue()
        try:
            self.assertEqual(
                await asyncio.wait_for(peer.submit("ok", 0, 1, lambda: 3), 2), 3
            )
        finally:
            await peer.close()
        self.assertTrue(calls)

    async def test_dispatcher_cancel_during_claim_releases_leases(self):
        original = module._Admission.tick
        claimed = threading.Event()
        release = threading.Event()

        def delayed(admission, new, cancelled):
            result = original(admission, new, cancelled)
            claimed.set()
            release.wait(3)
            return result

        with patch.object(module._Admission, "tick", delayed):
            task = asyncio.create_task(self.queue.submit("cancel", 0, 1, lambda: 1))
            await asyncio.wait_for(asyncio.to_thread(claimed.wait), 2)
            self.queue.dispatcher.cancel()
            release.set()
            await asyncio.gather(task, self.queue.dispatcher, return_exceptions=True)
        peer = module.QuestionRewardQueue()
        try:
            self.assertEqual(
                await asyncio.wait_for(peer.submit("ok", 0, 1, lambda: 3), 2), 3
            )
        finally:
            await peer.close()

    async def test_two_processes_share_exactly_sixteen_slots(self):
        ctx = multiprocessing.get_context("spawn")
        active, peak, lock = ctx.Value("i", 0), ctx.Value("i", 0), ctx.Lock()
        children = [
            ctx.Process(
                target=process_scores, args=(self.temp.name, active, peak, lock)
            )
            for _ in range(2)
        ]
        for child in children:
            child.start()
        for child in children:
            await asyncio.to_thread(child.join, 10)
            self.assertEqual(child.exitcode, 0)
        self.assertEqual(active.value, 0)
        self.assertEqual(peak.value, 16)

    async def test_dead_owner_cannot_block_healthy_requests(self):
        ctx = multiprocessing.get_context("spawn")
        child = ctx.Process(target=dead_owner, args=(self.temp.name,))
        child.start()
        await asyncio.to_thread(child.join, 5)
        self.assertEqual(child.exitcode, 0)
        self.assertEqual(
            await asyncio.wait_for(self.queue.submit("q", 0, 1, lambda: 9), 2), 9
        )


if __name__ == "__main__":
    unittest.main()
