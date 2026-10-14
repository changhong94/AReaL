# SPDX-License-Identifier: Apache-2.0
"""Bounded preparation of an exclusive actor while the fixed replica generates."""

import time
from threading import Event

from areal.utils import request_trace


class TrainingRestorePrefetch:
    def __init__(self, executor, ready, prepare, fail, batch, on_wait=None):
        self.ready = ready
        self.prepare = prepare
        self.fail = fail
        self.batch = batch
        self.on_wait = on_wait
        self.cancelled = Event()
        self.forced = Event()
        self.future = executor.submit(self._run)

    def _run(self):
        try:
            started = time.monotonic()
            while not self.cancelled.is_set():
                if self.forced.is_set() or self.ready():
                    break
                self.cancelled.wait(0.01)
            if self.cancelled.is_set():
                return False
            if self.on_wait is not None:
                self.on_wait(time.monotonic() - started)
            request_trace.emit("train_prefetch_begin", batch=self.batch)
            self.prepare()
            request_trace.emit("train_prefetch_ready", batch=self.batch)
            return True
        except BaseException as exc:
            self.fail(exc)
            raise

    def cancel(self):
        # Once preparation starts it must finish before any worker is destroyed.
        self.cancelled.set()

    def join(self, force=False):
        if force:
            self.forced.set()
        return self.future.result()
