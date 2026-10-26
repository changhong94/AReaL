# SPDX-License-Identifier: Apache-2.0
"""Explicit terminal-generation notifications for single-turn workflows."""

import asyncio
from collections.abc import Awaitable, Callable


class GenerationCompletion:
    """Release one generation slot after every sample explicitly finishes.

    This coordinator belongs to one workflow task and one event loop. It must
    not be notified after intermediate turns of a multi-turn agent. Callback
    errors propagate, leaving final task cleanup responsible for admission.
    """

    def __init__(
        self, group_size: int, callback: Callable[[], Awaitable[None]]
    ) -> None:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        self.group_size = group_size
        self._callback = callback
        self._completed: set[int] = set()
        self._lock = asyncio.Lock()
        self._notified = False

    async def notify(self, sample_idx: int | None = None) -> None:
        if sample_idx is None:
            if self.group_size != 1:
                raise ValueError("Grouped completion requires sample_idx")
            sample_idx = 0
        if not 0 <= sample_idx < self.group_size:
            raise ValueError("sample_idx is outside the rollout group")
        async with self._lock:
            self._completed.add(sample_idx)
            if self._notified or len(self._completed) < self.group_size:
                return
            # Mark before awaiting so cancellation/retries cannot double-release.
            self._notified = True
            await self._callback()
