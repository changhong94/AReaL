# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock

import pytest

from areal.infra.generation_admission import GenerationCompletion


@pytest.mark.asyncio
async def test_group_waits_for_all_distinct_samples():
    callback = AsyncMock()
    progress = GenerationCompletion(3, callback)
    await progress.notify(2)
    await progress.notify(2)
    await progress.notify(0)
    callback.assert_not_awaited()
    await progress.notify(1)
    await progress.notify(0)
    callback.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_concurrent_duplicate_notifications_release_once():
    callback = AsyncMock()
    progress = GenerationCompletion(8, callback)
    await asyncio.gather(*(progress.notify(i % 8) for i in range(32)))
    callback.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_single_sample_accepts_ungrouped_context():
    callback = AsyncMock()
    progress = GenerationCompletion(1, callback)
    await progress.notify()
    await progress.notify(0)
    callback.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_callback_failure_propagates_without_duplicate_release():
    callback = AsyncMock(side_effect=RuntimeError("progress transport failed"))
    progress = GenerationCompletion(1, callback)
    with pytest.raises(RuntimeError, match="transport failed"):
        await progress.notify()
    await progress.notify()
    callback.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_invalid_sample_does_not_release_group():
    callback = AsyncMock()
    progress = GenerationCompletion(2, callback)
    for index in (None, -1, 2):
        with pytest.raises(ValueError):
            await progress.notify(index)
    callback.assert_not_awaited()


def test_empty_group_rejected():
    with pytest.raises(ValueError, match="positive"):
        GenerationCompletion(0, AsyncMock())
