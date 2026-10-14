# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock

import pytest

from areal.engine.vllm_ext.generation_gate import (
    run_weight_update_while_paused,
    update_weights_while_paused,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("running", [True, False])
async def test_weight_load_preserves_external_pause(running):
    """Successful loading only restores a gate that the loader itself closed."""
    event = asyncio.Event()
    if running:
        event.set()
    engine = AsyncMock()
    engine.collective_rpc.return_value = [(True, "ok")]
    await update_weights_while_paused(engine, event, "/model")
    assert event.is_set() == running
    assert engine.resume_generation.await_count == int(running)
    engine.pause_generation.assert_awaited_once_with(
        wait_for_inflight_requests=False, clear_cache=True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "worker", "empty"])
async def test_failed_weight_load_never_resumes(failure):
    """Both RPC exceptions and per-worker failure results leave admission closed."""
    event = asyncio.Event()
    event.set()
    engine = AsyncMock()
    if failure == "exception":
        engine.collective_rpc.side_effect = RuntimeError("load failed")
        with pytest.raises(RuntimeError, match="load failed"):
            await update_weights_while_paused(engine, event, "/model")
    elif failure == "empty":
        engine.collective_rpc.return_value = []
        with pytest.raises(RuntimeError, match="no worker acknowledgements"):
            await update_weights_while_paused(engine, event, "/model")
    else:
        engine.collective_rpc.return_value = [(False, "failed")]
        await update_weights_while_paused(engine, event, "/model")
    assert not event.is_set()
    engine.resume_generation.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("running", [True, False])
@pytest.mark.parametrize("failure", [None, "worker", "exception", "empty"])
async def test_xccl_update_preserves_pause_and_never_resumes_failure(running, failure):
    """The HCCL path shares the disk path's external-pause and failed-load contract."""
    event = asyncio.Event()
    if running:
        event.set()
    engine = AsyncMock()
    before_load = AsyncMock()
    load = AsyncMock(return_value=[(True, "ok")])
    if failure == "worker":
        load.return_value = [(False, "failed")]
    elif failure == "empty":
        load.return_value = []
    elif failure == "exception":
        load.side_effect = RuntimeError("HCCL failed")
    if failure in {"empty", "exception"}:
        with pytest.raises(RuntimeError):
            await run_weight_update_while_paused(engine, event, load, before_load)
    else:
        await run_weight_update_while_paused(engine, event, load, before_load)
    before_load.assert_awaited_once()
    load.assert_awaited_once()
    assert event.is_set() == (running and failure is None)
    assert engine.resume_generation.await_count == int(running and failure is None)
