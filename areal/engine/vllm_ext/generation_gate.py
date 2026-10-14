# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


async def update_weights_while_paused(
    engine: Any,
    generation_event: asyncio.Event,
    model_path: str,
    before_load: Callable[[], Awaitable[None]] | None = None,
    snapshot_version: int | None = None,
) -> list[tuple[bool, str]]:
    """Load weights without releasing an external pause or a failed update."""
    args = (model_path,) if snapshot_version is None else (model_path, snapshot_version)
    return await run_weight_update_while_paused(
        engine,
        generation_event,
        lambda: engine.collective_rpc("areal_update_weights", args=args),
        before_load,
    )


async def run_weight_update_while_paused(
    engine: Any,
    generation_event: asyncio.Event,
    load: Callable[[], Awaitable[list[tuple[bool, str]]]],
    before_load: Callable[[], Awaitable[None]] | None = None,
) -> list[tuple[bool, str]]:
    """Keep both native and HTTP admission paused on failed or externally paused updates."""
    if getattr(engine, "_areal_preserve_pause", False) is True:
        raise RuntimeError(
            "Drain retained old-version requests before loading new weights"
        )
    was_running = generation_event.is_set()
    generation_event.clear()
    await engine.pause_generation(wait_for_inflight_requests=False, clear_cache=True)
    await engine.reset_mm_cache()
    if before_load is not None:
        await before_load()
    results = await load()
    if not results:
        raise RuntimeError("Weight update returned no worker acknowledgements")
    if all(success for success, _ in results) and was_running:
        await engine.resume_generation()
        generation_event.set()
    return results
