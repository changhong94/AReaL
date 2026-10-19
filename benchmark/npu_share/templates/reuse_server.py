"""Standalone benchmark-owned AReaL server with explicit idle reset operations."""

import os
from pathlib import Path

import uvloop
import vllm.entrypoints.openai.api_server as native
from fastapi import APIRouter, HTTPException, Request
from reuse_core import install_core_reset

from areal.engine.vllm_ext import areal_vllm_server as base
from areal.utils import generation_progress

install_core_reset()
router = APIRouter()
parked = False
captured = False


@router.post("/benchmark/session")
async def session(request: Request):
    global parked, captured
    payload = await request.json()
    action = payload["action"]
    llm = request.app.state.engine_client
    async with base._memory_transition_lock:
        if base._trace_active_requests:
            raise HTTPException(409, "Active HTTP generation prevents reset")
        base._generation_run_event.clear()
        if not await llm.is_paused():
            await llm.pause_generation(
                wait_for_inflight_requests=False, clear_cache=True
            )
        trace_dir = payload.get("trace_dir", "/session/service-events")
        resolved = Path(trace_dir).resolve()
        if Path("/session") not in resolved.parents:
            raise HTTPException(400, "Trace destination must be inside this session")
        idle = await llm.engine_core.call_utility_async(
            "benchmark_idle", dict(trace_dir=trace_dir)
        )
        adaptive = os.getenv("AREAL_AUTO_MEMORY") == "1"
        if action == "capture":
            if captured:
                raise HTTPException(409, "Snapshot already captured")
            result = await llm.collective_rpc("benchmark_session", args=("capture", {}))
            captured = True
        elif action == "park":
            if not parked:
                if adaptive:
                    result = await llm.collective_rpc(
                        "areal_colocation_memory", args=("offload", True)
                    )
                else:
                    await llm.sleep(level=1)
                    result = await llm.collective_rpc(
                        "benchmark_session", args=("status", {})
                    )
                parked = True
            else:
                result = await llm.collective_rpc(
                    "benchmark_session", args=("status", {})
                )
        elif action == "reset":
            if not captured:
                raise HTTPException(409, "No initial snapshot")
            if parked:
                if adaptive:
                    await llm.collective_rpc(
                        "areal_colocation_memory", args=("restore", True)
                    )
                else:
                    await llm.wake_up()
                parked = False
            os.environ["AREAL_REQUEST_TRACE_DIR"] = trace_dir
            result = await llm.collective_rpc(
                "benchmark_session", args=("reset", payload)
            )
            await llm.reset_prefix_cache()
            generation_progress.registry = generation_progress.ProgressRegistry()
            base._kv_budget_valid = False
            base._kv_external_tickets.clear()
            base._trace_paused_requests.clear()
            getattr(llm.output_processor, "_areal_token_observations", {}).clear()
            if adaptive:
                await llm.collective_rpc(
                    "areal_colocation_memory", args=("kv_budget_clear", True)
                )
            await llm.resume_generation()
            base._generation_run_event.set()
        else:
            raise HTTPException(400, "Unknown session action")
        return dict(
            action=action,
            parked=parked,
            idle=idle,
            workers=result,
            api_pid=os.getpid(),
            snapshot_separate_from_mutable_backup=True,
        )


def build_app(*args, **kwargs):
    app = base._original_build_app(*args, **kwargs)
    replace = {"/v1/completions", "/inference/v1/generate"}
    app.router.routes = [
        r
        for r in app.router.routes
        if not (
            getattr(r, "path", None) in replace
            and "POST" in getattr(r, "methods", set())
        )
    ]
    app.include_router(base.router)
    app.include_router(router)
    return app


if __name__ == "__main__":
    native.build_app = build_app
    base.cli_env_setup()
    parser = base.make_arg_parser(base.FlexibleArgumentParser())
    args = parser.parse_args()
    base.validate_parsed_serve_args(args)
    uvloop.run(base.run_server(args))
