# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import logging
from http import HTTPStatus

import uvloop
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.openai.api_server import build_app as _original_build_app
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.completion.api_router import (
    create_completion as _decorated_create_completion,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, OpenAIBaseModel
from vllm.entrypoints.scale_out.token_in_token_out.api_router import (
    generate as _decorated_generate,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.entrypoints.serve.utils.api_utils import (
    cli_env_setup,
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils.argparse_utils import FlexibleArgumentParser

from areal.engine.vllm_ext.generation_gate import (
    run_weight_update_while_paused,
    update_weights_while_paused,
)
from areal.engine.vllm_ext.kv_slab_adapter import (
    install_scheduler_slab,
    slab_blocks,
    transition_slab_cache,
)
from areal.engine.vllm_ext.queue_rebalance import install_queue_rebalance
from areal.engine.vllm_ext.request_admission import (
    RoutingPaused,
    install_admission_guard,
    routed_request,
)
from areal.utils import generation_progress, request_trace

# Spawned EngineCore processes import this entry module, but not worker extensions.
install_scheduler_slab()
install_queue_rebalance()
request_trace.install_scheduler_trace()


async def validate_slab_endpoint(request: Request):
    if slab_blocks() and request.url.path not in {
        "/areal_update_weights",
        "/areal_init_weights_update_group",
        "/areal_set_update_weight_meta",
        "/areal_update_weights_xccl",
        "/areal_pause_generation",
        "/areal_pause_generation_preserve",
        "/areal_continue_generation",
        "/areal_colocation_memory",
        "/areal_generation_progress",
        "/areal_queue_rebalance",
        "/v1/completions",
        "/inference/v1/generate",
    }:
        raise HTTPException(
            status_code=400, detail="Endpoint unsupported in single-NPU KV slab mode"
        )


# AReaL's own router for custom endpoints.
router = APIRouter(dependencies=[Depends(validate_slab_endpoint)])
# Install in the worker extension after Ascend initializes its native imports.
# Importing the model runner in the API process causes an Ascend import cycle.
_memory_transition_lock = asyncio.Lock()

# vLLM already decorates its handlers with @with_cancellation and
# @load_aware_call. AReaL's wrappers below re-apply both, so they must delegate
# to the *undecorated* handler. Applying with_cancellation twice starts two
# listen_for_disconnect tasks on one ASGI receive channel -- its own docstring
# calls that unsafe, since each consumes and discards messages the other needs,
# so a disconnect can be seen by the outer listener while the inner generation
# keeps running and the load counter stays elevated.
original_create_completion = inspect.unwrap(_decorated_create_completion)
request_trace.install_request_id_trace()

generation_progress.install()
_concurrent_budget_valid = False
_trace_active_requests = set()
_trace_paused_requests = set()

original_generate = inspect.unwrap(_decorated_generate)


logger = init_logger("areal_vllm_server")
logger.setLevel(logging.INFO)

# Global event to control generation resume/pause
_generation_run_event = asyncio.Event()
_generation_run_event.set()  # Initially not paused
install_admission_guard(_memory_transition_lock, _generation_run_event)


def _apply_ascend_patch_awex():
    try:
        from areal.engine.patch_awex import patch_awex

        patch_awex()
        print("patching awex success for vllm server process.", flush=True)
    except ImportError:
        print("Failed to import awex, skip patching awex.", flush=True)


_apply_ascend_patch_awex()


class UpdateWeightsRequest(OpenAIBaseModel):
    snapshot_version: int | None = None
    # The model path with the new weights
    model_path: str
    # The format to load the weights
    load_format: str | None = "auto"
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False


class UpdateWeightsRequestLora(OpenAIBaseModel):
    # The model path with the new weights of lora adaptor
    lora_model_path: str
    # The name of lora adaptor
    lora_name: str
    # The id of the lora adaptor in vllm
    lora_int_id: int
    # The name of the base model for lora adaptors
    base_model_name: str
    # The format to load the weights
    load_format: str | None = "auto"
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False


class UpdateGroupRequest(OpenAIBaseModel):
    master_address: str
    master_port: str
    rank_offset: int
    world_size: int
    backend: str
    group_name: str


class UpdateWeightsFromXcclRequest(OpenAIBaseModel):
    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]
    group_name: str


class UpdateWeightsFromXcclRequestLora(OpenAIBaseModel):
    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]
    lora_name: str
    lora_int_id: int
    lora_target_modules: list[str] | str
    lora_rank: int
    lora_alpha: int
    lora_bias: str
    base_model_name: str
    group_name: str


def to_json_response(success, message):
    content = {"success": success, "message": message}
    if success:
        return JSONResponse(content, status_code=200)
    else:
        return JSONResponse(content, status_code=400)


def build_response(ret_list):
    success = True
    message = ""
    for rank, ret_value in enumerate(ret_list):
        if_success, msg = ret_value
        success = success if if_success else False
        if if_success:
            message += f"TP rank: {rank} success\n"
        else:
            message += f"TP rank: {rank} failed. reason: {msg}\n"
    return to_json_response(success, message)


def _infer_runtime_lora_path(serving_models, lora_name: str, lora_int_id: int) -> str:
    existing = serving_models.lora_requests.get(lora_name)
    if existing is not None and getattr(existing, "lora_path", ""):
        return existing.lora_path
    for request in serving_models.lora_requests.values():
        if getattr(request, "lora_int_id", None) == lora_int_id and getattr(
            request, "lora_path", ""
        ):
            return request.lora_path
    # Runtime XCCL updates do not come with a filesystem path. Use a stable
    # synthetic path so vLLM can still construct a LoRARequest for routing.
    return f"xccl://{lora_name}"


def _register_runtime_lora_name(
    app,
    *,
    lora_name: str,
    lora_int_id: int,
    base_model_name: str | None,
) -> None:
    serving_models = getattr(app.state, "openai_serving_models", None)
    if serving_models is None:
        logger.warning(
            "openai_serving_models missing; skip runtime LoRA registration for %s",
            lora_name,
        )
        return

    requests = serving_models.lora_requests
    runtime_lora_path = _infer_runtime_lora_path(serving_models, lora_name, lora_int_id)

    # Keep at most one public name per adapter id so /v1/models and request
    # routing reflect the current versioned adapter name.
    for name, request in list(requests.items()):
        if getattr(request, "lora_int_id", None) == lora_int_id and name != lora_name:
            del requests[name]

    lora_request = LoRARequest(
        lora_name=lora_name,
        lora_int_id=lora_int_id,
        lora_path=runtime_lora_path,
    )
    if base_model_name is not None:
        lora_request.base_model_name = base_model_name
    # Keep previous versioned names routable for requests admitted before the
    # adapter was updated. Versions can be bounded once request draining is tracked.
    requests[lora_request.lora_name] = lora_request
    logger.info(
        "Registered runtime LoRA adapter alias '%s' for adapter id %s",
        lora_name,
        lora_int_id,
    )


@router.post("/areal_update_weights")
async def areal_update_weight(request: UpdateWeightsRequest, raw_request: Request):
    logger.info(f"API server starts areal_update_weight, {request.model_path}")
    llm = raw_request.app.state.engine_client
    async with _memory_transition_lock:
        ret_list = await update_weights_while_paused(
            llm,
            _generation_run_event,
            request.model_path,
            before_load=lambda: transition_slab_cache(llm, mapped=True),
            snapshot_version=request.snapshot_version,
        )
    return build_response(ret_list)


@router.post("/areal_update_weights_lora")
async def areal_update_weight_lora(
    request: UpdateWeightsRequestLora, raw_request: Request
):
    logger.info(
        f"API server starts areal_update_weight_lora, lora_model_path-{request.lora_model_path}, lora_name-{request.lora_name}, lora_int_id-{request.lora_int_id}, base_model_name-{request.base_model_name}"
    )
    llm = raw_request.app.state.engine_client
    await llm.pause_generation(wait_for_inflight_requests=False, clear_cache=True)
    await llm.reset_mm_cache()

    try:
        ret_list = await llm.collective_rpc(
            "areal_update_weights_lora",
            args=(
                request.lora_model_path,
                request.lora_name,
                request.lora_int_id,
                request.base_model_name,
            ),
        )
    finally:
        generation_progress.registry.break_service_interval()
        await llm.resume_generation()

    return build_response(ret_list)


@router.post("/areal_update_weights_xccl")
async def areal_update_weight_xccl(raw_request: Request):
    logger.info("API server starts areal_update_weight_xccl")
    llm = raw_request.app.state.engine_client
    async with _memory_transition_lock:
        ret_list = await run_weight_update_while_paused(
            llm,
            _generation_run_event,
            lambda: llm.collective_rpc("areal_update_weight_xccl"),
            before_load=lambda: transition_slab_cache(llm, mapped=True),
        )
    return build_response(ret_list)


@router.post("/areal_update_weights_lora_xccl")
async def areal_update_weight_lora_xccl(
    request: UpdateWeightsFromXcclRequestLora, raw_request: Request
):
    logger.info("API server starts areal_update_weight_lora_xccl")
    llm = raw_request.app.state.engine_client
    await llm.pause_generation(wait_for_inflight_requests=False, clear_cache=True)
    await llm.reset_mm_cache()

    try:
        ret_list = await llm.collective_rpc("areal_update_weight_lora_xccl")
        if all(success for success, _ in ret_list):
            _register_runtime_lora_name(
                raw_request.app,
                lora_name=request.lora_name,
                lora_int_id=request.lora_int_id,
                base_model_name=request.base_model_name,
            )
    finally:
        generation_progress.registry.break_service_interval()
        await llm.resume_generation()

    return build_response(ret_list)


@router.post("/areal_init_weights_update_group")
async def areal_init_weights_update_group(
    request: UpdateGroupRequest, raw_request: Request
):
    logger.info("API server starts areal_init_weights_update_group")
    llm = raw_request.app.state.engine_client
    ret_list = await llm.collective_rpc(
        "areal_init_update_weight_group",
        args=(
            request.master_address,
            request.master_port,
            request.rank_offset,
            request.world_size,
            request.backend,
            request.group_name,
        ),
    )
    return build_response(ret_list)


@router.post("/areal_set_update_weight_meta")
async def areal_set_weight_meta_xccl(
    request: UpdateWeightsFromXcclRequest, raw_request: Request
):
    logger.info("API server starts areal_set_update_weight_meta_xccl")
    llm = raw_request.app.state.engine_client
    ret_list = await llm.collective_rpc(
        "areal_set_weight_meta",
        args=(
            request.names,
            request.dtypes,
            request.shapes,
            request.group_name,
        ),
    )
    return build_response(ret_list)


@router.post("/areal_set_update_weight_meta_lora")
async def areal_set_weight_meta_xccl_lora(
    request: UpdateWeightsFromXcclRequestLora, raw_request: Request
):
    logger.info(
        f"API server starts areal_set_update_weight_meta_lora for {request.lora_name} with id {request.lora_int_id}"
    )
    llm = raw_request.app.state.engine_client
    ret_list = await llm.collective_rpc(
        "areal_set_weight_meta_lora",
        args=(
            request.names,
            request.dtypes,
            request.shapes,
            request.group_name,
            request.lora_name,
            request.lora_int_id,
            request.lora_target_modules,
            request.lora_rank,
            request.lora_alpha,
            request.lora_bias,
            request.base_model_name,
        ),
    )
    return build_response(ret_list)


@router.post("/areal_pause_generation_preserve")
async def areal_pause_generation_preserve(raw_request: Request):
    """Freeze the scheduler and retain request identities and live KV contents."""
    llm = raw_request.app.state.engine_client
    async with _memory_transition_lock:
        llm._areal_preserve_pause = True
        generation_progress.registry.break_service_interval()
        _generation_run_event.clear()
        _trace_paused_requests.update(_trace_active_requests)
        request_trace.emit(
            "native_pause_request",
            request_ids=sorted(_trace_active_requests),
            mode="keep",
        )
        await llm.pause_generation(mode="keep", clear_cache=False)
        # Worker synchronization fences the last decode before any memory copy.
        await llm.collective_rpc(
            "areal_colocation_memory", args=("pause_preserved", True)
        )
        request_trace.emit(
            "native_pause_ack", request_ids=sorted(_trace_paused_requests), mode="keep"
        )
    return to_json_response(True, "Generation paused with live KV preserved")


@router.post("/areal_pause_generation")
async def areal_pause_generation(raw_request: Request):
    logger.info("API server starts areal_pause_generation and aborts all requests")
    llm = raw_request.app.state.engine_client
    # Abort requests and confirm scheduler blocks are free before discarding KV.
    async with _memory_transition_lock:
        if getattr(llm, "_areal_preserve_pause", False):
            raise HTTPException(
                status_code=409,
                detail="Restore and drain retained requests before aborting",
            )
        if request_trace.enabled():
            _trace_paused_requests.update(_trace_active_requests)
            request_trace.emit(
                "native_pause_request", request_ids=sorted(_trace_active_requests)
            )
        generation_progress.registry.break_service_interval()
        _generation_run_event.clear()
        await llm.pause_generation(
            wait_for_inflight_requests=False,
            clear_cache=True,
        )
        await llm.reset_mm_cache()
        await transition_slab_cache(llm, mapped=False)

        request_trace.emit(
            "native_pause_ack", request_ids=sorted(_trace_paused_requests)
        )

    return to_json_response(True, "Generation paused and all requests aborted")


class ColocationMemoryRequest(OpenAIBaseModel):
    action: str = "status"
    target_free_bytes: int | None = None
    kv_budget_bytes: int | None = None
    member_rank: int | None = None


@router.post("/areal_generation_progress")
async def areal_generation_progress(raw_request: Request):
    body = await raw_request.json()
    result = generation_progress.registry.snapshot(rotate=body.get("rotate", False))
    result["budget_valid"] = _concurrent_budget_valid
    result["generation_running"] = _generation_run_event.is_set()
    return result


@router.post("/areal_queue_rebalance")
async def areal_queue_rebalance(raw_request: Request):
    import os

    if (
        os.getenv("AREAL_QUEUE_REBALANCE", "1") != "1"
        or os.getenv("AREAL_DRAIN_METRICS", "0") != "1"
    ):
        raise HTTPException(status_code=409, detail="Queue rebalance disabled")
    body = await raw_request.json()
    action = body.get("action", "snapshot")
    if action not in ("snapshot", "abort"):
        raise HTTPException(status_code=400, detail="Invalid queue operation")
    async with _memory_transition_lock:
        if not _generation_run_event.is_set():
            raise HTTPException(status_code=409, detail="Replica transitioning")
        llm = raw_request.app.state.engine_client
        # Keep the exact attempt IDs: a late claim must not cancel its retry.
        with generation_progress.registry.lock:
            mapping = {
                i: r["request_id"]
                for i, r in generation_progress.registry.active.items()
            }
        ids = [i for i in body.get("internal_ids", []) if i in mapping]
        result = await llm.engine_core.call_utility_async(
            "areal_queue_rebalance", action, ids
        )
        if action == "snapshot":
            return dict(
                queue=dict(
                    queued=[mapping[i] for i in result["queued"] if i in mapping],
                    waiting=[mapping[i] for i in result["waiting"] if i in mapping],
                    running=[mapping[i] for i in result["running"] if i in mapping],
                    width=result["width"],
                ),
                internal_ids={mapping[i]: i for i in result["queued"] if i in mapping},
                **generation_progress.registry.snapshot(),
            )
        request_trace.emit("queue_rebalance_source_ack", internal_ids=result["aborted"])
        return result


@router.post("/areal_colocation_memory")
async def areal_colocation_memory(
    request: ColocationMemoryRequest, raw_request: Request
):
    global _concurrent_budget_valid
    from areal.engine.colocation_memory import adaptive_memory_enabled

    if not adaptive_memory_enabled():
        raise HTTPException(status_code=400, detail="Adaptive memory is disabled")
    async with _memory_transition_lock:
        if _generation_run_event.is_set():
            raise HTTPException(
                status_code=409, detail="Pause generation before memory handoff"
            )
        llm = raw_request.app.state.engine_client
        if not await llm.is_paused():
            raise HTTPException(
                status_code=409, detail="Engine scheduler must remain paused"
            )
        stats = await llm.engine_core.call_utility_async("areal_kv_slab_stats")
        preserving = getattr(llm, "_areal_preserve_pause", False)
        if stats["used_blocks"] and not preserving:
            raise HTTPException(
                status_code=409, detail="Live KV blocks prevent memory handoff"
            )
        result = await llm.collective_rpc(
            "areal_colocation_memory",
            args=(
                request.action,
                True,
                request.target_free_bytes,
                request.kv_budget_bytes,
                request.member_rank,
            ),
        )
        if request.action in {"kv_budget", "kv_budget_clear"}:
            if not result:
                raise RuntimeError("Missing TP memory acknowledgements")
            cap = min(row["capacity_blocks"] for row in result)
            # One full-length request must be able to complete without deadlock.
            minimum = (
                llm.model_config.max_model_len
                + llm.vllm_config.cache_config.block_size
                - 1
            ) // llm.vllm_config.cache_config.block_size + 1
            admitted = cap >= minimum
            for row in result:
                row["concurrent_admitted"] = admitted
                row["effective_capacity_blocks"] = cap
            _concurrent_budget_valid = request.action == "kv_budget" and admitted
            if admitted:
                await llm.engine_core.call_utility_async("areal_set_kv_capacity", cap)
        return result


@router.post("/areal_continue_generation")
async def areal_continue_generation(raw_request: Request):
    logger.info("API server starts areal_continue_generation")
    llm = raw_request.app.state.engine_client
    async with _memory_transition_lock:
        await transition_slab_cache(llm, mapped=True)
        preserving = getattr(llm, "_areal_preserve_pause", False)
        if preserving:
            await llm.collective_rpc(
                "areal_colocation_memory", args=("resume_preserved", True)
            )
        generation_progress.registry.break_service_interval()
        await llm.resume_generation()
        llm._areal_preserve_pause = False
        _generation_run_event.set()
        request_trace.emit("native_resume", request_ids=sorted(_trace_paused_requests))
        _trace_paused_requests.clear()
    return to_json_response(True, "Generation continued")


async def _wait_if_paused():
    """Wait if generation is paused."""
    if not _generation_run_event.is_set():
        await _generation_run_event.wait()


@router.post(
    "/v1/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Wrapped completions endpoint that respects pause state."""

    await _wait_if_paused()

    # Will not use streaming response here.
    response = await original_create_completion(request, raw_request)

    return response


@router.post(
    "/inference/v1/generate",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def generate(request: GenerateRequest, raw_request: Request):
    """Wrapped token-in/token-out endpoint that respects pause state.

    Rollout generation runs here, so without this gate a request admitted
    during a weight update would be served by a model that is mid-update. The
    policy lives in AReaL rather than in the vLLM patches: same path, same
    schema, same handler underneath, so the patches stay deletable.
    """

    if raw_request.headers.get("X-Areal-Request-Routing") == "1":
        if request.stream:
            raise HTTPException(
                status_code=400, detail="Shared request routing requires stream=False"
            )
        token = routed_request.set(True)
        rid = request.request_id
        request_trace.emit(
            "server_arrival",
            request_id=rid,
            server_rank=int(raw_request.headers["X-Areal-Route-Rank"]),
        )
        _trace_active_requests.add(rid)
        try:
            if not _generation_run_event.is_set():
                if getattr(
                    raw_request.app.state.engine_client, "_areal_preserve_pause", False
                ):
                    request_trace.emit("server_gate_wait_start", request_id=rid)
                    await _generation_run_event.wait()
                    request_trace.emit("server_gate_wait_end", request_id=rid)
                else:
                    raise RoutingPaused("Replica paused")
            return await original_generate(request, raw_request)
        except RoutingPaused:
            return JSONResponse(status_code=409, content={"areal_route_rejected": True})
        finally:
            _trace_active_requests.discard(rid)
            generation_progress.registry.discard_external(rid)
            request_trace.emit("server_return", request_id=rid)
            routed_request.reset(token)
    if not request_trace.enabled():
        await _wait_if_paused()
        return await original_generate(request, raw_request)
    rid = request.request_id
    request_trace.emit("server_arrival", request_id=rid)
    _trace_active_requests.add(rid)
    try:
        waiting = not _generation_run_event.is_set()
        if waiting:
            request_trace.emit("server_gate_wait_start", request_id=rid)
        await _wait_if_paused()
        if waiting:
            request_trace.emit("server_gate_wait_end", request_id=rid)
        return await original_generate(request, raw_request)
    finally:
        _trace_active_requests.discard(rid)
        request_trace.emit("server_return", request_id=rid)


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI
    # entrypoints.
    import vllm.entrypoints.openai.api_server as _api_server_module

    # Generation routes AReaL replaces with pause-aware wrappers. Every route
    # that reaches the engine must wait on the pause event, or a weight update
    # can land underneath an admitted request.
    _OVERRIDDEN_POST_ROUTES = ("/v1/completions", "/inference/v1/generate")

    def _areal_build_app(*args, **kwargs):
        """Monkey-patched build_app: swap in AReaL's generation routes + custom
        endpoints. ``**kwargs`` forwards version-specific params (model_config was
        added in vLLM 0.19) so it works on both 0.18 and 0.19."""
        app = _original_build_app(*args, **kwargs)
        replaced = [
            route
            for route in app.router.routes
            if hasattr(route, "path")
            and route.path in _OVERRIDDEN_POST_ROUTES
            and hasattr(route, "methods")
            and "POST" in route.methods
        ]
        # A silently missing route would drop the pause gate, so fail loudly
        # instead of serving unpaused traffic.
        missing = set(_OVERRIDDEN_POST_ROUTES) - {r.path for r in replaced}
        if missing:
            raise RuntimeError(
                f"vLLM did not register the generation routes AReaL wraps: "
                f"{sorted(missing)}. Without the wrapper these routes bypass "
                f"AReaL's weight-update pause gate."
            )
        app.router.routes = [r for r in app.router.routes if r not in replaced]
        app.include_router(router)
        return app

    # Patch build_app so run_server uses our version
    _api_server_module.build_app = _areal_build_app

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    if getattr(args, "headless", False):
        from vllm.entrypoints.cli.serve import run_headless

        if getattr(args, "api_server_count", None) is None:
            args.api_server_count = 0
        run_headless(args)
    else:
        uvloop.run(run_server(args))
