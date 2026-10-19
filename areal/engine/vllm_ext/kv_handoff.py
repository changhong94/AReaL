# SPDX-License-Identifier: Apache-2.0
"""Experimental TP=1 request-local KV staging and native scheduler import.

Transport tickets never authorize execution. Source abort and the normal routed
continuation remain the sole ownership handoff. Unsupported requests fail closed.
"""

import hashlib
import os
import time
from functools import wraps


def enabled():
    return os.getenv("AREAL_RUNNING_REBALANCE", "off") == "kv"


def prefix_key(tokens):
    return hashlib.sha256(
        b"".join(int(t).to_bytes(4, "little", signed=False) for t in tokens)
    ).hexdigest()


def validate_sampling(params):
    if (
        params.n != 1
        or params.presence_penalty != 0
        or params.frequency_penalty != 0
        or params.repetition_penalty != 1
        or params.min_tokens != 0
        or params.seed is not None
        or params.stop
        or params.prompt_logprobs is not None
    ):
        raise ValueError("KV handoff requires unseeded, penalty-free token generation")


def worker_operation(worker, action, data):
    import torch
    import torch.distributed as dist

    runner = worker.model_runner
    if action == "discard":
        getattr(worker, "_areal_kv_staging", {}).pop(data["ticket"], None)
        return {}
    pool = getattr(runner, "_areal_dynamic_kv", None)
    if not enabled() or pool is None or pool.broken or pool.paused or pool.swapped:
        raise RuntimeError("KV handoff requires healthy resident dynamic KV")
    if runner.vllm_config.parallel_config.tensor_parallel_size != 1:
        raise ValueError("KV handoff currently requires TP=1")
    staging = getattr(worker, "_areal_kv_staging", {})
    worker._areal_kv_staging = staging
    ticket = data.get("ticket")
    geometry = dict(
        tensors=len(pool.tensors),
        block_bytes=pool.bytes_per_block,
        block_tokens=runner.vllm_config.cache_config.block_size,
    )
    if action == "info":
        return geometry
    if action == "init":
        from datetime import timedelta

        from areal.engine.core.distributed import init_custom_process_group

        worker._areal_kv_group = init_custom_process_group(
            backend="hccl",
            world_size=2,
            rank=data["rank"],
            init_method=data["init_method"],
            group_name=data["group_name"],
            timeout=timedelta(seconds=20),
        )
        warm = torch.zeros(1, dtype=torch.float16, device=runner.device)
        options = dist.BroadcastOptions()
        options.rootRank = 0
        options.rootTensor = 0
        worker._areal_kv_group.broadcast([warm], options).wait()
        torch.npu.synchronize()
        return geometry
    if action == "discard":
        staging.pop(ticket, None)
        return {}
    if action == "pack":
        operation_started = time.perf_counter()
        if ticket in staging:
            raise RuntimeError("Duplicate staging ticket")
        blocks = data["blocks"]
        if not blocks or not set(blocks).issubset(pool.live):
            raise ValueError("Source KV blocks are not resident")
        size = len(blocks) * pool.bytes_per_block * len(pool.tensors)
        if (
            size > 64 * 1024**2
            or torch.npu.mem_get_info()[0] < 2 * size + pool.reserve_bytes
        ):
            raise RuntimeError("Insufficient staging headroom")
        torch.npu.synchronize()
        started = time.perf_counter()
        indices = torch.tensor(blocks, dtype=torch.int64, device=runner.device)
        payload = torch.cat(
            [
                t.view(torch.uint8)
                .reshape(pool.num_blocks, pool.bytes_per_block)
                .index_select(0, indices)
                .reshape(-1)
                for t in pool.tensors
            ]
        )
        torch.npu.synchronize()
        staging[ticket] = payload
        return dict(
            geometry,
            bytes=size,
            pack_s=time.perf_counter() - started,
            worker_s=time.perf_counter() - operation_started,
        )
    if action == "transfer":
        operation_started = time.perf_counter()
        size = data["bytes"]
        if geometry != data["geometry"] or size <= 0 or size > 64 * 1024**2 or size % 2:
            raise ValueError("Incompatible KV payload geometry")
        if data["source"]:
            payload = staging[ticket]
        else:
            if (
                ticket in staging
                or torch.npu.mem_get_info()[0] < 2 * size + pool.reserve_bytes
            ):
                raise RuntimeError("Insufficient target staging headroom")
            payload = torch.empty(size, dtype=torch.uint8, device=runner.device)
            staging[ticket] = payload
        started = time.perf_counter()
        # Direct group API uses the dedicated group's rank namespace.
        opts = dist.BroadcastOptions()
        opts.rootRank = 0
        opts.rootTensor = 0
        worker._areal_kv_group.broadcast([payload.view(torch.float16)], opts).wait()
        torch.npu.synchronize()
        return dict(
            transfer_s=time.perf_counter() - started,
            worker_s=time.perf_counter() - operation_started,
        )
    if action == "import":
        operation_started = time.perf_counter()
        payload = staging[ticket]
        blocks = data["blocks"]
        size = len(blocks) * pool.bytes_per_block * len(pool.tensors)
        if payload.numel() != size:
            raise ValueError("Target KV block count does not match staged payload")
        pins = getattr(runner, "_areal_kv_import_pins", {})
        runner._areal_kv_import_pins = pins
        pins[data["request_id"]] = set(blocks)
        pool.reconcile(pool.live | set(blocks))
        indices = torch.tensor(blocks, dtype=torch.int64, device=runner.device)
        offset = 0
        for tensor in pool.tensors:
            n = len(blocks) * pool.bytes_per_block
            tensor.view(torch.uint8).reshape(
                pool.num_blocks, pool.bytes_per_block
            ).index_copy_(
                0,
                indices,
                payload[offset : offset + n].reshape(len(blocks), pool.bytes_per_block),
            )
            offset += n
        torch.npu.synchronize()
        del staging[ticket]
        return dict(import_s=time.perf_counter() - operation_started)
    raise ValueError("Unknown KV worker action")


def install_freeze(cls):
    original = cls.schedule
    if getattr(original, "_areal_kv_freeze", False):
        return

    @wraps(original)
    def schedule(self, *args, **kwargs):
        frozen = getattr(self, "_areal_kv_frozen", {})
        held = [r for r in self.running if r.request_id in frozen]
        self.running[:] = [r for r in self.running if r.request_id not in frozen]
        width = self.max_num_running_reqs
        self.max_num_running_reqs = max(0, width - len(held))
        try:
            return original(self, *args, **kwargs)
        finally:
            self.max_num_running_reqs = width
            self.running.extend(held)

    schedule._areal_kv_freeze = True
    cls.schedule = schedule


def install():
    if not enabled():
        return
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore

    if getattr(Scheduler, "_areal_kv_handoff", False):
        return
    Scheduler._areal_kv_handoff = True
    original_add = EngineCore.add_request
    install_freeze(Scheduler)

    def operate(self, action, data):
        scheduler = self.scheduler
        # Ascend replaces the scheduler class after entry-module import. Patch
        # the actual runtime class, not only the upstream base alias.
        install_freeze(type(scheduler))
        frozen = getattr(scheduler, "_areal_kv_frozen", {})
        scheduler._areal_kv_frozen = frozen
        imports = getattr(self, "_areal_kv_imports", {})
        self._areal_kv_imports = imports
        if action == "import_status":
            return dict(results=getattr(self, "_areal_kv_import_results", {}))
        if action == "export":
            request = scheduler.requests.get(data["request_id"])
            if request is None or request not in scheduler.running:
                return dict(skipped="request_finished")
            validate_sampling(request.sampling_params)
            previous = frozen.get(request.request_id)
            if previous is not None and previous != data["ticket"]:
                raise RuntimeError("Request already frozen by another ticket")
            frozen[request.request_id] = data["ticket"]
            computed = request.num_computed_tokens
            tokens = list(request.all_token_ids)
            if (
                getattr(request, "num_in_flight_tokens", 0)
                or getattr(request, "num_output_placeholders", 0)
                or computed <= 0
                or computed >= len(tokens)
            ):
                return dict(
                    skipped="unstable_computed_tokens",
                    computed=computed,
                    tokens=len(tokens),
                    inflight=getattr(request, "num_in_flight_tokens", 0),
                    placeholders=getattr(request, "num_output_placeholders", 0),
                )
            blocks = scheduler.kv_cache_manager.get_block_ids(request.request_id)
            if len(blocks) != 1:
                raise ValueError("KV handoff requires one attention group")
            blocks = list(blocks[0])[
                : (computed + scheduler.block_size - 1) // scheduler.block_size
            ]
            frozen[request.request_id] = data["ticket"]
            try:
                result = self.model_executor.collective_rpc(
                    "areal_kv_handoff", args=("pack", dict(data, blocks=blocks))
                )[0]
            except Exception:
                frozen.pop(request.request_id, None)
                raise
            return dict(
                result,
                computed=computed,
                key=prefix_key(tokens),
                blocks=len(blocks),
                tokens=len(tokens),
            )
        if action == "unfreeze":
            frozen.pop(data["request_id"], None)
            return {}
        if action == "register":
            if data["ticket"] in imports:
                raise RuntimeError("Duplicate KV prefix ticket")
            imports[data["ticket"]] = data
            return {}
        if action == "discard":
            self._areal_kv_imports = {
                k: v for k, v in imports.items() if v["ticket"] != data["ticket"]
            }
        return self.model_executor.collective_rpc(
            "areal_kv_handoff", args=(action, data)
        )[0]

    @wraps(original_add)
    def add(self, request, *args, **kwargs):
        imports = getattr(self, "_areal_kv_imports", {})
        tag = (request.sampling_params.extra_args or {}).get("areal_kv_ticket")
        ticket = imports.pop(tag, None)
        if tag and ticket is None:
            raise RuntimeError("Missing or already consumed KV ticket")
        if ticket and prefix_key(request.prompt_token_ids) != ticket["key"]:
            raise RuntimeError("KV continuation prefix mismatch")
        import_s = 0.0
        if ticket is not None:
            if getattr(self.scheduler, "needs_kv_cache_zeroing", False):
                raise ValueError("KV handoff does not support worker block zeroing")
            outcome = "replay_fallback"
            if not (0 < ticket["computed"] < len(request.prompt_token_ids)):
                raise ValueError("Invalid imported computed prefix")
            if (
                ticket["blocks"]
                != (ticket["computed"] + self.scheduler.block_size - 1)
                // self.scheduler.block_size
            ):
                raise ValueError("Invalid imported block count")
            validate_sampling(request.sampling_params)
            if request.abort_immediately:
                self.model_executor.collective_rpc(
                    "areal_kv_handoff", args=("discard", ticket)
                )
            else:
                manager = self.scheduler.kv_cache_manager
                allocation = manager.allocate_slots(
                    request,
                    num_new_tokens=1,
                    num_external_computed_tokens=ticket["computed"],
                    delay_cache_blocks=True,
                )
                if allocation is None:
                    self.model_executor.collective_rpc(
                        "areal_kv_handoff", args=("discard", ticket)
                    )
                else:
                    blocks = manager.get_block_ids(request.request_id)
                    import_started = time.perf_counter()
                    self.model_executor.collective_rpc(
                        "areal_kv_handoff",
                        args=(
                            "import",
                            dict(
                                ticket,
                                request_id=request.request_id,
                                blocks=list(blocks[0])[: ticket["blocks"]],
                            ),
                        ),
                    )
                    import_s = time.perf_counter() - import_started
                    request.num_computed_tokens = ticket["computed"]
                    outcome = "import_applied"
        result = original_add(self, request, *args, **kwargs)
        if ticket is not None:
            from areal.utils import request_trace

            reports = getattr(self, "_areal_kv_import_results", {})
            self._areal_kv_import_results = reports
            reports[ticket["ticket"]] = dict(
                outcome=outcome,
                import_s=import_s,
                computed=request.num_computed_tokens,
                request_id=request.request_id,
            )
            while len(reports) > 128:
                del reports[next(iter(reports))]
            request_trace.emit(
                "kv_import_result", ticket=ticket["ticket"], **reports[ticket["ticket"]]
            )
        return result

    EngineCore.add_request = add
    EngineCore.areal_kv_handoff = operate
