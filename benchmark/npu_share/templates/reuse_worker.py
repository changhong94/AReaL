"""Opt-in benchmark reset hooks; not part of the research scheduling policy."""

import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from initial_weights import InitialWeights
from reuse_core import install_core_reset
from reuse_paths import snapshot_path

from areal.engine.vllm_ext.vllm_worker_extension import VLLMWorkerExtension


class ReuseWorkerExtension(VLLMWorkerExtension):
    def areal_update_weights(self, model_path, snapshot_version=None):
        path = snapshot_path(model_path, os.environ["AREAL_REQUEST_TRACE_DIR"])
        return VLLMWorkerExtension.areal_update_weights(self, path, snapshot_version)

    def benchmark_session(self, action, payload):
        started = time.perf_counter()
        runner = self.model_runner
        if action == "capture":
            if hasattr(self, "_benchmark_initial"):
                raise RuntimeError("Initial weights already captured")
            torch.npu.synchronize()
            self._benchmark_initial = InitialWeights(runner.model)
            result = dict(
                initial_dram_bytes=self._benchmark_initial.bytes,
                initial_sha256=self._benchmark_initial.digest,
            )
        elif action == "reset":
            os.environ["AREAL_REQUEST_TRACE_DIR"] = payload["trace_dir"]
            if "packed_tp_fence" in payload:
                if not isinstance(payload["packed_tp_fence"], bool):
                    raise ValueError("packed_tp_fence must be boolean")
                from areal.engine.vllm_ext.dynamic_kv import set_tp_protocol

                fused = payload.get("fused_tp_fence", False)
                if not isinstance(fused, bool) or (
                    fused and not payload["packed_tp_fence"]
                ):
                    raise ValueError("Fused KV fences require packed TP consensus")
                voted = payload.get("voted_tp_fence", False)
                if not isinstance(voted, bool) or (voted and not fused):
                    raise ValueError("Residency voting requires fused TP consensus")
                protocol = (
                    "voted"
                    if voted
                    else "fused"
                    if fused
                    else "packed"
                    if payload["packed_tp_fence"]
                    else "object"
                )
                dynamic = getattr(runner, "_areal_dynamic_kv", None)
                if dynamic is not None:
                    set_tp_protocol(dynamic, protocol)
            if getattr(self, "_areal_kv_staging", {}):
                raise RuntimeError("Outstanding KV staging prevents reset")
            groups = list(getattr(self, "weight_update_groups", {}).values())
            group = getattr(self, "_areal_kv_group", None)
            if group is not None:
                groups.append(group)
            groups.extend(
                peer["group"] for peer in getattr(self, "_areal_kv_groups", {}).values()
            )
            for group in dict.fromkeys(groups):
                dist.destroy_process_group(group)
            self.weight_update_groups = {}
            for name in (
                "areal_weight_meta_names",
                "areal_weight_meta_dtypes",
                "areal_weight_meta_shapes",
                "areal_weight_meta_group_name",
            ):
                if hasattr(self, name):
                    delattr(self, name)
            if hasattr(self, "_areal_kv_group"):
                del self._areal_kv_group
            self._areal_kv_groups = {}
            self._areal_kv_staging = {}
            runner._areal_kv_import_pins = {}
            memory = getattr(runner, "_adaptive_memory", None)
            if memory is not None:
                memory.mark_dirty(("weights",))
            result = self._benchmark_initial.restore(runner.model)
            random.seed(payload["seed"])
            np.random.seed(payload["seed"])
            torch.manual_seed(payload["seed"])
            torch.npu.manual_seed_all(payload["seed"])
            torch.npu.synchronize()
        elif action == "status":
            result = dict(
                initial_dram_bytes=self._benchmark_initial.bytes,
                initial_sha256=self._benchmark_initial.digest,
            )
        else:
            raise ValueError("Unknown benchmark worker action")
        free, total = torch.npu.mem_get_info()
        result.update(
            seconds=time.perf_counter() - started,
            device_used_bytes=total - free,
            worker_pid=os.getpid(),
            tp_fence_protocol=getattr(
                getattr(runner, "_areal_dynamic_kv", None), "tp_protocol", "object"
            ),
            fused_tp_fence=getattr(
                getattr(runner, "_areal_dynamic_kv", None), "tp_protocol", "object"
            )
            in {"fused", "voted"},
            packed_tp_fence=getattr(
                getattr(runner, "_areal_dynamic_kv", None), "tp_protocol", "object"
            )
            in {"packed", "fused", "voted"},
        )
        memory = getattr(runner, "_adaptive_memory", None)
        if memory is not None:
            result["mutable_offload"] = memory.snapshot()
        from vllm_ascend.device_allocator.camem import CaMemAllocator

        allocator = CaMemAllocator.get_instance()
        result["mutable_dram_bytes"] = sum(
            data.cpu_backup_tensor.numel() * data.cpu_backup_tensor.element_size()
            for data in allocator.pointer_to_data.values()
            if data.cpu_backup_tensor is not None
        )
        return result


install_core_reset()
