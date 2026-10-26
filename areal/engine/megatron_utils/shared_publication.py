# SPDX-License-Identifier: Apache-2.0
"""Single-rank dense Megatron publication for shared Ascend inference.

The engine must join this helper before training, offloading, or destroying model
storage. Converted tensors may alias model parameters; retaining their Python
references protects allocation lifetime, not concurrent mutation. The completed
snapshot owns immutable host bytes independently of the model.
"""

from __future__ import annotations

import torch

from areal.engine.weight_snapshot import IndependentSnapshot, write_snapshot
from areal.utils import request_trace


class SharedMegatronPublication:
    """Convert once and share stable HF tensors between HCCL and a CPU writer."""

    def __init__(self, engine):
        self.engine = engine
        self.snapshot = None

    def _validate(self, meta):
        engine = self.engine
        if (
            engine.config.use_lora
            or getattr(engine.config, "use_merged_lora", False)
            or engine.quantization_config
            or engine.tf_config.num_moe_experts
            or engine.bridge_cls != "mbridge"
        ):
            raise ValueError(
                "Shared Megatron publication requires a dense full-model pure-DP mbridge actor"
            )
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "context_parallel_size",
            "expert_parallel_size",
            "expert_tensor_parallel_size",
        ):
            if getattr(engine.parallel_strategy, name, 1) != 1:
                raise ValueError(
                    "Shared Megatron publication requires all model-parallel sizes to be one"
                )
        if meta.path is None or meta.version is None or meta.version < 0:
            raise ValueError(
                "Shared publication requires a path and nonnegative version"
            )

    def _parameters(self):
        # Import lazily so publication lifecycle tests do not require Megatron.
        from areal.engine.megatron_utils.megatron import (
            convert_to_hf,
            get_named_parameters,
        )

        engine = self.engine
        tensors = []
        names = set()
        with torch.no_grad():
            for name, param in get_named_parameters(engine.model, None):
                param, _ = engine._collect_param(name, param)
                for hf_name, tensor in convert_to_hf(
                    engine.tf_config,
                    engine.hf_config.model_type,
                    name,
                    param,
                    quantization_config=engine.quantization_config,
                    fp8_direct_convert=engine.fp8_direct_convert,
                    hf_config=engine.hf_config,
                ):
                    if not hf_name or hf_name in names:
                        raise ValueError("Converted HF parameter names must be unique")
                    names.add(hf_name)
                    # Broadcast requires contiguous storage. Keep every conversion
                    # allocation alive until both independent readers finish.
                    tensor = tensor.detach()
                    if tensor.is_floating_point():
                        tensor = tensor.to(engine.dtype)
                    tensors.append((hf_name, tensor.contiguous()))
        if not tensors:
            raise ValueError("Megatron conversion produced no HF weights")
        return tensors

    def wait(self, version=None):
        if self.snapshot is None:
            if version is not None:
                raise RuntimeError("No shared snapshot was started")
            return None
        if version is not None and self.snapshot.version != version:
            raise RuntimeError("Shared snapshot version mismatch")
        return self.snapshot.wait()

    def publish_fixed(self, meta):
        """Publish only to the connected fixed receiver; never pause all rollout."""
        self._validate(meta)
        if not meta.npu_share_fixed:
            raise ValueError("Fixed publication requires fixed-receiver metadata")
        limit = int(meta.weight_chunked_mem_mb * 1024 * 1024)
        if limit <= 0:
            raise ValueError("Weight bucket size must be positive")
        self.wait()
        engine = self.engine
        meta.nccl_master_address = engine.weight_update_master_addr
        meta.nccl_master_port = engine.weight_update_master_port
        meta.nccl_group_name = engine.weight_update_group_name
        parameters = self._parameters()
        self.snapshot = None
        if meta.npu_share_snapshot:
            self.snapshot = IndependentSnapshot(
                meta.path, meta.version, parameters, engine.dtype
            )
        bucket, size = [], 0
        try:
            for name, tensor in parameters:
                nbytes = tensor.numel() * tensor.element_size()
                if bucket and size + nbytes > limit:
                    engine._update_bucket_weights_from_distributed(meta, bucket)
                    bucket, size = [], 0
                bucket.append((name, tensor))
                size += nbytes
            if bucket:
                engine._update_bucket_weights_from_distributed(meta, bucket)
        except BaseException as error:
            if self.snapshot is not None:
                self.snapshot.abort(error)
            raise
        # Bucket calls join every HCCL handle and receiver ACK. Do not synchronize
        # the entire device here: the CPU snapshot runs on a separate stream.
        request_trace.emit(
            "fixed_hccl_complete",
            version=meta.version,
            snapshot_mode=(
                "megatron_independent_reader"
                if meta.npu_share_snapshot
                else "megatron_fixed_only"
            ),
            snapshot_submit_seconds=0.0,
        )

    def export(self, meta):
        """Write the initial staged snapshot before a fixed publication exists."""
        self._validate(meta)
        self.wait()
        parameters = self._parameters()
        return write_snapshot(
            meta.path,
            meta.version,
            parameters,
            self.engine.dtype,
            lambda tensor: tensor,
        )
