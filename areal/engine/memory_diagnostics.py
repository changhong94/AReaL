# SPDX-License-Identifier: Apache-2.0

"""Opt-in allocator snapshots for diagnosing training memory; no cache mutation."""

import json
import os
import time
from collections import defaultdict
from pathlib import Path


def record_training_memory(stage, microbatch, backend=None, tensor=None):
    destination = os.getenv("AREAL_MEMORY_DIAGNOSTICS", "")
    if not destination:
        return
    try:
        import torch

        device = torch.npu.current_device()
        snapshot = torch.npu.memory._snapshot()
        groups = defaultdict(lambda: defaultdict(int))
        for segment in snapshot["segments"]:
            key = str(
                (
                    segment.get("segment_pool_id"),
                    segment.get("stream"),
                    segment.get("device"),
                )
            )
            group = groups[key]
            group["reserved"] += segment["total_size"]
            group["active"] += segment.get("active_size", 0)
            group["allocated"] += segment.get("allocated_size", 0)
            for block in segment.get("blocks", []):
                group[block["state"]] += block["size"]
        tags = defaultdict(int)
        if backend is not None:
            for allocation in list(backend.inventory().values()):
                tags[str(allocation.tag)] += allocation.handle[1]
        event = dict(
            time=time.time(),
            pid=os.getpid(),
            rank=os.getenv("RANK"),
            device=device,
            stage=stage,
            microbatch=microbatch,
            free_total=torch.npu.mem_get_info(),
            allocated=torch.npu.memory_allocated(),
            reserved=torch.npu.memory_reserved(),
            segments=groups,
            camem_inventory_bytes=tags,
        )
        if tensor is not None:
            event["tensor"] = dict(
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                bytes=tensor.numel() * tensor.element_size(),
            )
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"events-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(event, default=str) + "\n")
        (directory / f"snapshot-{os.getpid()}-{stage}.json").write_text(
            json.dumps(snapshot, default=str)
        )
    except Exception as error:
        # Diagnostic failures must not replace the original training exception.
        import sys

        print(f"Memory diagnostic failed: {error!r}", file=sys.stderr, flush=True)
