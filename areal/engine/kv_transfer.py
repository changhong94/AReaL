# SPDX-License-Identifier: Apache-2.0
"""Experimental packed KV transport; scheduler ownership remains caller-owned.

Call only after freezing the request and synchronizing its producer. Both peers
must validate identical model version, layout and tensor metadata out of band.
This primitive does not transfer vLLM scheduler state or grant request ownership.
"""

import time


def transfer_kv(tensors, *, source, group=None):
    """Broadcast a homogeneous KV payload and report synchronized stage costs."""
    import torch
    import torch.distributed as dist

    if not tensors:
        raise ValueError("Empty KV payload")
    first = tensors[0]
    if any(t.dtype != first.dtype or t.device != first.device for t in tensors):
        raise ValueError("KV payload must have one dtype and device")
    if first.device.type != "npu":
        raise ValueError("This experimental transport requires NPU tensors")
    torch.npu.synchronize()
    start = time.perf_counter()
    sender = dist.get_rank() == source
    count = sum(t.numel() for t in tensors)
    payload = (
        torch.cat([t.reshape(-1) for t in tensors])
        if sender
        else torch.empty(count, device=first.device, dtype=first.dtype)
    )
    torch.npu.synchronize()
    packed = time.perf_counter()
    # The caller establishes a dedicated HCCL group and identical call order.
    dist.broadcast(payload, src=source, group=group)
    torch.npu.synchronize()
    received = time.perf_counter()
    if not sender:
        offset = 0
        for tensor in tensors:
            size = tensor.numel()
            tensor.copy_(payload[offset : offset + size].view(tensor.shape))
            offset += size
    torch.npu.synchronize()
    end = time.perf_counter()
    return dict(
        bytes=count * first.element_size(),
        pack_s=packed - start,
        transfer_s=received - packed,
        unpack_s=end - received,
        total_s=end - start,
    )
