#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-rank real-model KV/HCCL probe, not a vLLM live request handoff.

Launch with torchrun --nproc_per_node=2. Source retains its cache. Target
reconstructs a fresh DynamicCache, then both continue from the same last sampled
but not yet computed token. No scheduler ownership claim is made by this probe.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, DynamicCache

from areal.engine.kv_transfer import transfer_kv


def cache_pairs(cache):
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def fresh_cache(config, pairs):
    try:
        cache = DynamicCache(config=config)
    except TypeError:
        cache = DynamicCache()
    for i, (key, value) in enumerate(pairs):
        cache.update(key, value, i)
    return cache


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lengths", default="64,256,1024")
    p.add_argument("--continuation", type=int, default=16)
    args = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 2:
        raise ValueError("The KV probe requires exactly two ranks")
    device = torch.device("npu", rank)
    torch.manual_seed(1)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        .eval()
        .to(device)
    )
    warm = torch.ones(1, device=device)
    dist.broadcast(warm, src=0)
    rows = []
    for length in [int(x) for x in args.lengths.split(",")]:
        if length < 40:
            raise ValueError("Probe context must contain at least 40 tokens")
        # Fixed prompt and 32 autoregressive steps create a real running cache.
        prefix = (torch.arange(length - 32, device=device) % 1000 + 100).view(1, -1)
        out = model(prefix, use_cache=True)
        cache = out.past_key_values
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(32):
            prefix = torch.cat((prefix, token), dim=1)
            out = model(token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            token = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.npu.synchronize()
        pairs = cache_pairs(cache)
        if cache.get_seq_length() != length:
            raise RuntimeError("Unexpected computed prefix length")
        # Both ranks computed baseline independently; validate every imported
        # element against that baseline, outside transport timing.
        expected = [t.cpu() for pair in pairs for t in pair]
        targets = (
            [t for pair in pairs for t in pair]
            if rank == 0
            else [torch.empty_like(t) for pair in pairs for t in pair]
        )
        measurements = []
        for _ in range(5):
            dist.barrier()
            measurements.append(transfer_kv(targets, source=0))
        exact = all(torch.equal(a.cpu(), b) for a, b in zip(targets, expected))
        if rank == 1:
            cache = fresh_cache(model.config, list(zip(targets[::2], targets[1::2])))
        torch.npu.synchronize()
        replay_start = time.perf_counter()
        replay = model(prefix, use_cache=True)
        torch.npu.synchronize()
        replay_s = time.perf_counter() - replay_start
        # Replay and transferred cache must predict the same pending token.
        replay_token = replay.logits[:, -1].argmax(-1, keepdim=True)
        replay_equal = bool(torch.equal(replay_token, token))
        generated = []
        started = time.perf_counter()
        for _ in range(args.continuation):
            generated.append(int(token.item()))
            out = model(token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            token = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.npu.synchronize()
        rows.append(
            dict(
                computed_tokens=length,
                payload_bytes=measurements[0]["bytes"],
                measurements=measurements,
                kv_exact=exact,
                replay_pending_token_equal=replay_equal,
                replay_s=replay_s,
                continuation_s=time.perf_counter() - started,
                generated=generated,
            )
        )
        print(
            json.dumps(
                dict(
                    rank=rank,
                    length=length,
                    kv_exact=exact,
                    payload_bytes=measurements[0]["bytes"],
                )
            ),
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank-{rank}.json").write_text(json.dumps(rows, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
