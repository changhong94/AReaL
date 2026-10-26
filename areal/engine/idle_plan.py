# SPDX-License-Identifier: Apache-2.0

"""Preserve a PPO microbatch plan across a DP1 to DP2 boundary."""

import copy
import json
from pathlib import Path

import torch

from areal.api.cli_args import MicroBatchSpec
from areal.utils.data import split_padded_tensor_dict_into_mb_list


def segment_bounds(answer_counts, target_answers):
    if target_answers < 2 or not answer_counts:
        raise ValueError("A nonempty plan and at least two answers are required")
    start, answers = 0, 0
    result = []
    for stop, count in enumerate(answer_counts, 1):
        answers += count
        if stop == len(answer_counts) or (answers >= target_answers and stop % 2 == 0):
            result.append((start, stop))
            start, answers = stop, 0
    return result


def select_microbatches(plan, start, stop, rank, world):
    """Keep every original packed item intact; pad collective rounds with zero loss."""
    if (
        world not in (1, 2)
        or not 0 <= rank < world
        or not 0 <= start < stop <= len(plan.mbs)
    ):
        raise ValueError("Invalid preserved-plan interval or DP rank")
    indices = list(range(start + rank, stop, world))
    rounds = (stop - start + world - 1) // world
    indices += [None] * (rounds - len(indices))
    out = copy.copy(plan)
    # Forward/backward consumes selected items, never the full source batch.
    out.data = {}
    for field in (
        "mbs",
        "padded_mbs",
        "group_lens",
        "padding_lengths",
        "padded_to_lengths",
        "old_cu_seqlens_list",
    ):
        values = getattr(plan, field)
        if values is not None:
            setattr(out, field, [values[start if i is None else i] for i in indices])
    for index, original in enumerate(indices):
        if original is None:
            for field in ("mbs", "padded_mbs"):
                values = getattr(out, field)
                if values is not None:
                    item = dict(values[index])
                    item["loss_mask"] = torch.zeros_like(item["loss_mask"])
                    values[index] = item
    out.forward_indices = out.backward_indices = None
    return out


def prepare_plan(engine, data, path, token):
    if (
        engine.world_size != 1
        or engine.parallel_helper.sp_size != 1
        or engine.enable_tree_training
    ):
        raise ValueError(
            "Preserved plans require initial DP1, SP1 and ordinary attention"
        )
    if engine.config.ppo_n_minibatches != 1 or not engine.config.disable_dropout:
        raise ValueError(
            "Preserved plans require one optimizer minibatch and disabled dropout"
        )
    root = Path(path)
    if not root.resolve().is_relative_to(Path("/dev/shm")):
        raise ValueError("Preserved plans must reside in shared DRAM")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    batch, _ = engine._normalize_batch_input(data)
    denominator = int(batch["loss_mask"].count_nonzero().item())
    if denominator <= 0:
        raise ValueError("Empty preserved-plan loss mask")
    # Match ordinary PPO's outer split before its token-budget microbatch packing.
    batch = {
        k: v
        for k, v in batch.items()
        if k not in ("rewards", "tot_rewards", "kl_rewards")
    }
    outer = split_padded_tensor_dict_into_mb_list(batch, MicroBatchSpec(n_mbs=1))
    assert len(outer.mbs) == 1
    plan = engine._prepare_mb_list(outer.mbs[0]).to("cpu")
    counts = [int(mb["cu_seqlens"].numel() - 1) for mb in plan.mbs]
    if sum(counts) != batch["input_ids"].shape[0]:
        raise RuntimeError("Preserved-plan answer coverage mismatch")
    meta = dict(
        token=token,
        version=engine.get_version(),
        denominator=denominator,
        microbatches=len(counts),
        answer_counts=counts,
        work_tokens=[int(mb["input_ids"].numel()) for mb in plan.mbs],
    )
    torch.save(plan, root / "plan.pt")
    (root / "meta.json").write_text(json.dumps(meta))
    engine._idle_plan_cache = (str(root), meta, plan)
    return meta


def load_plan(engine, path, token):
    cached = getattr(engine, "_idle_plan_cache", None)
    if cached is None or cached[0] != str(path):
        root = Path(path)
        if not root.resolve().is_relative_to(Path("/dev/shm")):
            raise ValueError("Preserved plans must reside in shared DRAM")
        meta = json.loads((root / "meta.json").read_text())
        plan = torch.load(root / "plan.pt", map_location="cpu", weights_only=False)
        cached = engine._idle_plan_cache = (str(path), meta, plan)
    _, meta, plan = cached
    if meta["token"] != token or meta["version"] != engine.get_version():
        raise RuntimeError("Preserved-plan generation mismatch")
    return meta, plan
