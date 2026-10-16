# SPDX-License-Identifier: Apache-2.0
"""Budgeted, fail-closed handoffs; no retry of a partially executed train step."""

import json
import os

GIB = 1024**3


def single_status(value):
    rows = []

    def flatten(item):
        if isinstance(item, list):
            for child in item:
                flatten(child)
        elif isinstance(item, dict) and "free_bytes" in item:
            rows.append(item)
        else:
            raise RuntimeError("Expected per-device memory status")

    flatten(value)
    if not rows:
        raise RuntimeError("Missing memory status from worker group")
    if len(rows) == 1:
        return rows[0]
    # Conservative per-device bounds, never a sum of unrelated free memory.
    result = dict(rows[0])
    for key in set.intersection(*(set(row) for row in rows)):
        values = [row[key] for row in rows]
        if all(isinstance(v, bool) for v in values):
            result[key] = all(values)
        elif all(isinstance(v, (int, float)) for v in values):
            result[key] = (min if key in {"free_bytes", "capacity_blocks"} else max)(
                values
            )
    if all(row.get("dynamic_kv") for row in rows):
        result["dynamic_kv"] = dict(rows[0]["dynamic_kv"])
        result["dynamic_kv"]["mapped_bytes"] = min(
            row["dynamic_kv"]["mapped_bytes"] for row in rows
        )
    else:
        result.pop("dynamic_kv", None)
    result["devices"] = rows
    return result


class ColocationMemoryPolicy:
    def __init__(self, config, actor, rollout, log):
        self.config = config
        self.actor = actor
        self.rollout = rollout
        self.log = log

    def free(self, status):
        cap = self.config.device_budget_gib
        free = status["free_bytes"]
        if cap is not None:
            free = min(free, int(cap * GIB) - (status["total_bytes"] - free))
        return max(0, free)

    def required(self, workspace, extra=0):
        return int((workspace + self.config.safety_margin_gib) * GIB) + extra

    def _actor(self, action="status", **kwargs):
        return single_status(self.actor(action, **kwargs))

    def _rollout(self, action="status", **kwargs):
        return single_status(self.rollout(action, **kwargs))

    def _check(self, status, needed, phase, decisions):
        record = {
            "phase": phase,
            "decisions": decisions,
            "required_free_bytes": needed,
            "available_bytes": self.free(status),
            "device_free_bytes": status["free_bytes"],
        }
        self.log("AUTO_MEMORY " + json.dumps(record))
        if self.free(status) < needed:
            raise RuntimeError(f"Insufficient memory budget for {phase}: {record}")

    def _pressure_enabled(self, status):
        return os.getenv("AREAL_PRESSURE_MEMORY", "1") == "1" and status.get(
            "supports_partial_reclaim", False
        )

    def _reclaim_shortfall(self, owner, status, needed, phase, check=True):
        import time

        deficit = max(0, needed - self.free(status))
        if not deficit:
            self._check(status, needed, phase, ["resident"])
            return status
        # Translate the logical device cap into a physical free-memory target.
        # needed already includes the configured safety margin as hysteresis.
        cap = self.config.device_budget_gib
        target = (
            needed + max(0, status["total_bytes"] - int(cap * GIB))
            if cap is not None
            else needed
        )
        deficit = max(0, target - status["free_bytes"])
        started = time.perf_counter()
        after = owner("reclaim", target_free_bytes=target)
        self.log(
            "PRESSURE_MEMORY "
            + json.dumps(
                {
                    "phase": phase,
                    "shortfall_bytes": deficit,
                    "target_free_bytes": target,
                    "free_bytes_before": status["free_bytes"],
                    "free_bytes_after": after["free_bytes"],
                    "cpu_backup_bytes": after.get("cpu_backup_bytes", 0),
                    "seconds": time.perf_counter() - started,
                }
            )
        )
        if check:
            self._check(after, needed, phase, ["partial_reclaim"])
        return after

    def initial_inference(self, memory_fraction):
        actor = self._actor()
        needed = self.required(
            0,
            int(memory_fraction * actor["total_bytes"])
            + actor.get("publication_workspace_bytes", 0),
        )
        if self._pressure_enabled(actor):
            self._reclaim_shortfall(self._actor, actor, needed, "initial_inference")
            return
        decisions = ["resident"]
        if self.free(actor) < needed:
            actor = self._actor("release_cache")
            decisions.append("release_actor_cache")
        if self.free(actor) < needed:
            actor = self._actor("offload")
            decisions.append("actor_cpu")
        self._check(actor, needed, "initial_inference", decisions)

    def train(self):
        actor = self._actor()
        needed = self.required(
            self.config.train_workspace_gib,
            actor["restore_bytes"]
            + actor["future_optimizer_bytes"]
            + actor["gradient_bytes"]
            + actor.get("publication_workspace_bytes", 0),
        )
        status = self._rollout()
        if self._pressure_enabled(actor) and self._pressure_enabled(status):
            # Ordinary actor caches cannot always be reused by the training pool.
            # Under pressure reclaim disposable memory before copying live
            # weights to RAM. Never evict parked or CPU-backed allocations.
            if (
                self.free(status) < needed
                and not actor["restore_bytes"]
                and not actor.get("workspace_parked_bytes", 0)
            ):
                status = self._actor("release_cache")
            status = self._reclaim_shortfall(
                self._rollout, status, needed, "train", check=False
            )
            self._check(status, needed, "train", ["pressure_budget_verified"])
            self._actor("restore")
            return
        decisions = ["resident"]
        if self.free(status) < needed:
            status = self._rollout("release_kv")
            decisions = ["release_kv"]
        if self.free(status) < needed and not actor.get("workspace_parked_bytes", 0):
            # Cached bytes are not guaranteed reusable across pools or streams.
            # Reclaim and measure rather than crediting reserved - allocated.
            status = self._actor("release_cache")
            decisions.append("release_actor_cache")
        if self.free(status) < needed:
            status = self._rollout("offload")
            decisions.append("inference_weights_cpu")
        self._check(status, needed, "train", decisions)
        self._actor("restore")

    def concurrent_train(self):
        actor = self._actor()
        rollout = self._rollout()
        if actor["restore_bytes"] or rollout["restore_bytes"]:
            return False
        if not rollout.get("dynamic_kv"):
            return False
        needed = self.required(
            self.config.train_workspace_gib + self.config.inference_workspace_gib,
            actor["future_optimizer_bytes"]
            + actor["gradient_bytes"]
            + actor.get("publication_workspace_bytes", 0),
        )
        # Existing mapped KV is counted in free-memory usage; credit it once.
        budget = max(
            0, self.free(rollout) - needed + rollout["dynamic_kv"]["mapped_bytes"]
        )
        status = self._rollout("kv_budget", kv_budget_bytes=budget)
        admitted = status["concurrent_admitted"]
        self.log(
            "CONCURRENT_MEMORY "
            + json.dumps(
                dict(
                    admitted=admitted,
                    kv_budget_bytes=budget,
                    reserved_growth_bytes=needed,
                    capacity_blocks=status["capacity_blocks"],
                )
            )
        )
        if not admitted:
            self._rollout("kv_budget_clear")
        return admitted

    def inference(self):
        status = self._rollout()
        needed = self.required(
            self.config.inference_workspace_gib, status["restore_bytes"]
        )
        # Retain training caches unless actual free memory cannot fit inference.
        actor = self._actor("finish_train")
        if self._pressure_enabled(actor) and self._pressure_enabled(status):
            self._reclaim_shortfall(self._actor, actor, needed, "inference")
            self._rollout("restore")
            return
        decisions = ["resident"]
        if self.free(actor) < needed:
            action = (
                "park_cache" if "workspace_parked_bytes" in actor else "release_cache"
            )
            actor = self._actor(action)
            decisions.append(action)
        if self.free(actor) < needed:
            actor = self._actor("offload")
            decisions.append("actor_cpu")
        self._check(actor, needed, "inference", decisions)
        self._rollout("restore")


class GroupColocationMemoryPolicy:
    """Independent per-device budgets with a group execution barrier."""

    def __init__(self, config, actor, rollout, log, devices):
        from functools import partial

        self.policies = [
            ColocationMemoryPolicy(
                config,
                partial(actor, device_rank=rank),
                partial(rollout, device_rank=rank),
                lambda message, rank=rank: log(f"device_rank={rank} {message}"),
            )
            for rank in range(devices)
        ]
        self.rollout = rollout

    def initial_inference(self, memory_fraction):
        for policy in self.policies:
            policy.initial_inference(memory_fraction)

    def train(self):
        for policy in self.policies:
            policy.train()

    def inference(self):
        for policy in self.policies:
            policy.inference()

    def concurrent_train(self):
        # Apply each member's budget while every inference scheduler is paused.
        # Resume is a separate group operation, after every member ACKs.
        admitted = [policy.concurrent_train() for policy in self.policies]
        if not all(admitted):
            self.rollout("kv_budget_clear")
        return all(admitted)
