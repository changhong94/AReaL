# SPDX-License-Identifier: Apache-2.0
"""Batch-boundary DP1/DP2 Megatron training with measured transition costs."""

import copy
import json
import logging
import math
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from areal.api.alloc_mode import ModelAllocation
from areal.trainer.idle_entry import EntryBudget
from areal.trainer.idle_training import IdleTrainingPool
from areal.utils import request_trace


class MegatronIdleTrainingPool(IdleTrainingPool):
    def __init__(self, trainer, ft_spec):
        cfg = trainer.config
        if not (
            cfg.npu_share
            and cfg.scheduler.type == "local"
            and trainer.actor_alloc.backend == "megatron"
            and trainer.actor_alloc.world_size == 1
            and trainer.rollout_alloc.world_size == 2
            and trainer.rollout_alloc.parallel.tp_size == 1
            and cfg.critic is None
            and cfg.ref is None
            and cfg.teacher is None
            and cfg.recover.mode == "disabled"
        ):
            raise ValueError(
                "Megatron idle training requires local two-card pure-DP sharing"
            )
        if any(
            os.getenv(k, "0") == "1"
            for k in ("AREAL_IDLE_MID_BATCH", "AREAL_IDLE_HELPER")
        ):
            raise ValueError(
                "Megatron idle training currently switches at full batch boundaries only"
            )
        self.trainer = trainer
        self.primary = trainer.actor
        self._base = trainer.actor_alloc.world_size
        self._expanded = trainer.rollout_alloc.world_size
        training_capacity = trainer.rollout_alloc.world_size
        self._retains = {w: w < training_capacity for w in (self._base, self._expanded)}
        self._standby = {self._base: self._expanded, self._expanded: self._base}
        self.current = self.snapshot_actor = self.primary
        self.active = False
        self.stats = {}
        self._entry_budget = EntryBudget(self._base, tuple(self._retains))
        self._entry_prefetch = None
        self._preempted_inference = False
        self._publication_tail_costs = {1: [], 2: []}
        self._cpu_jobs = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="megatron-cpu-stage"
        )
        self._cpu_stage = None
        self._cpu_export = None
        self._cpu_capture = None
        self._maintenance_batch = None
        self._transfer_costs = []
        self._restore_costs = {1: [], 2: []}
        self._prepared = None
        self.path = Path("/dev/shm") / ("areal-megatron-idle-" + uuid.uuid4().hex)
        self.path.mkdir(mode=0o700)
        self.critical = os.getenv("AREAL_IDLE_CRITICAL_PATH", "0") == "1"
        self._resident_batch = None
        self.probes = {}
        self.cal_costs = {}
        self.live_compute = {}
        self.actual_costs = {}
        self.cost_origins = {}
        self.cost_observations = []
        self.real_publications = {}
        self.primary.idle_training_state("prepare")
        self.primary.offload()
        multi_cfg = copy.deepcopy(cfg.actor)
        multi_cfg.backend = "megatron:d2p1t1"
        for spec in multi_cfg.scheduling_spec:
            spec.env_vars["HCCL_NPU_SOCKET_PORT_RANGE"] = "auto"
            spec.env_vars["AREAL_IDLE_COLLECTIVE_MARGIN_BYTES"] = str(
                int(cfg.colocation_memory.safety_margin_gib * 1024**3)
            )
        alloc = ModelAllocation.from_str(multi_cfg.backend, name="actor-multi")
        self.multi = trainer._create_train_engine(multi_cfg, alloc)
        self.multi.initialize(addr=None, ft_spec=ft_spec, role="actor-multi")
        self.multi.idle_training_state("prepare")
        self.multi.offload()
        transfer_start = time.monotonic()
        initial_path = self.path / "generation-0"
        exported = self.primary.idle_training_state("save", str(initial_path))
        imported = self.multi.idle_training_state("stage", str(initial_path))
        self.deployment_transfer = time.monotonic() - transfer_start
        if exported["version"] != 0 or any(
            r["version"] != 0 for r in imported["ranks"]
        ):
            raise RuntimeError("Initial CPU preparation has a nonzero training version")
        if os.getenv("AREAL_IDLE_PREEMPT_FIRST", "0") == "1":
            # Verify deployment's prebuilt version 0 outside the timed RL loop.
            self.multi.onload()
            comm = self.multi._custom_function_call("prime_idle_optimizer_collective")
            request_trace.emit("idle_collective_memory_primed", **comm)
            checked = self.multi.idle_training_state("verify", str(initial_path))
            if not all(row["verified"] for row in checked["ranks"]):
                raise RuntimeError("Initial dual engine verification failed")
            self.multi.offload()
            request_trace.emit(
                "idle_initial_state_verified", version=0, ranks=checked["ranks"]
            )
        self._prepared = dict(
            source=self.primary,
            destination=self.multi,
            path=initial_path,
            exported=exported,
            imported=imported,
        )
        self.primary.onload()
        request_trace.emit(
            "idle_training_initialized",
            engine="megatron",
            single_world_size=1,
            multi_world_size=2,
            transport="background_cpu_staging",
        )

    def _remaining_preparation(self, batch):
        source_world = (
            self._base if self.snapshot_actor is self.primary else self._expanded
        )
        target_world = self._standby[source_world]
        if self._cpu_stage is not None and self._cpu_stage.done():
            self._join_cpu_stage()
        ready = self._maintenance_batch == batch and self._prepared is not None
        captured = (
            self._cpu_capture.result()
            if self._cpu_capture is not None and self._cpu_capture.done()
            else None
        )
        # Pending capture has no measured remaining duration yet. Do not fabricate it.
        capture_known = ready or captured is not None or batch == 1
        remaining = {
            world: max(costs[-3:]) if costs else None
            for world, costs in self._restore_costs.items()
        }
        if not capture_known:
            return source_world, dict.fromkeys(self._retains), ready
        if not ready:
            if not self._transfer_costs or not captured:
                remaining[target_world] = None
            elif remaining[target_world] is not None:
                transfer = max(self._transfer_costs[-3:])
                elapsed = max(0.0, time.monotonic() - captured["started"])
                pending = max(0.0, transfer - elapsed)
                if self._maintenance_batch != batch:
                    pending += transfer
                remaining[target_world] += pending
        return source_world, remaining, ready

    def _full_entry_prediction(self, batch, snapshot):
        current, remaining, ready = self._remaining_preparation(batch)
        rollout = self.trainer.rollout
        leases = getattr(rollout, "_request_pool", None)
        requests = (
            sum(
                len(leases.active_requests(rank))
                for rank in range(len(rollout.workers))
            )
            if leases is not None
            else None
        )
        # Dispatcher counts tasks, leases count requests. Never mix their units.
        if requests == 0 and sum(snapshot.values()):
            requests = None
        if getattr(self, "critical", False):
            return self._critical_prediction(batch, current, remaining, ready)
        decision = self._entry_budget.decide_paths(
            batch,
            current_layout=current,
            remaining_prepare=remaining,
            drain_work={
                w: 0 if keep else requests for w, keep in self._retains.items()
            },
            include_tail=True,
            include_handoff=True,
        )
        return dict(
            decision,
            accepted=decision["world_size"] == self._expanded,
            standby_ready=ready,
            target_version=batch - 1,
        )

    def commission(self):
        # Rehearse only on fresh version zero before any real submissions.
        if not self.critical:
            return
        t = self.trainer
        ro = t.rollout
        if ro.get_version() != 0 or ro.staleness_manager.get_stats().accepted:
            raise RuntimeError("Calibration must precede rollout")
        folder = os.getenv("AREAL_BENCH_WORKSPACE")
        cache = Path(folder) / "layout-calibration-in.json" if folder else None
        if cache and cache.exists():
            data = json.loads(cache.read_text())
            self.cache_ref = json.loads(
                (cache.parent / "calibration-inputs.json").read_text()
            )["training-layout"]
            self.cost_origins = self.cache_ref["origins"]
            self.probes = {int(w): v for w, v in data["probes"].items()}
            self.cal_costs = {int(w): v for w, v in data["costs"].items()}
            self.actual_costs = {int(w): dict(v) for w, v in data["actual"].items()}
            self.live_compute = {
                w: v["compute_per_position"]
                for w, v in self.actual_costs.items()
                if "compute_per_position" in v
            }
            request_trace.emit(
                "idle_deployment_cache_hit", actual=self.actual_costs, version=0
            )
            return
        start = time.monotonic()
        self._requested_batch = self._maintenance_batch = 1
        length = min(
            t.config.gconfig.max_new_tokens, t.config.actor.mb_spec.max_tokens_per_mb
        )
        # Same global token workload, multiple microbatches on each candidate.
        rows = (
            2
            * len(ro.workers)
            * max(1, t.config.actor.mb_spec.max_tokens_per_mb // length)
        )
        for world, actor in ((1, self.primary), (2, self.multi)):
            enter = time.monotonic()
            if not self._apply_preemption(0, dict(accepted=(world == 2))):
                raise RuntimeError("Calibration capacity rejection")
            entry = time.monotonic() - enter
            self.probes[world] = actor._custom_function_call(
                "idle_compute_probe", rows, length
            )
            checked = actor.idle_training_state("verify", str(self._prepared["path"]))
            if not all(
                x["verified"] and x["version"] == 0 and x["scheduler_steps"] == 0
                for x in checked["ranks"]
            ):
                raise RuntimeError("Calibration altered initial actor state")
            self.current = actor
            pub = time.monotonic()
            if world == 1:
                t._return_shared_inference = True
                t._publish_fixed_weights(0)
                t._finish_shared_weights(0)
            else:
                self.publish_multi(0)
            ro.wait_shared_publication()
            ro.wait_running_rebalance()
            self.cal_costs[world] = dict(
                entry=entry, publication=time.monotonic() - pub
            )
        self.current = self.snapshot_actor = self.primary
        self.active = False
        if ro.get_version() != 0 or ro.staleness_manager.get_stats().accepted:
            raise RuntimeError("Calibration altered rollout version or accepted data")
        self._entry_budget = EntryBudget(self._base, tuple(self._retains))
        for w in self.probes:
            for metric, val in (
                (
                    "compute_per_position",
                    self.probes[w]["seconds"] / self.probes[w]["tokens"],
                ),
                ("entry", self.cal_costs[w]["entry"]),
                ("publication", self.cal_costs[w]["publication"]),
            ):
                self._cost_sample(
                    w,
                    metric,
                    val,
                    0,
                    "deployment_probe",
                    self.probes[w]["tokens"]
                    if metric == "compute_per_position"
                    else None,
                )
        request_trace.emit(
            "idle_deployment_probe",
            probes=self.probes,
            costs=self.cal_costs,
            elapsed=time.monotonic() - start,
            version=0,
        )

    def _cost_sample(self, world, metric, value, batch, source, positions=None):
        if not math.isfinite(value) or value <= 0:
            return
        label = f"{world}:{metric}"
        row = dict(
            source=source,
            run=self.trainer.config.trial_name,
            workspace=os.getenv("AREAL_CALIBRATION_RUN_ID"),
            batch=batch,
            layout=world,
            metric=metric,
            value=value,
            unit="seconds_per_position" if positions else "seconds",
            positions=positions,
            seconds=value * positions if positions else value,
            time=time.time(),
        )
        self.cost_origins[label] = row
        self.cost_observations.append(row)
        if source == "runtime_observation":
            self.actual_costs.setdefault(world, {})[metric] = value

    def observe_full_publication(self, version, world, started):
        if not self.critical or version <= 0:
            return
        # Callback runs on publication completion. Never add a barrier to the training loop.
        future = self.trainer.rollout._shared_publication
        if future is not None:

            def capture(f):
                if not f.cancelled() and f.exception() is None:
                    self.real_publications[world] = (
                        version,
                        time.monotonic() - started,
                    )

            future.add_done_callback(capture)

    def collect_publications(self):
        for w, (version, secs) in self.real_publications.copy().items():
            if (
                self.cost_origins.get(f"{w}:publication", {}).get("workspace")
                != os.getenv("AREAL_CALIBRATION_RUN_ID")
                or self.cost_origins.get(f"{w}:publication", {}).get("batch") != version
            ):
                self._cost_sample(
                    w, "publication", secs, version, "runtime_observation"
                )

    def export_deployment_costs(self):
        self.collect_publications()
        return dict(probes=self.probes, costs=self.cal_costs, actual=self.actual_costs)

    def export_cost_audit(self):
        return dict(origins=self.cost_origins, observations=self.cost_observations)

    def prepare_decision(self, data, batch, terminal):
        if not self.critical:
            return
        self.collect_publications()
        tokens = 0
        for row in data:
            mask = row.get("attention_mask")
            if mask is None or not hasattr(mask, "shape"):
                tokens = None
                break
            # Shape is available for remote tensors without materializing rollouts.
            # This is padded workload volume, not an exact count of nonpadding tokens.
            tokens += math.prod(mask.shape)
        self._work_tokens = tokens
        self._no_next_batch = terminal

    def _critical_prediction(self, batch, current, remaining, ready):
        from areal.trainer.layout_horizon import predict_layout

        ro = self.trainer.rollout
        width = self.trainer.config.train_dataset.batch_size
        accepted = ro.staleness_manager.get_stats().accepted
        missing = 0 if self._no_next_batch else max(0, (batch + 1) * width - accepted)
        elapsed = time.monotonic() - self._batch_clock
        observed = accepted - self._accepted_at_arm
        # Question rate includes scoring and acceptance. Request service uses
        # separately timed lease completions and is never inferred from task count.
        rate = observed / elapsed if observed > 0 and elapsed > 0 else None
        if rate is not None:
            self._question_rate = rate
        route = ro._request_pool.drain_snapshot()
        leases = len(route["active"])
        durations = [
            o["lease_seconds"]
            for o in route["window_observations"]
            if o.get("lease_seconds", 0) > 0
        ]
        if durations:
            self._lease_service = max(durations)
        work = self._work_tokens
        compute = {}
        for world, probe in self.probes.items():
            if not work:
                compute[world] = None
            elif world in self.live_compute:
                compute[world] = self.live_compute[world] * work
            else:
                compute[world] = work * probe["seconds"] / probe["tokens"]
        entry = {
            w: self.actual_costs.get(w, {}).get("entry", c["entry"])
            for w, c in self.cal_costs.items()
        }
        target = self._standby[current]
        if not ready:
            # Deployment entry already includes restore. Add only exposed CPU staging.
            if self._resident_batch == batch:
                entry[current] = (
                    self._entry_budget.estimate(batch, current, "handoff")
                    or entry[current]
                )
                entry[target] += max(
                    self._transfer_costs[-3:] or [self.deployment_transfer]
                )
            elif remaining[target] is None or not self._restore_costs[target]:
                entry[target] = None
            else:
                entry[target] += max(
                    0.0, remaining[target] - max(self._restore_costs[target][-3:])
                )
        pub = {
            w: self.actual_costs.get(w, {}).get("publication", c["publication"])
            for w, c in self.cal_costs.items()
        }
        from areal.trainer.layout_progress import estimate_progress

        engines = None
        if leases and not self._no_next_batch:
            import requests

            try:
                with requests.Session() as session:
                    session.trust_env = False
                    engines = []
                    for address in ro._request_pool.addresses:
                        response = session.post(
                            f"http://{address}/areal_generation_progress",
                            json={"include_window": True},
                            timeout=2,
                        )
                        response.raise_for_status()
                        engines.append(response.json())
            except (requests.RequestException, ValueError):
                engines = None
        progress = estimate_progress(
            route, engines, missing, self.trainer.config.gconfig.n_samples
        )
        result = predict_layout(
            current,
            compute,
            entry,
            pub,
            missing,
            getattr(self, "_question_rate", None),
            leases,
            None,
            self._no_next_batch,
            baseline=self._base,
            retains_inference=self._retains,
            progress=progress,
        )
        return dict(
            result,
            accepted=result["world_size"] == self._expanded,
            standby_ready=ready,
            target_version=batch - 1,
            padded_token_positions=work,
            accepted_questions=accepted,
            calibration_revision=getattr(self, "cache_ref", {}).get("revision"),
            measurement_origins=dict(self.cost_origins),
        )

    def _try_initial_preemption(self, batch):
        all_batches = os.getenv("AREAL_IDLE_PREEMPT_ALL", "0") == "1"
        if (
            not all_batches
            and (batch != 1 or os.getenv("AREAL_IDLE_PREEMPT_FIRST", "0") != "1")
        ) or os.getenv("AREAL_IDLE_ENTRY_VALIDATE_WORLDS", ""):
            return False
        prepared = self._prepared
        if batch == 1 and (
            prepared is None
            or prepared["source"] is not self.primary
            or prepared["destination"] is not self.multi
            or prepared["exported"]["version"] != 0
            or prepared["imported"]["version"] != 0
        ):
            raise RuntimeError(
                "First-batch preemption requires intact prepared version 0"
            )
        rollout = self.trainer.rollout
        wait_started = time.monotonic()
        rollout.wait_shared_publication()
        rollout.wait_running_rebalance()
        tail_wait = time.monotonic() - wait_started
        if all_batches and batch > 1:
            world = 1 if self.snapshot_actor is self.primary else 2
            if not hasattr(self, "_publication_tail_costs"):
                self._publication_tail_costs = {1: [], 2: []}
            self._publication_tail_costs[world].append(tail_wait)
            self._entry_budget.observe(batch - 1, world, "tail", tail_wait)
            request_trace.emit(
                "idle_publication_exposed_tail",
                batch=batch - 1,
                world_size=world,
                seconds=tail_wait,
            )
        snapshot = rollout.dispatcher.generation_work_snapshot()
        decision = self._full_entry_prediction(batch, snapshot)
        # Keep inference running while an otherwise viable standby finishes staging.
        # Wait at most the measured compute+entry of the inference-blocking path;
        # after that, or on readiness, take a fresh snapshot rather than assuming it is ready.
        if (
            getattr(self, "critical", False)
            and not self._no_next_batch
            and not self._retains[decision["current_world"]]
            and decision["reason"] == "insufficient_current_run_evidence"
            and decision["missing_questions"] > 0
            and decision["lease_count"] > 0
            and self._maintenance_batch == batch
            and self._cpu_stage is not None
            and decision["candidates"][self._standby[decision["current_world"]]][
                "entry"
            ]
            is None
        ):
            here = decision["candidates"][decision["current_world"]]
            alternative = decision["lower_bounds"][
                self._standby[decision["current_world"]]
            ]
            budget = (
                here["total"] - alternative
                if here["total"] is not None
                else here["compute"]
            )
            if budget is not None and budget > 0:
                from concurrent.futures import wait

                before = time.monotonic()
                wait([self._cpu_stage], timeout=budget)
                snapshot = rollout.dispatcher.generation_work_snapshot()
                decision = self._full_entry_prediction(batch, snapshot)
                request_trace.emit(
                    "idle_inference_standby_recheck",
                    batch=batch,
                    elapsed=time.monotonic() - before,
                    budget=budget,
                    ready=decision["standby_ready"],
                )
        request_trace.emit(
            "idle_initial_preempt_decision",
            batch=batch,
            generation=snapshot,
            **decision,
        )
        end_to_end = time.monotonic()
        applied = self._apply_preemption(batch, decision)
        if applied and getattr(self, "critical", False):
            w = 2 if self._entry_actor is self.multi else 1
            # Only ready-entry samples: do not relabel stale CPU transfer or resident reuse.
            if (
                getattr(self, "_memory_rejected_batch", None) != batch
                and self._resident_batch != batch
                and (w == decision["current_world"] or decision["standby_ready"])
            ):
                self._cost_sample(
                    w,
                    "entry",
                    time.monotonic() - end_to_end,
                    batch,
                    "runtime_observation",
                )
        return applied

    def _reject_multi_memory(self, batch, reason):
        rollout = self.trainer.rollout
        held = getattr(rollout, "_shared_held_for_training", False)
        ranks = (
            tuple(rollout.fixed_ranks) if held else tuple(range(len(rollout.workers)))
        )
        rollout._group_rpc(ranks, "colocation_memory", action="restore")
        rollout._group_rpc(ranks, "continue_generation", http_timeout=60.0)
        for rank in ranks:
            rollout._request_pool.open(rank, rollout.get_version())
        rollout.resume()
        self._preempted_inference = False
        self._resident_batch = None
        self._memory_rejected_batch = batch
        request_trace.emit("idle_initial_preempt_rejected", reason=reason, batch=batch)
        if batch == 0:
            raise RuntimeError("Insufficient device memory for deployment calibration")
        # Downgrade through the existing version-checked single restore, once.
        return self._apply_preemption(batch, dict(accepted=False))

    def _apply_preemption(self, batch, decision):
        rollout = self.trainer.rollout
        all_batches = os.getenv("AREAL_IDLE_PREEMPT_ALL", "0") == "1"
        if not decision["accepted"]:
            if not all_batches:
                return False
            started = time.monotonic()
            held = getattr(self, "_resident_batch", None) == batch and getattr(
                rollout, "_shared_held_for_training", False
            )
            if held:
                self.trainer._memory_policy.train(resident=True)
                self.active = False
            else:
                rollout.borrow_for_training()
                self._entry_budget.observe(
                    batch, 1, "handoff", time.monotonic() - started
                )
                self._restore_single_from_dram()
            self._entry_actor = self.primary
            request_trace.emit(
                "idle_training_selected",
                batch=batch,
                world_size=1,
                reason="remaining_path_cost",
                preparation_seconds=time.monotonic() - started,
                direct_from_inference=True,
            )
            return True
        pool = getattr(rollout, "_request_pool", None)
        if pool is None or len(rollout.workers) != 2:
            raise RuntimeError("Preemption requires two request-lease replicas")
        from areal.trainer.colocation_memory import GIB, single_status

        policy = self.trainer._memory_policy
        if hasattr(policy, "policies"):
            policy = policy.policies[0]
        started = time.monotonic()
        # Called only by select, after prepare_batch returned all answers.
        # Close admission but preserve every running request and its old model.
        rollout.pause()
        rollout._collective_rpc("pause_generation_preserve", http_timeout=60.0)
        self._preempted_inference = True
        if self.snapshot_actor is self.primary:
            self.primary.offload()
        budgets = []
        actors = [
            single_status(self.multi.colocation_memory(device_rank=r)) for r in range(2)
        ]
        # Actor DP ranks and rollout ranks may enumerate devices oppositely.
        # Conservatively reserve the larger actor requirement on both devices.
        actor = max(actors, key=lambda a: a["total_bytes"])

        def workspace(a):
            return max(
                policy.config.train_workspace_gib * GIB,
                a.get("observed_workspace_peak_bytes", 0),
            )

        required = max(
            policy.required(
                workspace(a) / GIB,
                a["restore_bytes"]
                + a["future_optimizer_bytes"]
                + a["gradient_bytes"]
                + a.get("publication_workspace_bytes", 0),
            )
            for a in actors
        )
        for rank in range(2):
            cap = policy.config.device_budget_gib
            target = required + (
                max(0, actor["total_bytes"] - int(cap * GIB)) if cap is not None else 0
            )
            after = single_status(
                rollout._group_rpc(
                    (rank,),
                    "colocation_memory",
                    action="reclaim",
                    target_free_bytes=target,
                )
            )
            budgets.append(
                dict(rank=rank, required=required, available=policy.free(after))
            )
        request_trace.emit("idle_initial_preempt_memory", budgets=budgets)
        if any(row["available"] < row["required"] for row in budgets):
            return self._reject_multi_memory(batch, "pre_restore_memory")
        self._entry_budget.observe(batch, 2, "handoff", time.monotonic() - started)
        self._restore_multi_from_dram()
        # CPU staging and onload happen after the first check. Inspect the actual
        # actor ranks now, with mapped state and the native communicator resident.
        post = []
        for rank in range(len(actors)):
            status = single_status(self.multi.colocation_memory(device_rank=rank))
            need = policy.required(
                workspace(status) / GIB,
                status["future_optimizer_bytes"]
                + status["gradient_bytes"]
                + status.get("publication_workspace_bytes", 0),
            )
            if policy.free(status) < need:
                status = single_status(
                    self.multi.colocation_memory("release_cache", device_rank=rank)
                )
            post.append(
                dict(
                    rank=rank,
                    available=policy.free(status),
                    required=need,
                    observed_workspace=status.get("observed_workspace_peak_bytes", 0),
                )
            )
        request_trace.emit("idle_post_restore_memory", batch=batch, budgets=post)
        if any(x["available"] < x["required"] for x in post):
            self.multi.offload()
            return self._reject_multi_memory(batch, "post_restore_memory")
        self.active = True
        self._entry_actor = self.multi
        request_trace.emit(
            "idle_training_selected",
            batch=batch,
            world_size=2,
            reason="predicted_preemption_gain",
            direct_from_inference=True,
            preparation_seconds=time.monotonic() - started,
        )
        return True

    def _prepare_entry(self, batch):
        if self._try_initial_preemption(batch):
            return
        validation = os.getenv("AREAL_IDLE_ENTRY_VALIDATE_WORLDS", "")
        if validation:
            if os.getenv("AREAL_IDLE_FULL_VERIFY", "0") != "1":
                raise ValueError(
                    "Forced entry worlds require full correctness verification"
                )
            worlds = [int(value) for value in validation.split(",")]
            if batch <= len(worlds) and worlds[batch - 1] == 2:
                # Correctness-only runs must exercise the requested transition.
                # Finish in-flight generation; never evict an active inference request.
                started = time.monotonic()
                dispatcher = self.trainer.rollout.dispatcher
                while sum(dispatcher.generation_work_snapshot().values()):
                    if time.monotonic() - started > 120:
                        raise RuntimeError(
                            "Correctness sequence could not reach an idle inference fence"
                        )
                    time.sleep(0.05)
                request_trace.emit(
                    "idle_training_validation_drain",
                    batch=batch,
                    seconds=time.monotonic() - started,
                )
        # The same evidence policy applies to idle and preemptive entry.
        original = self._entry_budget.decide

        def decide(step, runnable, terminal=False):
            source_world, remaining, ready = self._remaining_preparation(batch)
            result = original(
                step, runnable, terminal=terminal, remaining_prepare=remaining
            )
            result.update(
                current_world=source_world,
                remaining_prepare=remaining,
                standby_ready=ready,
            )
            return result

        self._entry_budget.decide = decide
        try:
            super()._prepare_entry(batch)
        finally:
            self._entry_budget.decide = original
        if validation and batch <= len(worlds):
            actual = 2 if self._entry_actor is self.multi else 1
            if actual != worlds[batch - 1]:
                raise RuntimeError(
                    "Correctness sequence did not exercise the requested DP world"
                )

    def arm(self, batch):
        if getattr(self, "critical", False):
            self._batch_clock = time.monotonic()
            self._accepted_at_arm = (
                self.trainer.rollout.staleness_manager.get_stats().accepted
            )
        self._requested_batch = batch
        if (
            getattr(self, "critical", False)
            and batch > 1
            and self.snapshot_actor is self.primary
            and not getattr(self.trainer, "_return_shared_inference", True)
        ):
            # The previous publication explicitly kept this actor resident.
            # Do not evict it solely to refresh a CPU standby; refresh on demand before switching.
            self._resident_batch = batch
            request_trace.emit("idle_resident_held", batch=batch)
            return
        if batch == 1 and self._prepared is not None:
            self._maintenance_batch = batch
            self._maintenance_started = time.monotonic()
        elif self._cpu_stage is None or self._cpu_stage.done():
            self._join_cpu_stage()
            self._start_maintenance(batch)
        else:
            # Bound memory to one outstanding generation. Keep current engine
            # usable; catch up only if the other engine is actually selected.
            request_trace.emit(
                "idle_standby_coalesced",
                batch=batch,
                pending_batch=self._maintenance_batch,
            )
        if os.getenv("AREAL_IDLE_PREEMPT_ALL", "0") == "1" or (
            batch == 1 and os.getenv("AREAL_IDLE_PREEMPT_FIRST", "0") == "1"
        ):
            # A prefetch thread could pause requests needed by prepare_batch.
            # Defer only this selection to the data-ready select call.
            return
        super().arm(batch)

    def _start_maintenance(self, batch):
        if self._cpu_stage is not None:
            raise RuntimeError("Concurrent standby writers are forbidden")
        if self._prepared is not None:
            # _cpu_stage is joined and any verification RPC has returned. No
            # writer or reader retains these snapshot files; only CPU copies do.
            if os.getenv("AREAL_IDLE_RAW_TRANSPORT", "0") == "1":
                pool = self.path / "recycled-raw"
                pool.mkdir(mode=0o700, exist_ok=True)
                for raw in self._prepared["path"].glob("tensor-*.bin"):
                    raw.rename(pool / raw.name)
            shutil.rmtree(self._prepared["path"])
            self._prepared = None
        source = self.snapshot_actor
        destination = self.multi if source is self.primary else self.primary
        previous_tail = getattr(self.trainer, "_step_tail", None)
        path = self.path / ("generation-" + str(batch - 1))
        self._maintenance_batch = batch
        self._maintenance_started = time.monotonic()

        def capture():
            if previous_tail is not None:
                previous_tail.result()
            self.trainer.rollout.wait_shared_publication()
            source.offload()
            exported = source.idle_training_state("save_start", str(path))
            if exported["version"] != batch - 1:
                raise RuntimeError("Background export has stale training version")
            return dict(
                source=source,
                destination=destination,
                path=path,
                exported=exported,
                started=time.monotonic(),
            )

        self._cpu_capture = self._cpu_jobs.submit(capture)
        captured_future = self._cpu_capture

        def stage():
            result = captured_future.result()
            started = time.monotonic()
            world = result["exported"]["world"]
            while True:
                files = [path / f"ready-{rank}.json" for rank in range(world)]
                if all(f.exists() for f in files):
                    rows = [json.loads(f.read_text()) for f in files]
                    if any("error" in row for row in rows):
                        raise RuntimeError("CPU export failed: " + repr(rows))
                    if any(
                        row["rank"] != rank
                        or row["version"] != batch - 1
                        or row["world"] != world
                        for rank, row in enumerate(rows)
                    ):
                        raise RuntimeError("CPU export mixed training generations")
                    break
                if time.monotonic() - started > 180:
                    raise RuntimeError("CPU export did not complete within 180 seconds")
                time.sleep(0.01)
            (path / "world.txt").write_text(str(world))
            imported = destination.idle_training_state("stage", str(path))
            if (
                imported["version"] != batch - 1
                or len(imported["ranks"]) != imported["target_world"]
                or any(rank["version"] != batch - 1 for rank in imported["ranks"])
            ):
                raise RuntimeError("Background stage mixed training versions")
            seconds = time.monotonic() - result["started"]
            self._transfer_costs.append(seconds)
            request_trace.emit(
                "idle_training_cpu_ready",
                engine="megatron",
                batch=batch,
                version=imported["version"],
                seconds=seconds,
                destination_world=2 if destination is self.multi else 1,
                exports=rows,
                stage_ranks=imported["ranks"],
            )
            return {**result, "imported": imported}

        self._cpu_stage = self._cpu_jobs.submit(stage)
        request_trace.emit("idle_training_cpu_started", engine="megatron", batch=batch)

    def _join_cpu_stage(self):
        if self._cpu_stage is not None:
            started = time.monotonic()
            self._prepared = self._cpu_stage.result()
            self._cpu_stage = None
            request_trace.emit(
                "idle_training_cpu_exposed_wait",
                engine="megatron",
                seconds=time.monotonic() - started,
            )

    def cancel_entry(self):
        try:
            super().cancel_entry()
        finally:
            try:
                self._join_cpu_stage()
            finally:
                self.trainer._wait_step_tail()

    def _restore(self, destination):
        if destination is not self.snapshot_actor:
            self._join_cpu_stage()
            if self._maintenance_batch != self._requested_batch:
                self._start_maintenance(self._requested_batch)
                self._join_cpu_stage()
        elif (
            self._cpu_capture is not None
            and self._maintenance_batch == self._requested_batch
        ):
            # Only capture metadata and lease CPU backing, not clone/serialize
            # tensors or wait for the other training engine's writes.
            self._cpu_capture.result()
        if destination is not self.snapshot_actor:
            prepared = self._prepared
            if (
                prepared is None
                or prepared["source"] is not self.snapshot_actor
                or prepared["destination"] is not destination
            ):
                raise RuntimeError("Missing current-generation CPU preparation")
        started = time.monotonic()
        destination.onload()
        world = 2 if destination is self.multi else 1
        self._restore_costs[world].append(time.monotonic() - started)
        if destination is not self.snapshot_actor:
            verified = None
            if os.getenv("AREAL_IDLE_FULL_VERIFY", "1") == "1":
                verified = destination.idle_training_state(
                    "verify", str(prepared["path"])
                )
            request_trace.emit(
                "idle_training_state_transferred",
                engine="megatron",
                direction="to_multi" if destination is self.multi else "to_single",
                exported=prepared["exported"],
                imported=prepared["imported"],
                verified=verified,
                cpu_prepared=True,
            )
        else:
            request_trace.emit(
                "idle_training_current_restored",
                world_size=world,
                standby_pending=self._cpu_stage is not None
                and not self._cpu_stage.done(),
            )

    def _restore_single_from_dram(self):
        self._restore(self.primary)
        self.trainer._memory_policy.train(resident=True)
        self.active = False

    def _restore_multi_from_dram(self):
        # First-batch DP2 may start while deployment left DP1 resident.
        if self.snapshot_actor is self.primary:
            self.primary.offload()
        self._restore(self.multi)

    def ppo_update(self, actor, data, batch):
        actor.ppo_update(data)
        return actor

    def finish(self, batch):
        # Standby maintenance is not part of this batch's critical path. Its
        # immutable lease is reclaimed by the next maintenance task or close.
        self._entry_budget.observe(
            batch,
            self._entry_world,
            "compute",
            time.monotonic() - self._entry_compute_started,
        )
        if getattr(self, "critical", False) and self._work_tokens:
            self.live_compute[self._entry_world] = (
                time.monotonic() - self._entry_compute_started
            ) / self._work_tokens
            self._cost_sample(
                self._entry_world,
                "compute_per_position",
                self.live_compute[self._entry_world],
                batch,
                "runtime_observation",
                self._work_tokens,
            )
        self.snapshot_actor = self.current

    def publish_multi(self, version):
        from dataclasses import replace

        from areal.infra.utils.concurrent import call_maybe_async

        rollout = self.trainer.rollout
        started = time.monotonic()
        meta = replace(
            self.trainer.weight_update_meta.with_version(version),
            type="disk",
            npu_share_snapshot=True,
            clear_checkpoint_after_load=False,
        )
        rollout.wait_shared_publication()
        rollout.wait_running_rebalance()
        rollout.pause()
        if not self._preempted_inference:
            call_maybe_async(rollout.pause_generation)
        self.current._custom_function_call("idle_publish_snapshot", meta)
        self.current.set_version(version)
        stats = self.current.export_stats()
        self.current.colocation_memory("finish_train")
        self.current.colocation_memory("park_cache")
        self.current.offload()
        self._preempted_at_publish = self._preempted_inference
        drain_seconds = 0.0
        if self._preempted_inference:
            # New snapshot is on CPU; retain old inference weights and KV until
            # all their existing leases finish. Admission remains closed.
            old_version = rollout.get_version()
            counts = [
                len(rollout._request_pool.active_requests(rank)) for rank in range(2)
            ]
            rollout._collective_rpc("colocation_memory", action="restore")
            rollout._collective_rpc("continue_generation", http_timeout=60.0)
            drain_started = time.monotonic()
            for rank in range(2):
                rollout._request_pool.wait_empty(
                    rank, timeout=rollout.config.request_timeout
                )
            drain_seconds = time.monotonic() - drain_started
            self._entry_budget.observe_drain(version, 2, sum(counts), drain_seconds)
            request_trace.emit(
                "idle_preempt_old_version_drained",
                old_version=old_version,
                new_version=version,
                old_requests=counts,
                seconds=drain_seconds,
            )
            rollout._collective_rpc("pause_generation", http_timeout=60.0)
            self._preempted_inference = False
        rollout._group_rpc(rollout.fixed_ranks, "colocation_memory", action="restore")
        rollout.prepare_fixed_publication()
        rollout._group_rpc(rollout.fixed_ranks, "update_weights_from_disk", meta=meta)
        rollout.publish_fixed_version(version)
        rollout.load_shared_snapshot_async(meta, restore_memory=True)
        if getattr(self, "critical", False) and not any(
            counts if self._preempted_at_publish else ()
        ):
            self.observe_full_publication(version, 2, started)
        self._entry_budget.observe(
            version, 2, "return", max(0.0, time.monotonic() - started - drain_seconds)
        )
        self.active = False
        self.trainer._return_shared_inference = True
        return stats

    def close(self):
        error = None
        try:
            self.cancel_entry()
        except BaseException as exc:
            error = exc
        finally:
            self._cpu_jobs.shutdown(wait=True)
        for actor in (self.primary, self.multi):
            try:
                # Controller timeout does not cancel per-rank CPU writers.
                # Join their leases even on failures before engine destruction.
                actor.idle_training_state("save_wait")
                actor.offload()
            except BaseException as exc:
                error = error or exc
                logging.getLogger(__name__).exception(
                    "Could not close Megatron CPU maintenance"
                )
        if error is not None:
            raise error  # Retain transport and failure evidence.
        shutil.rmtree(self.path)
