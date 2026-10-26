# SPDX-License-Identifier: Apache-2.0

"""Use a preinitialized larger FSDP engine while inference has no runnable work."""

import copy
import json
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from areal.api.alloc_mode import ModelAllocation
from areal.utils import perf_tracer, request_trace
from areal.utils.perf_tracer import Category


class IdleTrainingPool:
    """Use the authoritative engine directly and charge incremental handoff costs.

    The independent process groups are initialized before rollout startup. The
    inactive engine is parked in DRAM. Selection is at complete batch or completed backward-segment boundaries.
    Partial gradients are explicitly preserved before a mid-update handoff. A failed handoff is fatal, not a fallback
    to potentially stale state.
    """

    def __init__(self, trainer, ft_spec):
        cfg = trainer.config
        if not (
            cfg.npu_share
            and cfg.scheduler.type == "local"
            and trainer.actor_alloc.backend == "fsdp"
            and trainer.actor_alloc.world_size == 1
            and trainer.rollout_alloc.world_size == 2
            and trainer.rollout_alloc.parallel.tp_size == 1
            and cfg.critic is None
            and cfg.ref is None
            and cfg.teacher is None
            and cfg.recover.mode == "disabled"
        ):
            raise ValueError(
                "Idle training currently requires local two-NPU FSDP sharing, TP=1, without auxiliary models or recovery"
            )
        if os.getenv("AREAL_IDLE_MID_BATCH", "0") == "1" and (
            cfg.actor.ppo_n_minibatches != 1 or cfg.actor.fsdp.per_layer_optim_step
        ):
            raise ValueError(
                "Mid-batch handoff requires one minibatch and no per-layer updates"
            )
        self.trainer = trainer
        self.primary = trainer.actor
        self.active = False
        from areal.trainer.idle_entry import EntryBudget

        self._entry_budget = EntryBudget()
        self._entry_prefetch = None
        self._entry_actor = None
        self.current = self.primary
        from areal.trainer.idle_budget import SwitchBudget

        self._switch_budget = SwitchBudget(
            initial_cost=float(os.getenv("AREAL_IDLE_SWITCH_COST_SECONDS", "6")),
            initial_speedup=float(os.getenv("AREAL_IDLE_SWITCH_SPEEDUP", "1.5")),
            margin=float(os.getenv("AREAL_IDLE_SWITCH_MARGIN_SECONDS", "0")),
        )
        self._jobs = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="idle-standby"
        )
        self._standby = None
        self.stats = {}
        self.path = Path("/dev/shm") / ("areal-idle-" + uuid.uuid4().hex)
        self.path.mkdir(mode=0o700)
        self.primary.idle_training_state("prepare")
        self.primary.idle_training_state("dram_prepare", str(self.path / "single"))
        self.primary.offload()
        multi_cfg = copy.deepcopy(cfg.actor)
        multi_cfg.backend = "fsdp:d2p1t1"
        # Separate actor processes share devices with parked inference engines.
        # Let HCCL allocate a free device-side port instead of fixed port 16666.
        for spec in multi_cfg.scheduling_spec:
            spec.env_vars["HCCL_NPU_SOCKET_PORT_RANGE"] = "auto"
        alloc = ModelAllocation.from_str(multi_cfg.backend, name="actor-multi")
        self.multi = trainer._create_train_engine(multi_cfg, alloc)
        self.multi.initialize(addr=None, ft_spec=ft_spec, role="actor-multi")
        self.multi.idle_training_state("prepare")
        self.multi.idle_training_state("dram_prepare", str(self.path / "multi"))
        # Exercise first-use export before the timed RL loop. Offload joins the
        # copy before unmapping; the next trained generation overwrites it.
        prewarm_started = time.monotonic()
        self.multi.idle_training_state("dram_start")
        prewarm = self.multi.idle_training_state("dram_wait")
        request_trace.emit(
            "idle_export_prewarmed",
            seconds=time.monotonic() - prewarm_started,
            copy_seconds=prewarm["copy_seconds"],
        )
        self.multi.offload()
        self.primary.onload()
        started_snapshot = self.primary.idle_training_state("dram_start")
        self.snapshot_generation = started_snapshot["generation"]
        self.snapshot_actor = self.primary
        self._device_keys = {
            "single": [
                item["device"] for item in self.primary.idle_handoff("identity")
            ],
            "multi": [item["device"] for item in self.multi.idle_handoff("identity")],
        }
        if (
            len(set(self._device_keys["single"])) != 1
            or len(set(self._device_keys["multi"])) != 2
            or not set(self._device_keys["single"]).issubset(self._device_keys["multi"])
        ):
            raise RuntimeError(
                "Training engines do not share the same device namespace"
            )
        self.helper = None
        if os.getenv("AREAL_IDLE_HELPER", "0") == "1":
            from areal.engine import FSDPPPOActor
            from areal.infra.scheduler import LocalScheduler

            helper_experiment = copy.deepcopy(cfg)
            helper_experiment.trial_name += "-gradient-helper"
            devices = trainer.scheduler.gpu_devices
            if len(devices) != 2:
                raise ValueError(
                    "Independent helper requires exactly two local devices"
                )
            self._helper_scheduler = LocalScheduler(
                gpu_devices=[devices[1]], exp_config=helper_experiment
            )
            helper_cfg = copy.deepcopy(cfg.actor)
            for spec in helper_cfg.scheduling_spec:
                spec.env_vars["HCCL_NPU_SOCKET_PORT_RANGE"] = "auto"
            self.helper = FSDPPPOActor.as_controller(helper_cfg, self._helper_scheduler)
            self.helper.create_process_group(
                parallel_strategy=trainer.actor_alloc.parallel
            )
            self.helper.initialize(addr=None, ft_spec=ft_spec, role="actor-helper")
            # A backward-only helper never owns or steps optimizer state.
            # Keep the previous path available for controlled comparisons.
            if os.getenv("AREAL_IDLE_HELPER_LEAN", "0") != "1":
                self.helper.idle_training_state("prepare")
            self.helper.offload()
            identity = self.helper.idle_handoff("identity")
            if (
                len(identity) != 1
                or identity[0]["device"] in self._device_keys["single"]
            ):
                raise RuntimeError("Helper must run on the independent second device")
            self._helper_costs = []
        self._prepare_standby()
        request_trace.emit(
            "idle_training_initialized", single_world_size=1, multi_world_size=2
        )

    def _prepare_standby(self, destination=None):
        if self._standby is not None:
            raise RuntimeError("Previous standby preparation has not been consumed")
        source = self.snapshot_actor
        destination = self.multi if destination is None else destination
        generation = self.snapshot_generation
        path = self.path / ("single" if source is self.primary else "multi")
        self._standby_destination = destination

        def prepare():
            while True:
                status = source.idle_training_state("dram_poll")
                if status["generation"] != generation:
                    raise RuntimeError(
                        "Standby snapshot generation changed during export"
                    )
                if status["done"]:
                    break
                time.sleep(0.05)
            # Poll returns the DP head's metadata; join every exporting rank before
            # exposing the immutable snapshot to target CPU preparation threads.
            snapshot = source.idle_training_state("dram_wait")
            destination.idle_training_state("dram_cancel_stage")
            destination.idle_training_state(
                f"dram_stage_async:{snapshot['version']}:{generation}", str(path)
            )
            request_trace.emit(
                "idle_standby_started",
                generation=generation,
                scheduler_epoch=snapshot["scheduler"]["last_epoch"],
            )
            return snapshot

        self._standby = self._jobs.submit(prepare)

    def _join_standby(self):
        future = getattr(self, "_standby", None)
        if future is not None:
            snapshot = future.result()
            # The source mmap stays pinned until all CPU readers have finished.
            self._standby_destination.idle_training_state("dram_stage_wait")
            self._standby = None
            return snapshot
        return None

    def _timed_transfer_step(self, direction, operation, call):
        started = time.monotonic()
        result = call()
        request_trace.emit(
            "idle_training_transfer_step",
            direction=direction,
            operation=operation,
            seconds=time.monotonic() - started,
        )
        return result

    def _transfer(self, source, destination, direction):
        with perf_tracer.trace_scope(
            "train.idle_state_" + direction, category=Category.IO
        ):
            if self._standby is None:
                self._prepare_standby(destination)
            if self._standby_destination is not destination:
                raise RuntimeError("Standby destination does not match the transfer")
            # This joins only launch of per-rank preparation, not CPU copies.
            snapshot = self._standby.result()
            before = self._timed_transfer_step(
                direction,
                "source_inspect",
                lambda: source.idle_training_state("inspect"),
            )
            if snapshot["scheduler"]["last_epoch"] != before["scheduler_epoch"]:
                raise RuntimeError("The DRAM snapshot is not the latest training state")
            root = self.path / ("transfer-" + uuid.uuid4().hex)
            root.mkdir()
            config = dict(
                token=root.name,
                direction=direction,
                snapshot_generation=self.snapshot_generation,
                version=before["version"],
                scheduler_epoch=before["scheduler_epoch"],
                source_devices=self._device_keys[
                    "single" if source is self.primary else "multi"
                ],
            )
            (root / "transfer.json").write_text(json.dumps(config))
            # Each target rank waits only for its own copy and device release.
            # Inference release has already been acknowledged before this point.
            restored = self._jobs.submit(destination.idle_handoff, "restore", str(root))
            try:
                source.idle_handoff("release", str(root))
                restored.result()
            except BaseException:
                (root / "failed.json").touch()
                try:
                    restored.result()
                except BaseException:
                    pass
                raise
            self._standby = None
            after = self._timed_transfer_step(
                direction,
                "destination_inspect",
                lambda: destination.idle_training_state("inspect"),
            )
            for key in (
                "version",
                "scheduler_epoch",
                "optimizer_step_min",
                "optimizer_step_max",
                "sampled_state_sha256",
            ):
                if before[key] != after[key]:
                    raise RuntimeError(
                        f"Training state handoff changed {key}: {before} -> {after}"
                    )
            request_trace.emit(
                "idle_training_state_transferred",
                direction=direction,
                before=before,
                after=after,
                token=root.name,
            )

    def ppo_update(self, actor, data, batch):
        """Accumulate one original optimizer minibatch across safe RPC boundaries."""
        if os.getenv("AREAL_IDLE_MID_BATCH", "0") != "1" or actor is self.multi:
            actor.ppo_update(data)
            return actor
        if self.trainer.config.actor.ppo_n_minibatches != 1:
            raise ValueError("Mid-batch expansion requires one optimizer minibatch")
        from areal.engine.idle_plan import segment_bounds

        token = f"{self.path.name}-batch-{batch}"
        plan_path = str(self.path / f"plan-{batch}")
        results = actor._custom_function_call(
            "idle_prepare_plan", data, plan_path, token, rpc_meta={"broadcast": True}
        )
        if not results or any(item != results[0] for item in results):
            raise RuntimeError("Inconsistent preserved-plan metadata")
        meta = results[0]
        denominator = meta["denominator"]
        counts = meta["answer_counts"]
        work = meta["work_tokens"]
        segments = segment_bounds(
            counts, int(os.getenv("AREAL_IDLE_CHUNK_ANSWERS", "32"))
        )
        request_trace.emit(
            "idle_backward_plan",
            batch=batch,
            groups=len(data),
            answers=sum(counts),
            microbatches=len(counts),
            segments=len(segments),
            preserved_packing=True,
        )
        prefix_path = None
        first = True
        for start, stop in segments:
            # Once expanded, no further handoff boundary is needed. Preserve the
            # original rank-local microbatch sequence in one remaining suffix.
            if actor is self.multi:
                stop = len(counts)
            request_trace.emit(
                "idle_backward_segment_begin",
                batch=batch,
                start=start,
                stop=stop,
                world_size=2 if actor is self.multi else 1,
            )
            segment_started = time.monotonic()
            actor.ppo_update(
                data,
                accumulation={
                    "denominator": denominator,
                    "first": first,
                    "last": stop == len(counts),
                    "plan_path": plan_path,
                    "start_mb": start,
                    "stop_mb": stop,
                    "skip_statistics": not first,
                    "prefix_path": prefix_path,
                    "token": token,
                },
            )
            segment_seconds = time.monotonic() - segment_started
            # Final segments include optimizer and gradient import overhead. Keep
            # that overhead in the estimate rather than claiming ideal DP speedup.
            segment_work = work[start:stop]
            if actor is self.multi:
                from areal.trainer.idle_budget import dual_work

                measured_work = dual_work(segment_work)
            else:
                measured_work = sum(segment_work)
            self._switch_budget.observe(
                batch, 2 if actor is self.multi else 1, measured_work, segment_seconds
            )
            first = False
            request_trace.emit(
                "idle_backward_segment_end",
                batch=batch,
                start=start,
                stop=stop,
                world_size=2 if actor is self.multi else 1,
            )
            if stop == len(counts):
                break
            if os.getenv("AREAL_IDLE_SWITCH_BUDGET", "1") == "1":
                decision = self._switch_budget.decide(
                    batch,
                    work[stop:],
                    self._entry_budget.cost_samples(batch),
                    terminal=batch == self.trainer.config.total_train_steps,
                )
                request_trace.emit(
                    "idle_training_switch_budget",
                    batch=batch,
                    consumed_microbatches=stop,
                    **decision,
                )
                if (
                    decision["accepted"]
                    and self.helper is not None
                    and os.getenv("AREAL_IDLE_COMPARE_HELPER", "0") == "1"
                ):
                    from areal.trainer.idle_budget import prefer_helper

                    helper = self._helper_prediction(batch, work[stop:])
                    selected = prefer_helper(decision, helper)
                    request_trace.emit(
                        "idle_execution_choice",
                        batch=batch,
                        consumed_microbatches=stop,
                        expansion_net_seconds=decision.get("net_seconds"),
                        helper_net_seconds=helper.get("net_seconds"),
                        preferred="helper" if selected else "expansion",
                    )
                    if selected and self._try_helper(
                        actor,
                        data,
                        batch,
                        stop,
                        work,
                        counts,
                        plan_path,
                        token,
                        denominator,
                    ):
                        break
                if not decision["accepted"]:
                    if self._try_helper(
                        actor,
                        data,
                        batch,
                        stop,
                        work,
                        counts,
                        plan_path,
                        token,
                        denominator,
                    ):
                        break
                    continue
            rollout = self.trainer.rollout
            if not rollout.dispatcher.pause_if_generation_idle():
                continue
            # All backward work for this segment has returned; no optimizer step
            # has occurred. Save gradients before the standard offload clears them.
            started = time.monotonic()
            with perf_tracer.trace_scope("train.idle_switch_in", category=Category.IO):
                prefix_path = str(self.path / f"gradients-{batch}")
                actor._custom_function_call("idle_export_gradients", prefix_path, token)
                # Do not overwrite prefix statistics with suffix-only means.
                # Keep each segment explicitly named until sufficient-statistic
                # aggregation across independent engines is implemented.
                self._prefix_stats = actor._custom_function_call("export_stats")
                self._activate_multi(batch)
            self._switch_budget.cost(batch, "in", time.monotonic() - started)
            self._mid_expanded = True
            actor = self.multi
            first = True
            request_trace.emit(
                "idle_training_mid_batch_selected",
                batch=batch,
                consumed=sum(counts[:stop]),
                remaining=sum(counts[stop:]),
                consumed_microbatches=stop,
                switch_seconds=time.monotonic() - started,
            )
        if prefix_path is not None:
            shutil.rmtree(prefix_path)
        shutil.rmtree(plan_path)
        return actor

    def _helper_prediction(self, batch, remaining):
        from areal.trainer.idle_budget import helper_plan

        rates = [
            s for s in self._switch_budget.samples if s[1] == 1 and s[0] >= batch - 3
        ]
        if not rates:
            return dict(accepted=False, reason="insufficient_helper_samples")
        rate = sum(s[3] for s in rates) / sum(s[2] for s in rates)
        costs = [s for s in self._helper_costs if s[0] >= batch - 3]
        restore = max((s[1] for s in costs), default=3.0)
        merge = max((s[2] for s in costs), default=3.0)
        return dict(
            cost_source="measured" if costs else "prior",
            **helper_plan(remaining, rate, restore, merge),
        )

    def _try_helper(
        self, actor, data, batch, stop, work, counts, plan_path, token, denominator
    ):
        """Keep the primary optimizer; delegate an exclusive packed interval."""
        if self.helper is None:
            return False
        decision = self._helper_prediction(batch, work[stop:])
        if os.getenv("AREAL_IDLE_HELPER_VALIDATE", "0") == "1":
            if os.getenv("AREAL_IDLE_FULL_VERIFY", "0") != "1":
                raise ValueError("Forced helper selection requires full verification")
            if len(work) - stop >= 2:
                decision.update(
                    accepted=True,
                    reason="correctness_probe",
                    helper_microbatches=(len(work) - stop) // 2,
                )
        request_trace.emit(
            "idle_helper_budget", batch=batch, consumed_microbatches=stop, **decision
        )
        if not decision["accepted"]:
            return False
        rollout = self.trainer.rollout
        if not rollout.dispatcher.pause_if_generation_idle():
            return False
        cut = stop + decision["helper_microbatches"]
        path = str(self.path / f"helper-gradients-{batch}")
        started = time.monotonic()
        rollout.wait_shared_publication()
        rollout.wait_running_rebalance()
        from areal.infra.utils.concurrent import call_maybe_async

        rollout.pause()
        call_maybe_async(rollout.pause_generation)
        rollout._collective_rpc("colocation_memory", action="offload")
        source = self.snapshot_actor
        source.idle_training_state("dram_wait")
        source_path = str(self.path / ("single" if source is self.primary else "multi"))
        version = self.primary.idle_training_state("dram_progress")["version"]

        def helper_job():
            self.helper._custom_function_call(
                "idle_helper_stage", source_path, version, self.snapshot_generation
            )
            self.helper.onload()
            prepared = time.monotonic() - started
            self.helper.ppo_update(
                data,
                accumulation=dict(
                    denominator=denominator,
                    first=True,
                    last=False,
                    plan_path=plan_path,
                    start_mb=stop,
                    stop_mb=cut,
                    skip_statistics=True,
                    prefix_path=None,
                    token=token,
                ),
            )
            export_started = time.monotonic()
            self.helper._custom_function_call("idle_export_gradients", path, token)
            export_seconds = time.monotonic() - export_started
            release_started = time.monotonic()
            self.helper.colocation_memory("finish_train")
            self.helper.colocation_memory("park_cache")
            self.helper.offload()
            # The primary publishes via HCCL to this replica. Restore only the
            # fixed side; the shared side still belongs to primary training.
            rollout._group_rpc(
                rollout.fixed_ranks, "colocation_memory", action="restore"
            )
            return prepared, export_seconds, time.monotonic() - release_started

        request_trace.emit(
            "idle_helper_begin",
            batch=batch,
            token=token,
            helper_start=stop,
            helper_stop=cut,
            primary_start=cut,
            primary_stop=len(work),
            denominator=denominator,
        )
        future = self._jobs.submit(helper_job)
        try:
            # Keep one final item for the sole optimizer step after helper import.
            if cut < len(work) - 1:
                actor.ppo_update(
                    data,
                    accumulation=dict(
                        denominator=denominator,
                        first=False,
                        last=False,
                        plan_path=plan_path,
                        start_mb=cut,
                        stop_mb=len(work) - 1,
                        skip_statistics=True,
                        prefix_path=None,
                        token=token,
                    ),
                )
            prepared, exported, released = future.result()
            merge_started = time.monotonic()
            actor.ppo_update(
                data,
                accumulation=dict(
                    denominator=denominator,
                    first=False,
                    last=True,
                    plan_path=plan_path,
                    start_mb=len(work) - 1,
                    stop_mb=len(work),
                    skip_statistics=True,
                    prefix_path=path,
                    token=token,
                ),
            )
            completion = time.monotonic() - merge_started
            imported = float(actor._custom_function_call("idle_helper_import_seconds"))
            self._helper_costs.append((batch, prepared, exported + imported + released))
            request_trace.emit(
                "idle_helper_end",
                batch=batch,
                token=token,
                seconds=time.monotonic() - started,
                prepared_seconds=prepared,
                export_seconds=exported,
                import_seconds=imported,
                release_seconds=released,
                completion_seconds=completion,
            )
        finally:
            future.result()
        self._helper_used = True
        shutil.rmtree(path)
        return True

    def _activate_multi(self, batch):
        rollout = self.trainer.rollout
        rollout.wait_shared_publication()
        rollout.wait_running_rebalance()
        rollout.pause()
        from areal.infra.utils.concurrent import call_maybe_async

        call_maybe_async(rollout.pause_generation)
        rollout._collective_rpc("colocation_memory", action="offload")
        self._transfer(self.primary, self.multi, "single_to_multi")
        self.active = True

    def arm(self, batch):
        """Prepare the selected world behind the previous step's ownership fence."""
        from areal.trainer.training_prefetch import TrainingRestorePrefetch

        if self._entry_prefetch is not None:
            raise RuntimeError("Previous entry preparation was not consumed")
        rollout = self.trainer.rollout
        target = rollout.training_prefetch_target()
        self._entry_actor = None

        def fail(exc):
            rollout.dispatcher.pause()
            rollout.dispatcher.fail_fast(exc)

        self._entry_prefetch = TrainingRestorePrefetch(
            self.trainer._step_tail_executor,
            lambda: rollout.training_prefetch_ready(target),
            lambda: self._prepare_entry(batch),
            fail,
            batch,
        )
        request_trace.emit("idle_training_entry_armed", batch=batch, target=target)

    def _prepare_entry(self, batch):
        rollout = self.trainer.rollout
        rollout.wait_shared_publication()
        rollout.wait_running_rebalance()
        snapshot = rollout.dispatcher.generation_work_snapshot()
        decision = self._entry_budget.decide(
            batch,
            sum(snapshot.values()),
            terminal=batch == self.trainer.config.total_train_steps,
        )
        validation = os.getenv("AREAL_IDLE_ENTRY_VALIDATE_WORLDS", "")
        if validation:
            if os.getenv("AREAL_IDLE_FULL_VERIFY", "0") != "1":
                raise ValueError(
                    "Forced entry worlds require full correctness verification"
                )
            worlds = [int(value) for value in validation.split(",")]
            if any(value not in (1, 2) for value in worlds):
                raise ValueError("Invalid validation world sequence")
            if batch <= len(worlds):
                decision = dict(
                    world_size=worlds[batch - 1], reason="correctness_sequence"
                )
        request_trace.emit(
            "idle_training_entry_decision", batch=batch, generation=snapshot, **decision
        )
        started = time.monotonic()
        use_multi = (
            decision["world_size"] == 2
            and rollout.dispatcher.pause_if_generation_idle()
        )
        if use_multi:
            # Park inference and restore DP2 directly from its prepared DRAM
            # backup. The primary can still be completely unmapped here.
            with perf_tracer.trace_scope(
                "train.idle_entry_multi", category=Category.IO
            ):
                rollout.pause()
                from areal.infra.utils.concurrent import call_maybe_async

                call_maybe_async(rollout.pause_generation)
                rollout._collective_rpc("colocation_memory", action="offload")
                self._restore_multi_from_dram()
            self.active = True
            self._entry_actor = self.multi
        else:
            with perf_tracer.trace_scope(
                "train.idle_entry_single", category=Category.IO
            ):
                rollout.borrow_for_training()
                self._restore_single_from_dram()
            self._entry_actor = self.primary
        request_trace.emit(
            "idle_training_selected",
            batch=batch,
            world_size=2 if use_multi else 1,
            reason=decision["reason"]
            if use_multi or decision["world_size"] == 1
            else "idle_fence_changed",
            preparation_seconds=time.monotonic() - started,
            direct_from_inference=True,
        )

    def _entry_snapshot(self, destination):
        source = self.snapshot_actor
        progress = source.idle_training_state("dram_progress")
        snapshot = source.idle_training_state("dram_wait")
        if (
            snapshot["scheduler"]["last_epoch"] != progress["scheduler_epoch"]
            or progress["snapshot_generation"] != self.snapshot_generation
        ):
            raise RuntimeError("Entry DRAM snapshot is not the latest optimizer state")
        if source is destination:
            # The inactive engine can finish its DRAM update in parallel with
            # this restore. Join before the source snapshot may be overwritten.
            return progress, False
        if self._standby is None:
            self._prepare_standby(destination)
        if self._standby_destination is not destination:
            self._join_standby()
            self._prepare_standby(destination)
        self._standby.result()
        destination.idle_training_state("dram_stage_wait")
        destination.idle_training_state(f"dram_relabel:{progress['version']}")
        return progress, True

    def _restore_single_from_dram(self):
        progress, imported = self._entry_snapshot(self.primary)
        self.primary.set_version(progress["version"])
        self.trainer._memory_policy.train(resident=True)
        self.primary.idle_training_state(
            "dram_verify" if imported else "dram_verify_own"
        )
        if imported:
            self.primary.idle_training_state("dram_clear_stage")
            self._standby = None
        # Keep DP2's backup prepared for an optional later safe-boundary change.
        if self._standby is None:
            self._prepare_standby(self.multi)

    def _restore_multi_from_dram(self):
        progress, imported = self._entry_snapshot(self.multi)
        if not imported:
            self.multi.set_version(progress["version"])
            self.multi.onload()
            self.multi.idle_training_state("dram_verify_own")
            request_trace.emit(
                "idle_training_direct_restored",
                source_parked=progress["is_offload"],
                world_size=2,
                version=progress["version"],
                own_backing=True,
            )
            return
        root = self.path / ("entry-" + uuid.uuid4().hex)
        root.mkdir()
        config = dict(
            token=root.name,
            direction="inference_to_multi",
            snapshot_generation=self.snapshot_generation,
            version=progress["version"],
            scheduler_epoch=progress["scheduler_epoch"],
            source_devices=self._device_keys["single"],
        )
        (root / "transfer.json").write_text(json.dumps(config))
        restored = self._jobs.submit(self.multi.idle_handoff, "restore", str(root))
        try:
            self.primary.idle_handoff("release", str(root))
            restored.result()
        except BaseException:
            (root / "failed.json").touch()
            try:
                restored.result()
            except BaseException:
                pass
            raise
        self._standby = None
        after = self.multi.idle_training_state("dram_progress")
        if (
            after["version"] != progress["version"]
            or after["scheduler_epoch"] != progress["scheduler_epoch"]
            or after["is_offload"]
        ):
            raise RuntimeError("Direct entry restored stale training progress")
        request_trace.emit(
            "idle_training_direct_restored",
            token=root.name,
            source_parked=progress["is_offload"],
            version=after["version"],
            scheduler_epoch=after["scheduler_epoch"],
            world_size=2,
        )

    def select(self, batch):
        started = time.monotonic()
        if self._entry_prefetch is None:
            self._prepare_entry(batch)
        else:
            try:
                if not self._entry_prefetch.join(force=True):
                    raise RuntimeError("Entry preparation was cancelled")
            finally:
                self._entry_prefetch = None
        world = 2 if self._entry_actor is self.multi else 1
        self._entry_budget.observe(batch, world, "prepare", time.monotonic() - started)
        if world == 1:
            self.trainer._resume_training_inference()
        self.current = self._entry_actor
        self._entry_world = world
        self._entry_compute_started = time.monotonic()
        return self._entry_actor

    def cancel_entry(self):
        prefetch = self._entry_prefetch
        if prefetch is not None:
            prefetch.cancel()
            try:
                prefetch.join()
            finally:
                self._entry_prefetch = None

    def finish(self, batch):
        if (
            hasattr(self, "_entry_compute_started")
            and not getattr(self, "_mid_expanded", False)
            and not getattr(self, "_helper_used", False)
        ):
            self._entry_budget.observe(
                batch,
                self._entry_world,
                "compute",
                time.monotonic() - self._entry_compute_started,
            )
        self._mid_expanded = False
        self._helper_used = False
        self._join_standby()
        self.current = self.multi if self.active else self.primary
        started_snapshot = self.current.idle_training_state("dram_start")
        self.snapshot_generation = started_snapshot["generation"]
        self.snapshot_actor = self.current
        request_trace.emit("idle_dram_sync_started", batch=batch, dual=self.active)
        destination = self.primary if self.active else self.multi
        self._prepare_standby(destination)
        # Preserve the active engine for checkpoint/publication. There is no
        # mandatory DP2 -> DP1 device restoration after an optimizer update.
        prefix = getattr(self, "_prefix_stats", None)
        if prefix is not None:
            suffix = self.current._custom_function_call("export_stats")
            self.stats = {"idle_prefix/" + k: v for k, v in prefix.items()}
            self.stats.update({"idle_suffix/" + k: v for k, v in suffix.items()})
            self._prefix_stats = None

    def publish_multi(self, version):
        """Release DP2 once, then publish its CPU snapshot to both replicas."""
        from dataclasses import replace

        from areal.infra.utils.concurrent import call_maybe_async

        rollout = self.trainer.rollout
        started = time.monotonic()
        meta = self.trainer.weight_update_meta.with_version(version)
        meta = replace(
            meta,
            type="disk",
            npu_share_snapshot=True,
            clear_checkpoint_after_load=False,
        )
        rollout.wait_shared_publication()
        rollout.wait_running_rebalance()
        rollout.pause()
        call_maybe_async(rollout.pause_generation)
        self.current._custom_function_call(
            "idle_publication_snapshot", meta.path, version
        )
        self.current.set_version(version)
        stats = self.current.export_stats()
        self.current.colocation_memory("finish_train")
        self.current.colocation_memory("park_cache")
        self.current.offload()
        # Both inference pools were parked during DP2 computation. Neither
        # training engine is restored for publication.
        async_tail = os.getenv("AREAL_IDLE_ASYNC_TAIL", "1") == "1"
        if async_tail:
            rollout._group_rpc(
                rollout.fixed_ranks, "colocation_memory", action="restore"
            )
        else:
            rollout._collective_rpc("colocation_memory", action="restore")
        rollout.prepare_fixed_publication()
        rollout._group_rpc(rollout.fixed_ranks, "update_weights_from_disk", meta=meta)
        rollout.publish_fixed_version(version)
        if async_tail:
            rollout.load_shared_snapshot_async(meta, restore_memory=True)
        else:
            rollout.load_shared_snapshot_async(meta)
        self._entry_budget.observe(version, 2, "return", time.monotonic() - started)
        request_trace.emit(
            "idle_training_dram_publication",
            version=version,
            seconds=time.monotonic() - started,
            publisher_world_size=2,
            primary_hbm_restored=False,
        )
        self.active = False
        self.trainer._return_shared_inference = True
        return stats

    def close(self):
        self.cancel_entry()
        self._join_standby()
        if self.helper is not None:
            self.helper.destroy()
        self._jobs.shutdown(wait=True)
        self.primary.idle_training_state("dram_wait")
        self.multi.idle_training_state("dram_wait")
        shutil.rmtree(self.path)

    def take_stats(self):
        result, self.stats = self.stats, {}
        return result


def idle_training_enabled():
    return os.getenv("AREAL_IDLE_TRAINING", "0") == "1"
