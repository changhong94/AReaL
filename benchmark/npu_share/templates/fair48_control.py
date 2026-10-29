"""Experiment-only workload/terminal controls, identical for both arms."""

import hashlib
import json
import time
from pathlib import Path
from types import MethodType

from areal.utils import request_trace

SETTINGS = json.loads(Path("/work/settings.json").read_text())


def configure_workload(trainer):
    dispatcher = trainer.rollout.dispatcher
    original_submit = dispatcher.submit_task_input
    submitted = []

    def submit(self, task):
        assert len(submitted) < SETTINGS["questions"]
        encoded = json.dumps(task.data, sort_keys=True, ensure_ascii=False, default=str)
        row = dict(
            task_id=task.task_id,
            data_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
        )
        original_submit(task)
        submitted.append(row)
        request_trace.emit("experiment_task_submitted", **row)

    dispatcher.submit_task_input = MethodType(submit, dispatcher)

    def active_submit_and_wait(self, input_generator, batch_size, dynamic_bs=False):
        if dynamic_bs or batch_size != SETTINGS["batch_size"]:
            raise ValueError(
                "Exact-workload control requires the configured static batch size "
                f"{SETTINGS['batch_size']}, got {batch_size} (dynamic={dynamic_bs})"
            )
        results = []
        deadline = time.monotonic() + min(SETTINGS["timeout"], 3600)
        while len(results) < batch_size:
            if time.monotonic() > deadline:
                raise TimeoutError("Bounded experiment batch timed out")
            with self._input_cv:
                pending_inputs = len(self._pending_inputs)
            capacity = min(
                self.staleness_manager.get_pending_limit() - pending_inputs,
                # Preserve the usual reserve, but leave one batch admissible
                # when the queue itself holds only one batch (concurrency=1).
                self.runner.max_queue_size
                - self.runner.get_input_queue_size()
                - min(batch_size, max(0, self.runner.max_queue_size - batch_size)),
                SETTINGS["questions"] - len(submitted),
            )
            assert self.runner.max_queue_size >= batch_size
            for _ in range(min(batch_size, max(0, capacity))):
                self.submit_task_input(next(input_generator))
            try:
                arrived = self.wait_results(count=batch_size - len(results), timeout=1)
            except TimeoutError:
                continue
            if any(r is None for r in arrived):
                raise RuntimeError("Rejected question in exact-workload experiment")
            results.extend(arrived)
        return results

    dispatcher.active_submit_and_wait = MethodType(active_submit_and_wait, dispatcher)
    # Fixed path otherwise publishes an unused final version. Shared path already skips it.
    if not trainer.config.npu_share:
        original_publish = trainer._publish_weights
        original_finish = trainer._finish_step

        def publish(version):
            if version == SETTINGS["batches"]:
                trainer.actor.set_version(version)
                stats = trainer.actor.export_stats()
                request_trace.emit("terminal_publication_skipped", version=version)
                return stats
            return original_publish(version)

        def finish(**kwargs):
            if kwargs["global_step"] == SETTINGS["batches"] - 1:
                kwargs["resume_rollout"] = False
            return original_finish(**kwargs)

        trainer._publish_weights = publish
        trainer._finish_step = finish
    return submitted
