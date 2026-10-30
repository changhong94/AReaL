"""Deployment-only return-cost cache; never restores mutable training state."""

import hashlib
import json
import math
import os
from collections import deque
from pathlib import Path


def fingerprint(context):
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def validate(payload, context):
    if payload.get("schema") != 1 or payload.get("fingerprint") != fingerprint(context):
        raise ValueError("Deployment calibration configuration/source mismatch")
    costs = payload.get("costs", [])
    if not costs:
        raise ValueError(
            "Deployment calibration has no completed return/restore sample"
        )
    for sample in costs:
        if set(sample) != {"publish", "restore"} or any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in sample.values()
        ):
            raise ValueError("Invalid deployment return-cost sample")
    return costs


def install(trainer, workspace, emit):
    path = Path(workspace) / "return-calibration.json"
    context_path = Path(workspace) / "return-calibration-context.json"
    if not path.exists() or not trainer.config.npu_share:
        return
    context = json.loads(context_path.read_text())
    payload = json.loads(path.read_text())
    costs = validate(payload, context)
    if trainer.rollout.get_version() != 0:
        raise RuntimeError("Deployment calibration requires initial model version zero")
    if getattr(trainer.rollout, "_return_costs", None):
        raise RuntimeError("Calibration import requires a fresh rollout controller")
    trainer.rollout._return_costs = deque((dict(c) for c in costs), maxlen=3)
    # New online observations still drive the forecast; missing metadata must
    # not manufacture a second calibration run inside the measured loop.
    os.environ["AREAL_RETURN_CALIBRATION_PROBE"] = "0"
    emit(
        "deployment_calibration.loaded",
        samples=len(costs),
        costs=costs,
        fingerprint=payload["fingerprint"],
        reset="fresh_process_original_model",
    )


def export(trainer, workspace, emit):
    workspace = Path(workspace)
    if (
        not (workspace / "return-calibration-export").exists()
        or not trainer.config.npu_share
    ):
        return
    costs = list(getattr(trainer.rollout, "_return_costs", []))
    context = json.loads((workspace / "return-calibration-context.json").read_text())
    payload = dict(
        schema=1,
        fingerprint=fingerprint(context),
        context=context,
        costs=costs,
        source="completed_real_return_and_training_restore",
    )
    validate(payload, context)
    target = workspace / "return-calibration.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(target)
    emit("deployment_calibration.exported", samples=len(costs), costs=costs)
