"""Development-only causal ablations; formal benchmark behavior is unchanged."""

import os

MODES = {
    "baseline",
    "no-first-return",
    "no-training-overlap",
    "both",
    "calibration-return",
}


def validate_modes(value, repetitions):
    modes = value.split(",") if value else ["baseline"] * repetitions
    if len(modes) != repetitions or any(mode not in MODES for mode in modes):
        raise ValueError("Provide one supported shared ablation per repetition")
    return modes


def install(mode, trainer, predictor, emit):
    if mode not in MODES:
        raise ValueError("Unknown development ablation")
    if mode == "calibration-return":
        os.environ["AREAL_RETURN_CALIBRATION_PROBE"] = "1"
    emit("development_ablation", mode=mode)
    if mode in ("no-first-return", "both"):
        original = predictor.estimate_return
        skipped = False

        def estimate(*args, **kwargs):
            nonlocal skipped
            result = original(*args, **kwargs)
            if not skipped and result.get("reason") == "calibration_probe":
                skipped = True
                emit("development_first_probe_skipped", original=dict(result))
                return dict(
                    result, action="keep", reason="development_first_probe_skipped"
                )
            return result

        predictor.estimate_return = estimate
    if mode in ("no-training-overlap", "both"):

        def suppress(self):
            emit("development_training_overlap_suppressed")

        trainer._resume_training_inference = suppress
