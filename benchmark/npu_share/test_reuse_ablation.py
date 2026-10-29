"""Verify isolated development controls preserve defaults and later probes."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "reuse_ablation", Path(__file__).parent / "templates/reuse_ablation.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize("mode", sorted(m.MODES))
def test_ablation_isolates_target_and_keeps_later_probe(mode, monkeypatch):
    monkeypatch.setenv("AREAL_RETURN_CALIBRATION_PROBE", "0")
    calls = []

    class Trainer:
        def _resume_training_inference(self):
            calls.append("overlap")

    predictor = SimpleNamespace(
        estimate_return=lambda: dict(action="return", reason="calibration_probe")
    )
    m.install(mode, Trainer, predictor, lambda *args, **kwargs: None)
    assert predictor.estimate_return()["action"] == (
        "keep" if mode in ("both", "no-first-return") else "return"
    )
    assert predictor.estimate_return()["action"] == "return"
    Trainer()._resume_training_inference()
    assert bool(calls) == (
        mode in ("baseline", "no-first-return", "calibration-return")
    )


def test_modes_reject_invalid_or_mismatched_matrix():
    assert m.validate_modes(None, 2) == ["baseline", "baseline"]
    with pytest.raises(ValueError):
        m.validate_modes("both", 2)
    with pytest.raises(ValueError):
        m.validate_modes("typo", 1)


def test_calibration_return_changes_only_probe_flag(monkeypatch):
    import os

    monkeypatch.setenv("AREAL_RETURN_CALIBRATION_PROBE", "0")
    before = dict(os.environ)
    trainer = type("Trainer", (), {})
    predictor = SimpleNamespace()
    m.install("calibration-return", trainer, predictor, lambda *a, **kw: None)
    expected = dict(before, AREAL_RETURN_CALIBRATION_PROBE="1")
    assert dict(os.environ) == expected
    assert not vars(predictor)
    assert not hasattr(trainer, "_resume_training_inference")
