"""Cost cache validation and fresh-process isolation contracts."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "deployment_calibration",
    Path(__file__).parent / "templates/deployment_calibration.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def payload(context, costs=None):
    return dict(
        schema=1,
        fingerprint=m.fingerprint(context),
        costs=costs if costs is not None else [dict(publish=9.5, restore=0.2)],
    )


def test_rejects_configuration_change():
    with pytest.raises(ValueError, match="mismatch"):
        m.validate(payload(dict(tp=2)), dict(tp=1))


@pytest.mark.parametrize(
    "costs",
    [
        [],
        [dict(publish=0, restore=1)],
        [dict(publish=float("nan"), restore=1)],
        [dict(publish=1)],
        [dict(publish=1, restore=-1)],
    ],
)
def test_rejects_unmeasured_or_invalid_costs(costs):
    with pytest.raises(ValueError):
        m.validate(payload({}, costs), {})


def test_import_only_costs_and_disables_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_RETURN_CALIBRATION_PROBE", "1")
    (tmp_path / "return-calibration-context.json").write_text("{}")
    (tmp_path / "return-calibration.json").write_text(json.dumps(payload({})))
    route = object()
    trainer = SimpleNamespace(
        config=SimpleNamespace(npu_share=True),
        rollout=SimpleNamespace(route=route, version=0, get_version=lambda: 0),
    )
    events = []
    m.install(trainer, tmp_path, lambda *a, **kw: events.append((a, kw)))
    assert list(trainer.rollout._return_costs) == payload({})["costs"]
    assert trainer.rollout.route is route and trainer.rollout.version == 0
    assert m.os.environ["AREAL_RETURN_CALIBRATION_PROBE"] == "0"
    assert len(events) == 1
    with pytest.raises(RuntimeError, match="fresh"):
        m.install(trainer, tmp_path, lambda *a, **kw: None)


def test_export_requires_completed_restore(tmp_path):
    (tmp_path / "return-calibration-export").touch()
    (tmp_path / "return-calibration-context.json").write_text("{}")
    trainer = SimpleNamespace(
        config=SimpleNamespace(npu_share=True), rollout=SimpleNamespace()
    )
    with pytest.raises(ValueError, match="no completed"):
        m.export(trainer, tmp_path, lambda *a, **kw: None)
    assert not (tmp_path / "return-calibration.json").exists()


def test_fixed_arm_does_not_import_or_disable_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_RETURN_CALIBRATION_PROBE", "1")
    (tmp_path / "return-calibration.json").write_text("invalid")
    trainer = SimpleNamespace(config=SimpleNamespace(npu_share=False))
    m.install(trainer, tmp_path, lambda *a, **kw: None)
    assert m.os.environ["AREAL_RETURN_CALIBRATION_PROBE"] == "1"
