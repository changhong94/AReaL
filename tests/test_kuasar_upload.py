"""Transient upload failures must not drop a question or become zero rewards."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def test_upload_retry(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "bench_kuasar",
        Path(__file__).parents[1] / "benchmark/npu_share/templates/kuasar.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Missing(Exception):
        pass

    exceptions = ModuleType("e2b.exceptions")
    exceptions.FileNotFoundException = Missing
    monkeypatch.setitem(sys.modules, "e2b.exceptions", exceptions)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    from areal.utils import request_trace

    events = []
    monkeypatch.setattr(request_trace, "emit", lambda *a, **kw: events.append((a, kw)))
    calls = []

    def write(*args):
        calls.append(args)
        if len(calls) < 3:
            raise Missing("sandbox not found")

    sandbox = SimpleNamespace(sandbox_id="test", files=SimpleNamespace(write=write))
    module._write_test(sandbox, "assert True", {})
    assert len(calls) == 3 and len(events) == 2

    def missing(*args):
        raise Missing("sandbox not found")

    sandbox.files.write = missing
    with pytest.raises(Missing):
        module._write_test(sandbox, "assert True", {})
    assert events[-1][1]["exhausted"] is True

    def wrong_path(*args):
        raise Missing("guest path missing")

    sandbox.files.write = wrong_path
    count = len(events)
    with pytest.raises(Missing, match="guest path"):
        module._write_test(sandbox, "assert True", {})
    assert len(events) == count
