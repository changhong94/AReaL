"""Transient upload failures must not drop a question or become zero rewards."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("failure", ["missing", "activation"])
def test_upload_retry(monkeypatch, failure):
    spec = importlib.util.spec_from_file_location(
        "bench_kuasar",
        Path(__file__).parents[1] / "benchmark/npu_share/templates/kuasar.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Missing(Exception):
        pass

    class ActivationTimeout(Exception):
        pass

    error = Missing if failure == "missing" else ActivationTimeout
    message = (
        "sandbox not found"
        if failure == "missing"
        else "sandbox activation failed\n: SDK timeout hint"
    )

    exceptions = ModuleType("e2b.exceptions")
    exceptions.FileNotFoundException = Missing
    exceptions.TimeoutException = ActivationTimeout
    monkeypatch.setitem(sys.modules, "e2b.exceptions", exceptions)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    # Upload retry has no dependency on training engines or accelerator imports.
    request_trace = ModuleType("areal.utils.request_trace")
    request_trace.emit = lambda *args, **kwargs: None
    utils = ModuleType("areal.utils")
    utils.request_trace = request_trace
    monkeypatch.setitem(sys.modules, "areal.utils", utils)

    events = []
    monkeypatch.setattr(request_trace, "emit", lambda *a, **kw: events.append((a, kw)))
    calls = []

    def write(*args):
        calls.append(args)
        if len(calls) < 3:
            raise error(message)

    sandbox = SimpleNamespace(sandbox_id="test", files=SimpleNamespace(write=write))
    module._write_test(sandbox, "assert True", {})
    assert len(calls) == 3 and len(events) == 2

    def missing(*args):
        raise error(message)

    sandbox.files.write = missing
    with pytest.raises(error):
        module._write_test(sandbox, "assert True", {})
    assert events[-1][1]["exhausted"] is True

    def wrong_path(*args):
        raise Missing("guest path missing")

    sandbox.files.write = wrong_path
    count = len(events)
    with pytest.raises(Missing, match="guest path"):
        module._write_test(sandbox, "assert True", {})
    assert len(events) == count

    def unrelated_timeout(*args):
        raise ActivationTimeout("request timed out")

    sandbox.files.write = unrelated_timeout
    count = len(events)
    with pytest.raises(ActivationTimeout, match="request timed out"):
        module._write_test(sandbox, "assert True", {})
    assert len(events) == count
