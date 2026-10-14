# SPDX-License-Identifier: Apache-2.0
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.utils import request_trace


def test_trace_disabled_does_not_create_files(monkeypatch, tmp_path):
    monkeypatch.delenv("AREAL_REQUEST_TRACE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    request_trace.emit("start", request_id="r")
    assert not list(tmp_path.iterdir())


def test_trace_keeps_repeated_and_unfinished_events(monkeypatch, tmp_path):
    monkeypatch.setenv("AREAL_REQUEST_TRACE_DIR", str(tmp_path))
    for event in ["answer_start", "native_pause", "native_resume", "native_pause"]:
        request_trace.emit(event, request_id="r")
    records = [
        json.loads(x) for x in next(tmp_path.glob("*.jsonl")).read_text().splitlines()
    ]
    assert [x["event"] for x in records] == [
        "answer_start",
        "native_pause",
        "native_resume",
        "native_pause",
    ]
    assert all(x["request_id"] == "r" and x["time"] > 0 for x in records)


def test_device_window_brackets_execution_and_sync(monkeypatch, tmp_path):
    import torch
    from vllm.v1.core.sched.scheduler import Scheduler

    class NPUModelRunner:
        def execute_model(self, *args, **kwargs):
            return None

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.worker.model_runner_v1",
        SimpleNamespace(NPUModelRunner=NPUModelRunner),
    )

    monkeypatch.setenv("AREAL_REQUEST_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("AREAL_NPU_DEVICE_TRACE", "1")
    order = []
    monkeypatch.setattr(torch.npu, "synchronize", lambda: order.append("sync"))
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
    monkeypatch.setattr(Scheduler, "add_request", lambda self, request: None)
    monkeypatch.setattr(
        NPUModelRunner, "execute_model", lambda *a, **k: order.append("execute")
    )
    emit = Mock(side_effect=lambda event, **kw: order.append(event))
    monkeypatch.setattr(request_trace, "emit", emit)
    request_trace.install_device_trace()
    runner = SimpleNamespace()
    output = SimpleNamespace(num_scheduled_tokens={"r1": 10, "r2": 1})
    NPUModelRunner.execute_model(runner, output)
    assert order == ["sync", "device_step_begin", "execute", "sync", "device_step_end"]
    assert emit.call_args_list[0].kwargs["internal_ids"] == ["r1", "r2"]
    assert runner._areal_trace_step == 1
    # Removing the switch also bypasses a wrapper already installed in-process.
    monkeypatch.delenv("AREAL_NPU_DEVICE_TRACE")
    order.clear()
    NPUModelRunner.execute_model(runner, output)
    assert order == ["execute"]


def test_native_id_mapping_preserves_external_id(monkeypatch, tmp_path):
    from vllm.v1.engine.input_processor import InputProcessor

    monkeypatch.setenv("AREAL_REQUEST_TRACE_DIR", str(tmp_path))

    def assign(req):
        req.external_req_id = req.request_id
        req.request_id += "-internal"

    monkeypatch.setattr(InputProcessor, "assign_request_id", staticmethod(assign))
    request_trace.install_request_id_trace()
    request_trace.install_request_id_trace()
    req = SimpleNamespace(request_id="answer-segment")
    InputProcessor.assign_request_id(req)
    row = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert row["internal_id"] == "answer-segment-internal"
    assert row["request_id"] == "answer-segment"


@pytest.mark.parametrize("flag", [None, "0", "false"])
def test_request_logging_does_not_install_device_sync(monkeypatch, tmp_path, flag):
    import torch
    from vllm.v1.core.sched.scheduler import Scheduler

    execute = Mock(return_value="output")

    class NPUModelRunner:
        execute_model = execute

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.worker.model_runner_v1",
        SimpleNamespace(NPUModelRunner=NPUModelRunner),
    )
    monkeypatch.setattr(Scheduler, "add_request", lambda self, request: None)
    monkeypatch.setenv("AREAL_REQUEST_TRACE_DIR", str(tmp_path))
    if flag is None:
        monkeypatch.delenv("AREAL_NPU_DEVICE_TRACE", raising=False)
    else:
        monkeypatch.setenv("AREAL_NPU_DEVICE_TRACE", flag)
    sync = Mock()
    monkeypatch.setattr(torch.npu, "synchronize", sync)
    request_trace.install_device_trace()
    assert NPUModelRunner.execute_model is execute
    assert (
        NPUModelRunner.execute_model(SimpleNamespace(), SimpleNamespace()) == "output"
    )
    sync.assert_not_called()


def test_device_flag_without_trace_directory_is_disabled(monkeypatch):
    import torch

    monkeypatch.setenv("AREAL_NPU_DEVICE_TRACE", "1")
    monkeypatch.delenv("AREAL_REQUEST_TRACE_DIR", raising=False)
    sync = Mock()
    monkeypatch.setattr(torch.npu, "synchronize", sync)
    request_trace.install_device_trace()
    sync.assert_not_called()
