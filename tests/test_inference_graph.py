# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from areal.engine.vllm_ext.graph_compat import assert_graph_storage, storage_signature


def test_inplace_weight_update_preserves_graph_storage():
    model = torch.nn.Linear(4, 4)
    runner = SimpleNamespace(model=model, _areal_graph_storage=storage_signature(model))
    with torch.no_grad():
        model.weight.add_(1)
    assert_graph_storage(runner)


def test_replaced_weight_storage_fails_closed():
    model = torch.nn.Linear(4, 4)
    runner = SimpleNamespace(model=model, _areal_graph_storage=storage_signature(model))
    model.weight = torch.nn.Parameter(model.weight.detach().clone())
    with pytest.raises(RuntimeError, match="storage changed"):
        assert_graph_storage(runner)
    assert runner._areal_graph_broken


def test_same_pointer_changed_layout_is_rejected():
    model = torch.nn.Linear(4, 4)
    runner = SimpleNamespace(model=model, _areal_graph_storage=storage_signature(model))
    model.weight.data = model.weight.data.t()
    with pytest.raises(RuntimeError, match="storage changed"):
        assert_graph_storage(runner)


def test_eager_runner_has_no_capture_requirement():
    assert_graph_storage(SimpleNamespace())


def test_capture_maps_only_null_and_refuses_live_recapture():
    from areal.engine.vllm_ext.graph_compat import install_graph_guard

    events = []
    table = SimpleNamespace(
        block_table=SimpleNamespace(cpu=torch.ones(4), gpu=torch.ones(4)),
        slot_mapping=SimpleNamespace(cpu=torch.ones(4), gpu=torch.ones(4)),
    )

    class Runner:
        def __init__(self):
            self.model = torch.nn.Linear(4, 4)
            self.model_config = SimpleNamespace(enforce_eager=False)
            self.requests = {}
            self.input_batch = SimpleNamespace(
                block_table=SimpleNamespace(block_tables=[table])
            )
            self._areal_dynamic_kv = SimpleNamespace(
                live={0},
                swapped={},
                tensors=[],
                backend=SimpleNamespace(synchronize=lambda: events.append("sync")),
                reconcile=lambda ids, trim: events.append((ids, trim)),
            )

        def capture_model(self):
            assert torch.equal(table.block_table.gpu, torch.zeros(4))
            assert torch.equal(table.slot_mapping.gpu, torch.full((4,), -1))
            events.append("capture")
            return 42

        def execute_model(self):
            return 7

        def _determine_batch_execution_and_padding(self):
            return (SimpleNamespace(name="NONE"),)

    install_graph_guard(Runner)
    runner = Runner()
    assert runner.capture_model() == 42
    assert events == ["sync", ({0}, True), "sync", "capture", "sync"]
    assert_graph_storage(runner)
    runner.requests["live"] = object()
    with pytest.raises(RuntimeError, match="idle"):
        runner.capture_model()
    runner._areal_graph_broken = True
    with pytest.raises(RuntimeError, match="invalid"):
        runner.execute_model()


def test_graph_server_queue_override_does_not_modify_training_environment():
    from areal.engine.vllm_remote import VLLMBackend

    original = {"TASK_QUEUE_ENABLE": "2"}
    assert (
        VLLMBackend.build_server_env(original, {"enforce_eager": False})[
            "TASK_QUEUE_ENABLE"
        ]
        == "1"
    )
    assert (
        VLLMBackend.build_server_env(original, {"enforce_eager": True})[
            "TASK_QUEUE_ENABLE"
        ]
        == "2"
    )
    assert original == {"TASK_QUEUE_ENABLE": "2"}


def test_graph_update_propagates_remote_tp_storage_failure(monkeypatch):
    import sys

    import torch.distributed as dist

    from areal.engine.vllm_ext.graph_compat import assert_graph_storage_collective

    group = SimpleNamespace(world_size=2, cpu_group=object())
    monkeypatch.setitem(
        sys.modules, "vllm.distributed", SimpleNamespace(get_tp_group=lambda: group)
    )

    def gather(errors, local, group):
        errors[:] = [local, "peer storage changed"]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    runner = SimpleNamespace(
        model=torch.nn.Linear(4, 4), model_config=SimpleNamespace(enforce_eager=False)
    )
    runner._areal_graph_storage = storage_signature(runner.model)
    with pytest.raises(RuntimeError, match="TP graph storage"):
        assert_graph_storage_collective(runner)
    assert runner._areal_graph_broken


@pytest.mark.parametrize(
    "tokens,sequences", [(2048, 64), (2048, 512), (1000, 32), (3, 1)]
)
def test_full_graph_capture_covers_prefill_limit(tokens, sequences):
    from areal.engine.vllm_ext.graph_compat import full_graph_config

    config = full_graph_config(tokens, sequences)
    sizes = config["cudagraph_capture_sizes"]
    assert config["cudagraph_mode"] == "FULL"
    assert sizes == sorted(set(sizes))
    assert min(sizes) >= 1
    assert max(sizes) == tokens
    assert len(sizes) < 100


def test_full_graph_capture_rejects_invalid_limits():
    from areal.engine.vllm_ext.graph_compat import full_graph_config

    with pytest.raises(ValueError):
        full_graph_config(0, 16)


def test_graph_dispatch_distinguishes_prefill_decode_and_eager(monkeypatch):
    import sys

    from areal.engine.vllm_ext.graph_compat import install_graph_guard

    messages = []
    monkeypatch.setitem(
        sys.modules,
        "vllm.logger",
        SimpleNamespace(
            init_logger=lambda _: SimpleNamespace(
                info=lambda *args: messages.append(args)
            )
        ),
    )

    class Runner:
        requests = {"request": object()}
        mode = "FULL"

        def capture_model(self):
            pass

        def execute_model(self):
            pass

        def _determine_batch_execution_and_padding(self, **kwargs):
            return (SimpleNamespace(name=self.mode),)

    install_graph_guard(Runner)
    runner = Runner()
    for query in [1, 1, 128, 128]:
        runner._determine_batch_execution_and_padding(max_num_scheduled_tokens=query)
    runner.mode = "NONE"
    runner._determine_batch_execution_and_padding(max_num_scheduled_tokens=2048)
    assert [(m[1], m[3]) for m in messages] == [
        ("DECODE", "FULL"),
        ("PREFILL_MIXED", "FULL"),
        ("PREFILL_MIXED", "NONE"),
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_graph_probe_rejects_nonfinite_probabilities(value):
    from benchmark.npu_share.kv_handoff_probe import validate_logprobs

    with pytest.raises(RuntimeError, match="non-finite"):
        validate_logprobs(
            {"choices": [{"logprobs": {"content": [{"logprob": value}]}}]}
        )
