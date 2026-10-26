# SPDX-License-Identifier: Apache-2.0
"""Execute actual engine routing methods without importing the NPU runtime."""

import ast
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest


@pytest.fixture
def methods():
    path = Path(__file__).parents[1] / "areal/engine/megatron_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronEngine"
    )
    selected = {
        "update_weights",
        "_update_weights_from_distributed",
        "_update_weights_from_disk_stage",
        "wait_shared_snapshot",
        "_ensure_ready",
        "optimizer_step",
        "release_memory_occupation",
        "resume_memory_occupation",
        "offload",
        "onload",
    }
    nodes = []
    for node in engine.body:
        if isinstance(node, ast.FunctionDef) and node.name in selected:
            node.decorator_list = []
            nodes.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    namespace = {
        "dist": SimpleNamespace(get_rank=lambda: 0, barrier=Mock()),
        "current_platform": SimpleNamespace(synchronize=Mock()),
        "nullcontext": nullcontext,
        "Future": Future,
        "is_tms_enabled": lambda: False,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("snapshot", [False, True])
def test_fixed_publication_does_not_pause_other_replica(methods, snapshot):
    publisher = Mock()
    engine = SimpleNamespace(
        _shared_publisher=Mock(return_value=publisher),
        rollout_engine=Mock(),
        cpu_group=object(),
    )
    meta = SimpleNamespace(npu_share_fixed=True, npu_share_snapshot=snapshot)
    methods["_update_weights_from_distributed"](engine, meta)
    publisher.publish_fixed.assert_called_once_with(meta)
    assert methods["dist"].barrier.call_args_list == [
        call(group=engine.cpu_group),
        call(group=engine.cpu_group),
    ]
    engine.rollout_engine.pause_generation.assert_not_called()
    engine.rollout_engine.continue_generation.assert_not_called()
    methods["current_platform"].synchronize.assert_not_called()


def test_nonshared_publication_preserves_original_pause_protocol(methods):
    engine = SimpleNamespace(
        weight_update_master_addr="localhost",
        weight_update_master_port=1,
        weight_update_group_name="test",
        rollout_engine=Mock(),
        cpu_group=None,
        bridge_cls="mbridge",
        config=SimpleNamespace(use_lora=False),
        _update_weights_via_registry=Mock(),
        _shared_publisher=Mock(),
    )
    meta = SimpleNamespace(npu_share_fixed=False, npu_share_snapshot=False)
    methods["_update_weights_from_distributed"](engine, meta)
    engine.rollout_engine.pause_generation.assert_called_once()
    engine._update_weights_via_registry.assert_called_once_with(meta)
    engine.rollout_engine.continue_generation.assert_called_once()
    engine._shared_publisher.assert_not_called()
    methods["current_platform"].synchronize.assert_called_once()


@pytest.mark.parametrize("shared", [False, True])
def test_initial_disk_export_routes_without_changing_ordinary_hf_save(methods, shared):
    publisher = Mock()
    engine = SimpleNamespace(
        _shared_publisher=Mock(return_value=publisher),
        _save_model_to_hf=Mock(),
        tokenizer=object(),
        processor=None,
        cpu_group=None,
        logger=Mock(),
    )
    meta = SimpleNamespace(
        colocate_stage=0, path="unused-test-path", npu_share_snapshot=shared
    )
    methods["_update_weights_from_disk_stage"](engine, meta)
    if shared:
        publisher.export.assert_called_once_with(meta)
        engine._save_model_to_hf.assert_not_called()
    else:
        engine._save_model_to_hf.assert_called_once_with(
            meta.path, engine.tokenizer, None
        )
        publisher.export.assert_not_called()


def test_shared_disk_receiver_does_not_enter_actor_onload_context(methods):
    entered = []

    @contextmanager
    def onload():
        entered.append(True)
        yield

    engine = SimpleNamespace(
        _check_rollout_engine_connected=Mock(),
        _offload_aware_context=onload,
        _update_weights_from_disk_stage=Mock(),
    )
    meta = SimpleNamespace(type="disk", colocate_mode=True, colocate_stage=1)
    methods["update_weights"](engine, meta)
    assert entered == []
    engine._update_weights_from_disk_stage.assert_called_once_with(meta)


def test_ready_joins_reader_before_restoring_model(methods):
    events = []
    engine = SimpleNamespace(
        wait_shared_snapshot=lambda: events.append("wait"),
        model=object(),
        is_offload=True,
        onload=lambda: events.append("restore"),
    )
    methods["_ensure_ready"](engine)
    assert events == ["wait", "restore"]


@pytest.mark.parametrize(
    "method,action", [("offload", "offload"), ("onload", "restore")]
)
def test_shared_transitions_wait_and_bypass_legacy_tensor_reassignment(
    methods, method, action
):
    events = []
    engine = SimpleNamespace(
        wait_shared_snapshot=lambda: events.append("wait"),
        _shared_memory=object(),
        colocation_memory=lambda a: events.append(a),
    )
    methods[method](engine)
    assert events == ["wait", action]


def test_optimizer_mutation_waits_for_snapshot_and_uses_managed_scope(methods):
    events = []

    @contextmanager
    def scope(tag):
        events.append(tag)
        yield
        events.append("exit")

    engine = SimpleNamespace(
        wait_shared_snapshot=lambda: events.append("wait"),
        _shared_scope=scope,
        _optimizer_step_impl=lambda: events.append("step") or {"loss": 1},
    )
    assert methods["optimizer_step"](engine) == {"loss": 1}
    assert events == ["wait", "actor_optimizer", "step", "exit"]


@pytest.mark.parametrize(
    "method,action",
    [
        ("release_memory_occupation", "offload"),
        ("resume_memory_occupation", "restore"),
    ],
)
@pytest.mark.parametrize("tags", [None, ["weights", "optimizer"]])
def test_public_memory_api_preserves_shared_ownership(methods, method, action, tags):
    events = []
    engine = SimpleNamespace(
        wait_shared_snapshot=lambda: events.append("wait"),
        _shared_memory=object(),
        colocation_memory=lambda a: events.append(a),
    )
    methods[method](engine, tags)
    assert events == ["wait", action]


@pytest.mark.parametrize(
    "method", ["release_memory_occupation", "resume_memory_occupation"]
)
@pytest.mark.parametrize("tags", [[], ["weights"], ["optimizer"], ["invalid"]])
def test_public_memory_api_rejects_partial_shared_transitions(methods, method, tags):
    engine = SimpleNamespace(
        wait_shared_snapshot=Mock(), _shared_memory=object(), colocation_memory=Mock()
    )
    with pytest.raises(ValueError, match="Partial tag transitions"):
        methods[method](engine, tags)
    engine.wait_shared_snapshot.assert_called_once()
    engine.colocation_memory.assert_not_called()
