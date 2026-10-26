# SPDX-License-Identifier: Apache-2.0
"""Ownership and prefix guards for the opt-in KV handoff adapter."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from areal.engine.vllm_ext.kv_handoff import install, prefix_key, validate_sampling


def sampling(**kwargs):
    return SimpleNamespace(
        **dict(
            dict(
                n=1,
                presence_penalty=0,
                frequency_penalty=0,
                repetition_penalty=1,
                min_tokens=0,
                seed=None,
                stop=[],
                prompt_logprobs=None,
                extra_args={},
            ),
            **kwargs,
        )
    )


@pytest.fixture
def classes(monkeypatch):
    class Scheduler:
        def schedule(self):
            return dict(
                ids=[r.request_id for r in self.running],
                width=self.max_num_running_reqs,
            )

    class EngineCore:
        def add_request(self, request, request_wave=0):
            self.admitted = request

    for name, cls in [
        ("vllm.v1.core.sched.scheduler", Scheduler),
        ("vllm.v1.engine.core", EngineCore),
    ]:
        module = ModuleType(name)
        setattr(module, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("AREAL_RUNNING_REBALANCE", "kv")
    install()
    return Scheduler, EngineCore


def test_frozen_request_keeps_capacity_and_returns_to_running(classes):
    scheduler = classes[0]()
    frozen = SimpleNamespace(request_id="frozen")
    other = SimpleNamespace(request_id="other")
    scheduler.running = [frozen, other]
    scheduler.max_num_running_reqs = 2
    scheduler._areal_kv_frozen = {"frozen": "ticket"}
    result = scheduler.schedule()
    assert result == dict(ids=["other"], width=1)
    assert scheduler.running == [other, frozen]
    assert scheduler.max_num_running_reqs == 2


def test_import_uses_external_prefix_and_native_admission(classes):
    core = classes[1]()
    calls = []
    core.model_executor = SimpleNamespace(
        collective_rpc=lambda *a, **kw: calls.append((a, kw)) or [{}]
    )
    allocations = []
    manager = SimpleNamespace(
        allocate_slots=lambda *a, **kw: allocations.append(kw) or object(),
        get_block_ids=lambda rid: ([3, 4],),
    )
    core.scheduler = SimpleNamespace(block_size=16, kv_cache_manager=manager)
    tokens = list(range(18))
    core._areal_kv_imports = {
        "t": dict(ticket="t", key=prefix_key(tokens), computed=17, blocks=2)
    }
    request = SimpleNamespace(
        request_id="new-attempt",
        prompt_token_ids=tokens,
        sampling_params=sampling(extra_args={"areal_kv_ticket": "t"}),
        abort_immediately=False,
        num_computed_tokens=0,
    )
    core.add_request(request)
    assert core.admitted is request
    assert request.num_computed_tokens == 17
    assert allocations[0]["num_external_computed_tokens"] == 17
    assert calls[0][1]["args"][1]["blocks"] == [3, 4]
    assert not core._areal_kv_imports


def test_import_rejects_wrong_prefix_before_allocation(classes):
    core = classes[1]()
    core._areal_kv_imports = {"t": dict(ticket="t", key=prefix_key([1, 2]))}
    request = SimpleNamespace(
        prompt_token_ids=[1, 3],
        sampling_params=sampling(extra_args={"areal_kv_ticket": "t"}),
    )
    with pytest.raises(RuntimeError, match="prefix mismatch"):
        core.add_request(request)
    assert not hasattr(core, "admitted")


def test_import_ticket_is_one_use(classes):
    core = classes[1]()
    request = SimpleNamespace(
        prompt_token_ids=[1, 2],
        sampling_params=sampling(extra_args={"areal_kv_ticket": "missing"}),
    )
    with pytest.raises(RuntimeError, match="already consumed"):
        core.add_request(request)


@pytest.mark.parametrize(
    "override",
    [
        dict(seed=1),
        dict(min_tokens=2),
        dict(repetition_penalty=1.1),
        dict(stop=["end"]),
    ],
)
def test_unsupported_sampling_is_rejected(override):
    with pytest.raises(ValueError):
        validate_sampling(sampling(**override))


def test_runtime_vendor_scheduler_override_is_frozen(classes):
    from areal.engine.vllm_ext.kv_handoff import install_freeze

    class VendorScheduler(classes[0]):
        def schedule(self):
            return [r.request_id for r in self.running]

    scheduler = VendorScheduler()
    request = SimpleNamespace(request_id="frozen")
    scheduler.running = [request]
    scheduler.max_num_running_reqs = 4
    scheduler._areal_kv_frozen = {"frozen": "ticket"}
    install_freeze(type(scheduler))
    assert scheduler.schedule() == []
    assert scheduler.running == [request]
    assert scheduler.max_num_running_reqs == 4


@pytest.mark.parametrize("width", [1, 2, 4])
def test_tp_acknowledgements_sum_bytes_and_take_slowest_rank(width):
    from areal.engine.vllm_ext.kv_handoff import aggregate_worker_results

    rows = [
        dict(
            tp_size=width,
            tp_rank=i,
            tensors=2,
            block_bytes=128,
            block_tokens=16,
            bytes=256,
            worker_s=0.1 * (i + 1),
        )
        for i in range(width)
    ]
    result = aggregate_worker_results(rows)
    assert result["bytes"] == 256 * width
    assert result["worker_s"] == 0.1 * width
    assert "tp_rank" not in result


@pytest.mark.parametrize(
    "rows",
    [
        [dict(tp_size=2, tp_rank=0)],
        [
            dict(tp_size=2, tp_rank=0, model_layout="a"),
            dict(tp_size=2, tp_rank=1, model_layout="b"),
        ],
        [dict(tp_size=2, tp_rank=0), dict(tp_size=2, tp_rank=0)],
        [
            dict(tp_size=2, tp_rank=0, block_bytes=128),
            dict(tp_size=2, tp_rank=1, block_bytes=256),
        ],
    ],
)
def test_tp_partial_duplicate_or_incompatible_ack_is_rejected(rows):
    from areal.engine.vllm_ext.kv_handoff import aggregate_worker_results

    with pytest.raises(RuntimeError):
        aggregate_worker_results(rows)


def test_multi_peer_transfer_selects_the_correct_group_and_direction():
    """Multiple donors must not overwrite each other's receive channels."""
    from areal.engine.vllm_ext.kv_handoff import transfer_group

    first, second = object(), object()
    worker = SimpleNamespace(
        _areal_kv_groups={
            "a-to-c": dict(group=first, source=False),
            "b-to-c": dict(group=second, source=False),
        }
    )
    assert transfer_group(worker, dict(pair_id="a-to-c", source=False)) is first
    assert transfer_group(worker, dict(pair_id="b-to-c", source=False)) is second
    for data in [
        dict(pair_id="missing", source=False),
        dict(pair_id="a-to-c", source=True),
    ]:
        with pytest.raises(ValueError, match="peer|direction"):
            transfer_group(worker, data)


def test_one_tp_member_rejecting_staging_blocks_collective_transfer():
    from areal.engine.vllm_ext.kv_handoff import aggregate_worker_results

    result = aggregate_worker_results(
        [
            dict(tp_size=2, tp_rank=0, worker_s=0.01),
            dict(tp_size=2, tp_rank=1, rejected="insufficient_target_staging_headroom"),
        ]
    )
    assert "rejected" in result
