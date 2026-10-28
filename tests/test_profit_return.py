# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.test_asymmetric_publication import controller, fixed_meta  # noqa: F401
from tests.test_resident_colocation import trainer as resident_trainer

from areal.infra.profit_return import estimate_return


def inputs(length=100, questions=4, answers=2):
    from areal.infra.rolling_drain import RollingBatchCalibration

    c = RollingBatchCalibration()
    c.lengths = [(length, True)] * 8
    c.decode = [
        {
            (8, 128): dict(
                concurrency=8,
                context=128,
                seconds=0.01,
                count=1,
                max_concurrency=8,
                source_window=1,
                source_windows=[1],
            )
        }
        for _ in range(2)
    ]
    c.first = [dict(unit=0.00001), dict(unit=0.00001)]
    route = dict(active=[], waiting=[], completed={})
    for q in range(questions):
        for a in range(answers):
            route["waiting"].append(
                dict(
                    request_id=f"{q}-{a}",
                    group_size=answers,
                    metrics=dict(
                        question_id=str(q),
                        generated_tokens=0,
                        prompt_tokens=32,
                        max_new_tokens=2048,
                    ),
                )
            )
    engines = [dict(active=[]), dict(active=[])]
    return c, route, engines


def test_ready_batch_does_not_need_calibration():
    assert estimate_return(None, {}, [], 0, [])["action"] == "keep"


def test_missing_cost_history_probe_is_explicit(monkeypatch):
    monkeypatch.setenv("AREAL_RETURN_CALIBRATION_PROBE", "1")
    c, r, e = inputs()
    assert estimate_return(c, r, e, 4, [])["reason"] == "calibration_probe"


def test_short_work_keeps_actor_resident():
    c, r, e = inputs()
    d = estimate_return(c, r, e, 4, [dict(publish=3, restore=0.5)])
    assert d["action"] == "keep" and d["gain_seconds"] < 0


def test_large_queue_returns_extra_replica():
    c, r, e = inputs(length=1000, questions=32)
    d = estimate_return(c, r, e, 32, [dict(publish=1, restore=0.1)])
    assert d["action"] == "return" and d["gain_seconds"] > 0


def test_partial_answer_group_is_not_counted_as_a_complete_question(monkeypatch):
    monkeypatch.setenv("AREAL_RETURN_CALIBRATION_PROBE", "1")
    c, r, e = inputs()
    r["waiting"].pop()
    assert (
        estimate_return(c, r, e, 4, [dict(publish=1, restore=0.1)])["reason"]
        == "incomplete_horizon_probe"
    )


def test_skip_publishes_fixed_without_snapshot_or_memory_switch():
    trainer = resident_trainer.__wrapped__()
    trainer._fixed_weight_update_meta = fixed_meta()
    trainer._return_shared_inference = False
    trainer._publish_weights_asymmetric(2)
    meta = trainer.actor.update_weights.call_args.args[0]
    assert meta.npu_share_fixed and not meta.npu_share_snapshot
    trainer.rollout.publish_fixed_version.assert_called_once_with(2)
    trainer.actor.wait_shared_snapshot.assert_not_called()
    trainer.rollout.load_shared_snapshot_async.assert_not_called()
    trainer._memory_policy.inference.assert_not_called()
    trainer._resume_training_inference()
    trainer.rollout.resume_training_inference.assert_not_called()


def test_skip_then_return_loads_latest_version():
    trainer = resident_trainer.__wrapped__()
    trainer._fixed_weight_update_meta = fixed_meta()
    trainer._return_shared_inference = False
    trainer._publish_weights_asymmetric(2)
    trainer._return_shared_inference = True
    trainer._publish_weights_asymmetric(3)
    trainer.actor.wait_shared_snapshot.assert_called_once_with(3)
    assert trainer.rollout.load_shared_snapshot_async.call_args.args[0].version == 3


def test_generic_resume_cannot_reopen_stale_shared(controller):  # noqa: F811
    controller.prepare_fixed_publication()
    controller.publish_fixed_version(2)
    controller.skip_inference_return(2)
    controller._worker_rpc.reset_mock()
    controller.resume()
    assert all(c.args[0] == 1 for c in controller._worker_rpc.call_args_list)
    assert controller._borrowed_pool.choose(0)[0] == 1
    with pytest.raises(RuntimeError, match="stale"):
        controller.resume_training_inference()


def test_retained_requests_do_not_veto_keep_decision(controller):  # noqa: F811
    controller._preserve_borrowed_requests = True
    controller._staleness_manager.get_stats.return_value.accepted = 100
    result = controller.decide_inference_return(16)
    assert result["action"] == "keep"
    assert result["reason"] == "next_batch_ready"


def test_resident_reuse_keeps_budget_check_without_repeated_pause():
    from areal.api.alloc_mode import ModelAllocation

    trainer = resident_trainer.__wrapped__()
    trainer.actor_alloc = ModelAllocation.from_str("fsdp:d1p1t1")
    trainer.rollout_alloc = ModelAllocation.from_str("vllm:d2p1t1")
    trainer._return_shared_inference = False
    trainer.rollout._shared_held_for_training = True
    trainer._start_train_prefetch(2)
    assert getattr(trainer, "_train_prefetch", None) is None
    trainer._offload_rollout()
    trainer._memory_policy.train.assert_called_once()
    trainer.rollout.borrow_for_training.assert_not_called()
    trainer.rollout.resume_training_inference.assert_not_called()


def test_missing_cost_history_keeps_training_by_default(monkeypatch):
    monkeypatch.delenv("AREAL_RETURN_CALIBRATION_PROBE", raising=False)
    c, route, engines = inputs()
    result = estimate_return(c, route, engines, 4, [])
    assert result["action"] == "keep"
    assert result["reason"] == "uncalibrated_keep"
    assert result["cost_samples"] == 0


@pytest.mark.parametrize(
    "reason",
    [
        "calibration_probe",
        "missing_observation_probe",
        "missing_metadata_probe",
        "incomplete_horizon_probe",
    ],
)
def test_every_uncalibrated_return_requires_opt_in(monkeypatch, reason):
    from areal.infra.profit_return import uncalibrated_return

    monkeypatch.delenv("AREAL_RETURN_CALIBRATION_PROBE", raising=False)
    result = uncalibrated_return(reason, known_questions=2)
    assert result == dict(
        action="keep",
        reason="uncalibrated_keep",
        missing_reason=reason,
        known_questions=2,
    )
    monkeypatch.setenv("AREAL_RETURN_CALIBRATION_PROBE", "1")
    assert uncalibrated_return(reason)["action"] == "return"
