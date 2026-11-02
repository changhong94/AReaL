import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "baseline", Path(__file__).parent / "templates/colocated_baseline.py"
)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def config():
    return dict(
        actor=dict(backend="megatron:d1p1t1"),
        rollout=dict(backend="vllm:d1p1t1", max_head_offpolicyness=2),
        cluster={},
        train_dataset=dict(batch_size=32),
        gconfig=dict(n_samples=8),
    )


def test_colocation_uses_all_cards_without_mutating_workload():
    original = config()
    result = baseline.build_colocated_config(original, devices=[4, 5], inference_tp=1)
    assert result["actor"]["backend"] == "megatron:d2p1t1"
    assert result["rollout"]["backend"] == "vllm:d2p1t1"
    assert result["rollout"]["scheduling_strategy"]["target"] == "actor"
    assert result["npu_share"] is False and result["enable_offload"] is True
    assert result["actor"]["weight_update_mode"] == "awex"
    assert result["rollout"]["max_head_offpolicyness"] == 2
    assert result["vllm"]["enable_sleep_mode"] is True
    assert result["vllm"]["additional_config"]["weight_nz_mode"] == 0
    assert original == config()


def test_six_card_uneven_batch_rejected_without_padding():
    with pytest.raises(ValueError, match="Answer batch"):
        baseline.build_colocated_config(
            config(), devices=list(range(6)), inference_tp=2
        )


def test_unverified_fsdp_backend_rejected():
    cfg = config()
    cfg["actor"]["backend"] = "fsdp:d1p1t1"
    with pytest.raises(ValueError, match="Megatron"):
        baseline.build_colocated_config(cfg, devices=[0, 1], inference_tp=1)


def test_comparison_requires_three_real_results():
    partial = baseline.comparison({"fixed": 100, "latest": 80})
    assert partial["colocated_vs_fixed_percent"] is None
    assert partial["shared_vs_fixed_percent"] == pytest.approx(20)
    result = baseline.comparison({"fixed": 100, "colocated": 90, "latest": 80})
    assert result["shared_vs_fixed_percent"] == pytest.approx(20)
    assert result["shared_vs_colocated_percent"] == pytest.approx(100 / 9)


def test_colocation_clears_inherited_slab_environment():
    cfg = config()
    cfg["actor"]["scheduling_spec"] = [{"env_vars": {"AREAL_KV_SLAB_BLOCKS": "16"}}]
    result = baseline.build_colocated_config(cfg, devices=[0, 1], inference_tp=1)
    assert (
        result["actor"]["scheduling_spec"][0]["env_vars"]["AREAL_KV_SLAB_BLOCKS"] == "0"
    )
    assert (
        cfg["actor"]["scheduling_spec"][0]["env_vars"]["AREAL_KV_SLAB_BLOCKS"] == "16"
    )


def test_colocation_clears_single_scheduling_spec():
    cfg = config()
    cfg["actor"]["scheduling_spec"] = {"env_vars": {"AREAL_KV_SLAB_BLOCKS": "16"}}
    result = baseline.build_colocated_config(cfg, devices=[0, 1], inference_tp=1)
    assert result["actor"]["scheduling_spec"]["env_vars"]["AREAL_KV_SLAB_BLOCKS"] == "0"


def test_six_card_48_question_batch_supported():
    cfg = config()
    cfg["train_dataset"]["batch_size"] = 48
    result = baseline.build_colocated_config(
        cfg, devices=list(range(6)), inference_tp=2
    )
    assert result["actor"]["backend"] == "megatron:d6p1t1"
    assert result["rollout"]["backend"] == "vllm:d3p1t2"
    assert result["train_dataset"]["batch_size"] * result["gconfig"]["n_samples"] == 384
