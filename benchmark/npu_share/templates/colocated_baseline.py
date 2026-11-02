"""Plan a community AWEX arm without borrowing resources or changing workload."""

import math
from copy import deepcopy

ARMS = ("fixed", "colocated", "latest")
LABELS = {"fixed": "固定", "colocated": "colocated", "latest": "共享"}


def build_colocated_config(config, *, devices, inference_tp, actor_tp=1):
    """Derive an explicit NPU AWEX baseline; unsupported layouts fail closed."""
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Distinct physical devices are required")
    count = len(devices)
    if inference_tp < 1 or actor_tp < 1 or count % inference_tp or count % actor_tp:
        raise ValueError("Parallel layouts must divide the allocated card count")
    if not config["actor"]["backend"].startswith("megatron:"):
        raise ValueError("Community AWEX baseline currently requires Megatron")
    if not config["rollout"]["backend"].startswith("vllm:"):
        raise ValueError("NPU baseline requires vLLM")
    dp = count // actor_tp
    answers = config["train_dataset"]["batch_size"] * config["gconfig"]["n_samples"]
    if answers % dp:
        raise ValueError(
            "Answer batch must divide actor DP; do not silently pad workload"
        )
    result = deepcopy(config)
    result.update(npu_share=False, enable_offload=True)
    vllm = result.setdefault("vllm", {})
    vllm["enable_sleep_mode"] = True
    additional = dict(vllm.get("additional_config") or {})
    additional["weight_nz_mode"] = 0
    vllm["additional_config"] = additional
    for role in ("actor", "rollout"):
        specs = result[role].get("scheduling_spec", [])
        if isinstance(specs, str):
            continue  # OmegaConf reference; resolved in baseline preflight.
        if isinstance(specs, dict):
            specs = [specs]
        for spec in specs:
            env = spec.setdefault("env_vars", {})
            env.update(
                AREAL_KV_SLAB_BLOCKS="0", AREAL_AUTO_MEMORY="0", AREAL_DYNAMIC_KV="0"
            )
    result["scheduler"] = {"type": "ray"}
    result["cluster"]["n_gpus_per_node"] = count
    result["actor"].update(
        backend=f"megatron:d{dp}p1t{actor_tp}", weight_update_mode="awex"
    )
    result["actor"]["scheduling_strategy"] = {"type": "separation"}
    result["rollout"].update(backend=f"vllm:d{count // inference_tp}p1t{inference_tp}")
    result["rollout"]["scheduling_strategy"] = {
        "type": "colocation",
        "target": "actor",
        "fork": True,
    }
    result["awex"] = dict(
        enable_colocate_mode=True,
        device_backend="npu",
        use_mindspeed=True,
        meta_server_addr="auto",
        comm_backend="hccl",
        weights_exchange_ipc_backend="cpu",
        weights_comm_nccl_group_size=1,
        weights_validation_steps=0,
        validate_weights_every_n_steps=1,
    )
    return result


def comparison(times):
    """Compare available arms; missing colocation never blocks fixed/shared."""
    if (
        not {"fixed", "latest"}.issubset(times)
        or set(times) - set(ARMS)
        or any(not math.isfinite(v) or v <= 0 for v in times.values())
    ):
        raise ValueError("Positive fixed/shared durations are required")
    return {
        "loop_seconds": dict(times),
        "colocated_vs_fixed_percent": 100 * (1 - times["colocated"] / times["fixed"])
        if "colocated" in times
        else None,
        "shared_vs_fixed_percent": 100 * (1 - times["latest"] / times["fixed"]),
        "shared_vs_colocated_percent": 100 * (1 - times["latest"] / times["colocated"])
        if "colocated" in times
        else None,
    }
