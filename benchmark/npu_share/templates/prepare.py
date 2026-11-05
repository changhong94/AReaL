"""Select exact non-repeating datasets and write matched run configurations."""

import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from areal.dataset import _get_custom_dataset
from areal.engine.vllm_ext.graph_compat import full_graph_config
from areal.utils.hf_utils import load_hf_tokenizer

root = Path("/work")
s = json.loads((root / "settings.json").read_text())
if s["engine"] == "megatron" and sys.version_info[:2] != (3, 12):
    raise RuntimeError("Megatron benchmark requires Python 3.12")
tokenizer = load_hf_tokenizer("/model")
kind = "gsm8k" if s["rl"] == "math" else "leetcode"
data = root / "data" / kind
data.mkdir(parents=True)
manifest = {}
for split, count in [("train", s["questions"]), ("test", 16)]:
    path = Path("/input") / (split + ".parquet")
    table = pq.read_table(path)
    if s["rl"] == "code":
        from areal.dataset.leetcode import _PROMPT

        texts = [r["question"] + _PROMPT for r in table.to_pylist()]
    else:
        texts = [
            q + "\nPlease put your final answer within \\boxed{}."
            for q in table.column("question").to_pylist()
        ]
    selected = []
    for i, text in enumerate(texts):
        if len(tokenizer.encode(text)) <= 2048:
            selected.append(i)
        if len(selected) == count:
            break
    eligible_count = len(selected)
    repeat_questions = (
        s.get("repeat_code_questions", False) and s["rl"] == "code" and split == "train"
    )
    if repeat_questions and selected and len(selected) < count:
        selected = (selected * ((count + len(selected) - 1) // len(selected)))[:count]
    if len(selected) < count:
        raise ValueError(
            f"{split}: need {count} eligible distinct questions; found {len(selected)}. Reduce --batches/--batch-size or configure a larger dataset."
        )
    pq.write_table(table.take(selected), data / (split + ".parquet"))
    manifest[split] = dict(
        source_rows=table.num_rows,
        sampling_policy="cycle" if repeat_questions else "without_replacement",
        eligible_rows_scanned=eligible_count,
        unique_source_rows=len(set(selected)),
        repeated_rows=len(selected) - len(set(selected)),
        selected_rows=selected,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
if s["rl"] == "math":
    (data / "README.md").write_text(
        "---\nconfigs:\n- config_name: main\n  data_files:\n  - split: train\n    path: train.parquet\n  - split: test\n    path: test.parquet\n---\n"
    )
for split, count in [("train", s["questions"]), ("test", 16)]:
    actual = _get_custom_dataset(
        path=str(data), split=split, type="rl", tokenizer=tokenizer, max_length=2048
    )
    if split == "train":
        from areal.workflow.rlvr import default_get_input_ids_fn

        expected = [
            hashlib.sha256(
                json.dumps(
                    default_get_input_ids_fn(row["messages"], tokenizer, False)
                ).encode()
            ).hexdigest()
            for row in actual
        ]
        (root / "expected-input-token-hashes.json").write_text(json.dumps(expected))
    if len(actual) != count:
        raise ValueError(
            f"{split}: actual preprocessing count {len(actual)} != {count}"
        )
(root / "data-manifest.json").write_text(json.dumps(manifest, indent=2))
training_devices = s.get("allocation", {}).get("training", s.get("npu_count", 2) // 2)
inference_devices = s.get("npu_count", 2) - training_devices
inference_tp = s.get("inference_tp", 1)
instances = inference_devices // inference_tp
shared_instances = s.get("npu_count", 2) // inference_tp
arm_status = {}
for arm in s.get("arms", ["fixed", "latest"]):
    shared = arm == "latest"
    c = yaml.safe_load(Path("/tool/base.yaml").read_text())
    c.update(
        experiment_name="npu-share-ab",
        trial_name="run",
        npu_share=shared,
        total_train_epochs=1,
        total_train_steps=s["batches"],
        seed=s["seed"],
    )
    c["cluster"].update(fileroot="/run-output", n_gpus_per_node=s.get("npu_count", 2))
    c["cluster"]["name_resolve"]["nfs_record_root"] = "/run-output/name-resolve"
    c["actor"].update(backend=f"{s['engine']}:d{training_devices}p1t1", path="/model")
    if s["engine"] == "fsdp":
        c["actor"].pop("megatron", None)
    c["actor"]["mb_spec"]["max_tokens_per_mb"] = 4096
    # Match the independent graph baseline's actor queue mode. Deep task
    # queues retain extra asynchronous operator workspaces on Ascend.
    if shared and s.get("inference_mode", "graph") == "graph":
        for spec in c["actor"]["scheduling_spec"]:
            spec.setdefault("env_vars", {})["TASK_QUEUE_ENABLE"] = "1"
    c["actor"]["weight_update_mode"] = "disk" if shared else "xccl"
    c["gconfig"].update(
        n_samples=s["answers"], max_new_tokens=s["max_new_tokens"], max_tokens=4096
    )
    c["rollout"].update(
        backend=f"vllm:d{shared_instances if shared else instances}p1t{inference_tp}",
        max_concurrent_rollouts=s["concurrency"],
        max_head_offpolicyness=s.get("max_head_offpolicyness", 2),
        # The community dispatcher reserves one consumer batch before submission.
        queue_size=max(32, s["concurrency"] * 16, s["batch_size"] + s["concurrency"]),
    )
    c["rollout"]["scheduling_strategy"].update(
        type="colocation" if shared else "separation",
        target="actor" if shared else None,
    )
    c["vllm"].update(
        distributed_executor_backend="uni" if inference_tp == 1 else "mp",
        max_model_len=4096,
        gpu_memory_utilization=0.4,
        max_num_seqs=max(64, s["concurrency"] * 4),
    )
    inference_mode = s.get("inference_mode", "eager")
    if inference_mode not in {"graph", "eager"}:
        raise ValueError("Invalid inference mode")
    c["vllm"]["enforce_eager"] = inference_mode == "eager"
    c["vllm"]["compilation_config"] = (
        full_graph_config(
            c["vllm"].get("max_num_batched_tokens", 2048), c["vllm"]["max_num_seqs"]
        )
        if inference_mode == "graph"
        else None
    )
    for split in ["train", "valid"]:
        c[split + "_dataset"].update(
            path=str(data),
            batch_size=s["batch_size"] if split == "train" else 4,
            max_length=2048,
        )
    c["train_dataset"]["shuffle"] = False
    c["perf_tracer"].update(enabled=True)
    c["perf_tracer"]["session_tracer"]["enabled"] = True
    try:
        if arm == "colocated":
            from colocated_baseline import build_colocated_config

            c = build_colocated_config(
                c,
                devices=s["devices"],
                inference_tp=inference_tp,
                actor_tp=s.get("colocated_actor_tp", 1),
            )
        config_path = root / arm / "config.yaml"
        config_path.write_text(yaml.safe_dump(c, sort_keys=False))
        if arm != "latest":
            # Baselines are validated by their own pinned source, not the research API.
            arm_status[arm] = {"status": "prepared"}
            continue
        from areal.api.alloc_mode import ModelAllocation
        from areal.api.cli_args import GRPOConfig, load_expr_config
        from areal.trainer.rl_trainer import PPOTrainer

        trainer = object.__new__(PPOTrainer)
        trainer.config, _ = load_expr_config(["--config", str(config_path)], GRPOConfig)
        trainer.actor_alloc = ModelAllocation.from_str(c["actor"]["backend"])
        trainer.rollout_alloc = ModelAllocation.from_str(c["rollout"]["backend"])
        for role in ["actor", "rollout", "critic", "ref", "teacher"]:
            setattr(
                trainer,
                "_should_offload_" + role,
                role == "rollout" and (shared or arm == "colocated"),
            )
        trainer._validate_cfg()
        arm_status[arm] = {"status": "prepared"}
    except Exception as exc:
        if arm != "colocated":
            raise
        arm_status[arm] = {
            "status": "unsupported",
            "reason": f"{type(exc).__name__}: {exc}",
        }
(root / "arm-status.json").write_text(json.dumps(arm_status, indent=2))
if s["rl"] == "code":
    print("Code SDK and workflow imports passed")
print(
    "Prepared exact workload",
    s["questions"],
    "questions;",
    s["batches"],
    "batches;",
    sys.version,
)
