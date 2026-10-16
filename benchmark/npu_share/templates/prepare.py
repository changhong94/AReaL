"""Select exact non-repeating datasets and write matched run configurations."""

import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from areal.dataset import _get_custom_dataset
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
    if len(selected) < count:
        raise ValueError(
            f"{split}: need {count} eligible distinct questions; found {len(selected)}. Reduce --batches or configure a larger dataset."
        )
    pq.write_table(table.take(selected), data / (split + ".parquet"))
    manifest[split] = dict(
        source_rows=table.num_rows,
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
    if len(actual) != count:
        raise ValueError(
            f"{split}: actual preprocessing count {len(actual)} != {count}"
        )
(root / "data-manifest.json").write_text(json.dumps(manifest, indent=2))
training_devices = s.get("npu_count", 2) // 2
inference_tp = s.get("inference_tp", 1)
instances = training_devices // inference_tp
for arm in ["fixed", "latest"]:
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
    c["actor"]["weight_update_mode"] = "disk" if shared else "xccl"
    c["gconfig"].update(
        n_samples=s["answers"], max_new_tokens=s["max_new_tokens"], max_tokens=4096
    )
    c["rollout"].update(
        backend=f"vllm:d{instances * (2 if shared else 1)}p1t{inference_tp}",
        max_concurrent_rollouts=s["concurrency"],
        max_head_offpolicyness=s.get("max_head_offpolicyness", 2),
        queue_size=max(32, s["concurrency"] * 16),
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
    for split in ["train", "valid"]:
        c[split + "_dataset"].update(
            path=str(data), batch_size=16 if split == "train" else 4, max_length=2048
        )
    c["train_dataset"]["shuffle"] = False
    c["perf_tracer"].update(enabled=True)
    c["perf_tracer"]["session_tracer"]["enabled"] = True
    config_path = root / arm / "config.yaml"
    config_path.write_text(yaml.safe_dump(c, sort_keys=False))
    from areal.api.alloc_mode import ModelAllocation
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from areal.trainer.rl_trainer import PPOTrainer

    trainer = object.__new__(PPOTrainer)
    trainer.config, _ = load_expr_config(["--config", str(config_path)], GRPOConfig)
    trainer.actor_alloc = ModelAllocation.from_str(c["actor"]["backend"])
    trainer.rollout_alloc = ModelAllocation.from_str(c["rollout"]["backend"])
    for role in ["actor", "rollout", "critic", "ref", "teacher"]:
        setattr(trainer, "_should_offload_" + role, role == "rollout" and shared)
    trainer._validate_cfg()
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
