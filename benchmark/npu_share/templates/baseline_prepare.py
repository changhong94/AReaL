"""Validate a config against the baseline checkout, without research imports."""

import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path

import yaml
from omegaconf import OmegaConf

import areal
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from areal.workflow.rlvr import default_get_input_ids_fn

arm = sys.argv[1]
root = Path("/work")
assert Path(areal.__file__).resolve().is_relative_to(Path("/repo"))
settings = json.loads((root / "settings.json").read_text())
if arm == "colocated":
    import_module("awex.meta.meta_server")
    import_module("awex.vllm_plugin")
path = root / arm / "config.yaml"
config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
# Only schema fields supported by the pinned community version are retained.
# Every removed field is recorded; no execution code is copied into the baseline.
schema = OmegaConf.to_container(OmegaConf.structured(GRPOConfig), resolve=False)
removed = []


def filter_fields(value, template, prefix=""):
    if not isinstance(value, dict) or not isinstance(template, dict):
        return value
    result = {}
    for key, item in value.items():
        full = prefix + key
        if key not in template:
            removed.append(full)
            continue
        default = template[key]
        result[key] = (
            filter_fields(item, default, full + ".")
            if isinstance(default, dict) and default
            else item
        )
    return result


config = filter_fields(config, schema)
config["enable_offload"] = arm == "colocated"
# Use upstream's supported local dataset mode, matching the common frozen workload.
for split in ("train_dataset", "valid_dataset"):
    config[split]["scheduling_spec"] = None
for role in ("actor", "rollout"):
    specs = config[role].get("scheduling_spec", [])
    if isinstance(specs, dict):
        specs = [specs]
    for spec in specs:
        env = spec.get("env_vars", {})
        spec["env_vars"] = {k: v for k, v in env.items() if not k.startswith("AREAL_")}
        spec["env_vars"]["AREAL_ALLOW_DEFAULT_ADMIN_KEY"] = "1"
        spec["env_vars"]["HCCL_NPU_SOCKET_PORT_RANGE"] = "auto"
        if settings.get("inference_mode", "graph") == "graph":
            spec["env_vars"]["TASK_QUEUE_ENABLE"] = "1"
config["scheduler"] = {"type": "ray"}
path.write_text(yaml.safe_dump(config, sort_keys=False))
validated, _ = load_expr_config(["--config", str(path)], GRPOConfig)

import_module("areal.engine.vllm_ext.areal_vllm_server")
assert validated.train_dataset.batch_size == settings["batch_size"]
assert validated.gconfig.n_samples == settings["answers"]
assert validated.total_train_steps == settings["batches"]

if settings["rl"] == "code":
    import_module("baseline_code_workflow")

tokenizer = load_hf_tokenizer(validated.tokenizer_path)
dataset = get_custom_dataset("train", validated.train_dataset, tokenizer)
assert len(dataset) == settings["questions"]
prompt_hashes = []
for row in dataset:
    ids = default_get_input_ids_fn(row["messages"], tokenizer, False)
    prompt_hashes.append(hashlib.sha256(json.dumps(ids).encode()).hexdigest())
(root / arm / "input-token-hashes.json").write_text(json.dumps(prompt_hashes))
expected_path = root / "expected-input-token-hashes.json"
if expected_path.exists():
    assert prompt_hashes == json.loads(expected_path.read_text()), (
        "Baseline prompt tokens differ from shared workload"
    )

(root / arm / "baseline-config-audit.json").write_text(
    json.dumps(
        {
            "areal_file": areal.__file__,
            "removed_research_fields": removed,
            "source": settings["source_provenance"][arm],
        },
        indent=2,
    )
)
print("Baseline schema and workload verified", arm)
