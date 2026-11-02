"""Community entrypoint: external timing only, no research scheduling patches."""

import json
import sys
import time
from pathlib import Path

import areal
from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer

root = Path("/run-output")


def event(name, **fields):
    with (root / "phases.jsonl").open("a") as f:
        f.write(json.dumps(dict(time=time.time(), event=name, **fields)) + "\n")


def main():
    assert Path(areal.__file__).resolve().is_relative_to(Path("/repo"))
    settings = json.loads(Path("/work/settings.json").read_text())
    config, _ = load_expr_config(sys.argv[1:], GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train = get_custom_dataset("train", config.train_dataset, tokenizer)
    valid = get_custom_dataset("test", config.valid_dataset, tokenizer)
    assert len(train) == settings["questions"]
    assert config.train_dataset.batch_size == settings["batch_size"]
    assert config.gconfig.n_samples == settings["answers"]
    event(
        "source",
        areal_file=areal.__file__,
        trainer_file=sys.modules[PPOTrainer.__module__].__file__,
    )
    event("initialize.begin")
    with PPOTrainer(config, train_dataset=train, valid_dataset=valid) as trainer:
        event("initialize.end")
        event("train.begin")
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow"
            if settings["rl"] == "math"
            else "baseline_code_workflow.CodeRLVRWorkflow",
            workflow_kwargs=dict(
                reward_fn="areal.reward.gsm8k.gsm8k_reward_fn"
                if settings["rl"] == "math"
                else "code_reward.grade",
                gconfig=config.gconfig,
                tokenizer=config.tokenizer_path,
                enable_thinking=False,
            ),
            eval_workflow=None,
        )
        event("train.end")
    event("closed")


if __name__ == "__main__":
    main()
