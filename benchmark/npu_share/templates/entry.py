"""Identical workload and phase instrumentation for both performance arms."""

import json
import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.infra.controller import rollout_controller as routing_controller
from areal.utils import perf_tracer
from areal.utils.hf_utils import load_hf_tokenizer


def event(name, **fields):
    with Path("/run-output/phases.jsonl").open("a") as f:
        f.write(json.dumps(dict(time=time.time(), event=name, **fields)) + "\n")


original_configure = perf_tracer.configure


def calibrated_configure(*args, **kwargs):
    tracer = original_configure(*args, **kwargs)
    mono, wall = time.perf_counter_ns(), time.time()
    Path("/run-output/clock-master.json").write_text(
        json.dumps(
            {
                "origin_wall_seconds": wall - (mono - tracer._origin_ns) / 1e9,
            }
        )
    )
    return tracer


perf_tracer.configure = calibrated_configure


# Experiment-only control; production uses task affinity by default.

OriginalLeasePool = routing_controller.RequestLeasePool


class ExperimentLeasePool(OriginalLeasePool):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, task_affinity=os.environ["EXPERIMENT_TASK_AFFINITY"] == "1", **kwargs
        )


routing_controller.RequestLeasePool = ExperimentLeasePool

SETTINGS = json.loads(Path("/work/settings.json").read_text())
KIND = "gsm8k" if SETTINGS["rl"] == "math" else "leetcode"


def main():
    if SETTINGS["engine"] == "megatron":
        assert sys.version_info[:2] == (3, 12), sys.version
    event("python", version=sys.version, executable=sys.executable)
    config, _ = load_expr_config(sys.argv[1:], GRPOConfig)
    assert os.environ.get("AREAL_NPU_DEVICE_TRACE") == "1"
    assert (
        config.total_train_epochs == 1
        and config.total_train_steps == SETTINGS["batches"]
    )
    assert config.gconfig.n_samples == SETTINGS["answers"]
    assert config.train_dataset.batch_size == SETTINGS["batch_size"]
    event(
        "settings",
        batch_drain=os.getenv("AREAL_BATCH_DRAIN", "0"),
        adaptive_drain=os.getenv("AREAL_ADAPTIVE_DRAIN", "0"),
        drain_metrics=os.getenv("AREAL_DRAIN_METRICS", "0"),
        shared_drain_seconds=float(os.getenv("AREAL_SHARED_DRAIN_SECONDS", "0")),
        training_priority_inference=os.getenv("AREAL_TRAIN_PRIORITY_INFERENCE") == "1",
        n_samples=config.gconfig.n_samples,
        total_batches=SETTINGS["batches"],
        npu_share=config.npu_share,
        concurrency=config.rollout.max_concurrent_rollouts,
        npu_device_trace=True,
        periodic_npu_sampling=True,
    )
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train = get_custom_dataset("train", config.train_dataset, tokenizer)
    valid = get_custom_dataset("test", config.valid_dataset, tokenizer)
    counts = {
        split: pq.read_metadata(f"/work/data/{KIND}/{split}.parquet").num_rows
        for split in ["train", "test"]
    }
    assert counts == {"train": SETTINGS["questions"], "test": 16}
    event("dataset", **counts, npu_share=config.npu_share)
    if os.getenv("CONFIGCHECK") == "1":
        from areal.dataset import _get_custom_dataset

        for split, expected in counts.items():
            processed = _get_custom_dataset(
                path=f"/work/data/{KIND}",
                split=split,
                type="rl",
                tokenizer=tokenizer,
                max_length=2048,
            )
            assert len(processed) == expected
        print("CHECKED", counts, config.npu_share, flush=True)
        return
    kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn"
        if SETTINGS["rl"] == "math"
        else "code_reward.grade",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
    )
    event("initialize.begin")
    with PPOTrainer(config, train_dataset=train, valid_dataset=valid) as trainer:
        assert len(train) == SETTINGS["questions"] and len(valid) == 16
        assert len(trainer.train_dataloader) == SETTINGS["batches"]
        event("initialize.end")
        for name in [
            "_offload_rollout",
            "_publish_weights",
            "_publish_fixed_weights",
            "_finish_shared_weights",
            "_finish_step",
            "_evaluate",
        ]:
            original = getattr(trainer, name)

            def traced(*args, _name=name, _original=original, **kw):
                event(_name + ".begin", args=str(args))
                try:
                    result = _original(*args, **kw)
                except BaseException:
                    event(_name + ".error")
                    raise
                event(_name + ".end")
                return result

            setattr(trainer, name, traced)
        from deployment_calibration import export, install
        from fair48_control import configure_workload

        install(trainer, "/work", event)
        submitted = configure_workload(trainer)
        event("train.begin")
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow"
            if SETTINGS["rl"] == "math"
            else "code_workflow.CodeRLVRWorkflow",
            workflow_kwargs=kwargs,
            eval_workflow=None,
            eval_workflow_kwargs={**kwargs, "gconfig": config.eval_gconfig},
        )
        event("train.end")
        export(trainer, "/work", event)
        assert len(submitted) == SETTINGS["questions"]
        Path("/run-output/submitted.json").write_text(json.dumps(submitted, indent=2))
    event("closed")


if __name__ == "__main__":
    main()
