#!/usr/bin/env python3
"""Run a reproducible fixed/shared two-NPU comparison on the local Linux host."""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent


class Progress:
    """Low-volume host-side progress; no device synchronization or training changes."""

    def __init__(self, timezone):
        self.zone = ZoneInfo(timezone)
        self.stage = "Starting"
        self.started = time.perf_counter()
        self.last = self.started
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def message(self, text):
        with self.lock:
            stamp = dt.datetime.now(self.zone).strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"[{stamp}] {text}", flush=True)
            self.last = time.perf_counter()

    def set(self, text):
        with self.lock:
            self.stage = text
            self.started = time.perf_counter()
            self.message(text)

    def watch(self):
        while not self.done.wait(5):
            with self.lock:
                now = time.perf_counter()
                if now - self.last >= 60:
                    self.message(
                        f"Waiting: {self.stage} ({now - self.started:.0f}s elapsed; completion not yet reported)"
                    )

    def close(self):
        self.done.set()
        self.thread.join(timeout=6)


class ResourceHooks:
    """Optional host commands; never run for preparation or rendering alone."""

    def __init__(self, profile, workspace, devices, progress):
        self.commands = profile.get("resource_hooks", {})
        self.workspace = workspace
        self.devices = devices
        self.progress = progress
        self.started = False
        self.finished = False

    def invoke(self, name):
        argv = self.commands.get(name)
        if not argv:
            return
        if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            raise ValueError("resource_hooks commands must be argument lists")
        self.progress.set(f"Running host resource hook: {name}")
        env = dict(
            os.environ,
            AREAL_BENCH_WORKSPACE=str(self.workspace),
            AREAL_BENCH_DEVICES=",".join(map(str, self.devices)),
        )
        with (self.workspace / f"resource-{name}.log").open("a") as log:
            subprocess.run(
                argv,
                check=True,
                timeout=self.commands.get("timeout", 120),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

    def start(self):
        self.started = True
        self.invoke("before")

    def finish(self):
        if self.started and not self.finished:
            self.finished = True
            self.invoke("after")


def report_events(path, offset, completed, total, progress):
    """Read appended phase records, retaining partial lines for the next poll."""
    if not path.exists():
        return offset, completed
    with path.open() as stream:
        stream.seek(offset)
        while True:
            line = stream.readline()
            if not line or not line.endswith("\n"):
                break
            offset = stream.tell()
            try:
                event = json.loads(line).get("event")
            except ValueError:
                continue
            if event == "initialize.end":
                progress.message("Model initialization complete; starting training")
            elif event == "_finish_step.end":
                completed += 1
                progress.message(f"Training batch {completed}/{total} complete")
            elif event == "train.end":
                progress.message("Training loop complete; waiting for worker shutdown")
    return offset, completed


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def npu_count(value):
    count = positive(value)
    if count < 2:
        raise argparse.ArgumentTypeError("NPU count must be at least 2")
    return count


def allocation_ratio(value):
    try:
        inference, training = map(int, value.split(":"))
        if inference < 1 or training < 1:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use positive inference:training integers, e.g. 2:1"
        ) from exc
    return value


def allocation_counts(count, ratio="1:1"):
    """Split physical devices exactly without rounding either group."""
    inference, training = map(int, allocation_ratio(ratio).split(":"))
    if count * training % (inference + training):
        raise ValueError(
            "NPU count cannot represent the requested inference:training ratio exactly"
        )
    train = count * training // (inference + training)
    return dict(
        total=count,
        training=train,
        dedicated_inference=count - train,
        shared_inference_max=count,
    )


def validate_topology(count, inference_tp=1, ratio="1:1"):
    if count < 2:
        raise ValueError("At least two NPUs are required")
    allocation = allocation_counts(count, ratio)
    if inference_tp < 1 or any(
        allocation[key] % inference_tp for key in ("training", "dedicated_inference")
    ):
        raise ValueError(
            "Inference TP must divide both training and dedicated inference groups"
        )


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rl", choices=["math", "code"], default="math")
    p.add_argument(
        "--code-reward-scheduling",
        choices=["answer", "question"],
        default="answer",
        help="experimental R029 shared-arm scoring policy; fixed arm remains answer",
    )
    p.add_argument(
        "--concurrency",
        type=positive,
        default=16,
        help="concurrent questions, not answers or questions per batch",
    )
    p.add_argument("--engine", choices=["fsdp", "megatron"], default="fsdp")
    p.add_argument("--batches", type=positive, default=6)
    p.add_argument(
        "--batch-size",
        type=positive,
        default=32,
        help="questions per training batch (default: 32); independent of concurrency",
    )
    p.add_argument(
        "--answers",
        type=positive,
        default=8,
        help="sampled answers per question (default: 8)",
    )
    p.add_argument(
        "--adaptive-concurrency",
        choices=["off", "capacity"],
        default="capacity",
        help="R025 shared-only soft question admission; default capacity (off disables)",
    )
    p.add_argument(
        "--inference-mode",
        choices=["graph", "eager"],
        default="graph",
        help="vLLM execution mode (default: graph); training execution is unchanged",
    )
    p.add_argument(
        "--running-rebalance",
        choices=["off", "shadow", "budget", "active", "kv"],
        default="kv",
        help="R001 policy for the shared arm only (default: kv; requires equal-TP replicas)",
    )
    p.add_argument(
        "--max-head-offpolicyness",
        type=nonnegative,
        default=2,
        help="rollout head off-policy limit for both arms (default: 2; 0 allows no stale lookahead)",
    )
    p.add_argument(
        "--npu-count",
        type=npu_count,
        default=2,
        help="total NPUs; divided according to --inference-training-ratio",
    )
    p.add_argument(
        "--inference-training-ratio",
        type=allocation_ratio,
        default="1:1",
        help="dedicated inference:training device ratio (default: 1:1)",
    )
    p.add_argument(
        "--inference-tp",
        type=positive,
        default=1,
        help="devices per vLLM instance; must divide each device group",
    )
    p.add_argument(
        "--profile",
        type=Path,
        default=Path.home() / ".config/areal-benchmark/profile.json",
    )
    p.add_argument("--workspace-root", type=Path)
    p.add_argument(
        "--timeout",
        type=positive,
        help="per-arm timeout in seconds; default max(3600, batches*600)",
    )
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="freeze source, select data and validate configuration without using NPUs or creating sandboxes",
    )
    p.add_argument(
        "--render-only",
        type=Path,
        help="regenerate charts from an existing workspace, without NPU access",
    )
    return p.parse_args()


def command(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, text=True, **kwargs)


def capture(args):
    return command(args, stdout=subprocess.PIPE).stdout.strip()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def running(name):
    return (
        capture(["docker", "inspect", "--format", "{{.State.Running}}", name]) == "true"
    )


def render(workspace, profile):
    renderer = workspace / "benchmark/render.py"
    command(
        [
            profile["plot_python"],
            renderer if renderer.exists() else HERE / "render.py",
            workspace,
        ]
    )


def freeze(repo, dest):
    names = capture(
        [
            "git",
            "-C",
            repo,
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ]
    ).split("\0")
    hashes = {}
    for name in sorted(set(filter(None, names))):
        src = repo / name
        if not src.is_file():
            continue
        if src.is_symlink():
            raise RuntimeError(f"Source symlink needs explicit review: {name}")
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        hashes[name] = hashlib.sha256(src.read_bytes()).hexdigest()
    return dict(head=capture(["git", "-C", repo, "rev-parse", "HEAD"]), files=hashes)


def environment(shared):
    env = {
        k: str(int(shared))
        for k in [
            "AREAL_BATCH_DRAIN_METRICS",
            "AREAL_BATCH_DRAIN",
            "AREAL_DRAIN_METRICS",
            "AREAL_ADAPTIVE_DRAIN",
            "AREAL_QUEUE_REBALANCE",
            "AREAL_PRESSURE_MEMORY",
            "AREAL_DYNAMIC_KV",
            "AREAL_PROFIT_AWARE_RETURN",
            "AREAL_RESIDENT_BUDGET",
        ]
    }
    env.update(
        AREAL_MEMORY_DIAGNOSTICS=os.getenv("AREAL_MEMORY_DIAGNOSTICS", ""),
        AREAL_ROUTE_TRANSPORT=os.getenv("AREAL_ROUTE_TRANSPORT", "push"),
        AREAL_RETURN_CALIBRATION_PROBE=os.getenv("AREAL_RETURN_CALIBRATION_PROBE", "1"),
        AREAL_TRAIN_PRIORITY_INFERENCE=(
            os.getenv("AREAL_TRAIN_PRIORITY_INFERENCE", "0") if shared else "0"
        ),
        AREAL_IDLE_FULL_VERIFY=os.getenv("AREAL_IDLE_FULL_VERIFY", "1"),
        AREAL_IDLE_HELPER_VALIDATE=os.getenv("AREAL_IDLE_HELPER_VALIDATE", "0"),
        AREAL_IDLE_HELPER_LEAN=os.getenv("AREAL_IDLE_HELPER_LEAN", "0"),
        AREAL_IDLE_COMPARE_HELPER=os.getenv("AREAL_IDLE_COMPARE_HELPER", "0"),
        AREAL_IDLE_ASYNC_TAIL=os.getenv("AREAL_IDLE_ASYNC_TAIL", "1"),
        AREAL_IDLE_HELPER=os.getenv("AREAL_IDLE_HELPER", "0"),
        AREAL_IDLE_MID_BATCH=os.getenv("AREAL_IDLE_MID_BATCH", "0"),
        AREAL_RETAIN_DRAM=os.getenv("AREAL_RETAIN_DRAM", "0"),
        AREAL_IDLE_CHUNK_ANSWERS=os.getenv("AREAL_IDLE_CHUNK_ANSWERS", "32"),
        AREAL_IDLE_SWITCH_BUDGET=os.getenv("AREAL_IDLE_SWITCH_BUDGET", "1"),
        AREAL_IDLE_ENTRY_VALIDATE_WORLDS=os.getenv(
            "AREAL_IDLE_ENTRY_VALIDATE_WORLDS", ""
        ),
        AREAL_IDLE_SWITCH_COST_SECONDS=os.getenv("AREAL_IDLE_SWITCH_COST_SECONDS", "6"),
        AREAL_IDLE_SWITCH_SPEEDUP=os.getenv("AREAL_IDLE_SWITCH_SPEEDUP", "1.5"),
        AREAL_IDLE_SWITCH_MARGIN_SECONDS=os.getenv(
            "AREAL_IDLE_SWITCH_MARGIN_SECONDS", "0"
        ),
        AREAL_IDLE_TRAINING="1"
        if shared and os.getenv("AREAL_IDLE_TRAINING") == "1"
        else "0",
        AREAL_SHARED_DRAIN_SECONDS="0",
        AREAL_NPU_SHARE_PAUSE_KEEP="0",
        AREAL_ASCEND_KERNEL_WARMUP="1",
        AREAL_OVERLAP_TRAIN_RESTORE="1",
        AREAL_PARK_TRAIN_CACHE="1",
        AREAL_NPU_DEVICE_TRACE="1",
        EXPERIMENT_TASK_AFFINITY="0",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        WANDB_MODE="disabled",
        NO_PROXY="*",
        no_proxy="*",
        PYTHONUNBUFFERED="1",
        USE_OPTIMIZED_MODEL="0",
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        HCCL_CONNECT_TIMEOUT="120",
        HCCL_EXEC_TIMEOUT="300",
    )
    # Drain policies require inference to remain active during training.
    # Keep metrics available even when that concurrent execution is disabled.
    if env["AREAL_TRAIN_PRIORITY_INFERENCE"] != "1":
        env["AREAL_ADAPTIVE_DRAIN"] = "0"
        env["AREAL_BATCH_DRAIN"] = "0"
    return env


def docker_args(profile, settings, workspace, arm=None):
    spec = profile["engines"][settings["engine"]]
    args = ["docker", "run", "--network", "host"]
    if arm:
        args += [
            "--privileged",
            "--shm-size",
            "64g"
            if arm == "latest" and os.getenv("AREAL_IDLE_TRAINING") == "1"
            else "16g",
            "--name",
            settings["container_prefix"] + "-" + arm,
        ]
    else:
        args += ["--rm"]
    mounts = [
        (workspace / "source", "/repo", "ro"),
        (workspace / "benchmark/templates", "/tool", "ro"),
        (workspace, "/work", "rw"),
        (Path(profile["model"]), "/model", "ro"),
    ]
    data = profile["datasets"][settings["rl"]]
    for split in ["train", "test"]:
        mounts.append((Path(data[split]), "/input/" + split + ".parquet", "ro"))
    env = environment(arm == "latest")
    env["AREAL_ADAPTIVE_CONCURRENCY"] = (
        settings.get("adaptive_concurrency", "off") if arm == "latest" else "off"
    )
    env["AREAL_RUNNING_REBALANCE"] = (
        settings.get("running_rebalance", "off") if arm == "latest" else "off"
    )
    if arm == "latest" and settings.get("inference_tp", 1) > 1:
        env["AREAL_NPU_SHARE_PAUSE_KEEP"] = "1"
    env.update(PYTHONPATH="/tool:/repo", AREAL_BENCH_WORKSPACE="/work")
    if arm:
        mounts += [
            (workspace / arm, "/run-output", "rw"),
            (workspace / (arm + "-cache"), "/tmp/areal-root/.cache/root/triton", "rw"),
            (Path(profile["driver"]), profile["driver"], "ro"),
            (Path(profile["npu_smi"]), "/usr/local/bin/npu-smi", "ro"),
        ]
        env.update(
            ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, profile["devices"])),
            AREAL_REQUEST_TRACE_DIR="/run-output/request-events",
        )
    if settings["rl"] == "code":
        c = profile["code"]
        mounts += [
            (Path(c["dependencies"]), "/deps", "ro"),
            (Path(c["sdk"]), "/e2b-v250", "ro"),
            (Path(c["config_dir"]), c["container_config_dir"], "ro"),
        ]
        env.update(
            AREAL_CODE_REWARD_SCHEDULING=(
                settings.get("code_reward_scheduling", "answer")
                if arm == "latest"
                else "answer"
            ),
            PYTHONPATH="/tool:/repo:/e2b-v250:/deps",
            KUASAR_CONFIG=c["container_config_dir"] + "/access.json",
            SSL_CERT_FILE=c["container_config_dir"] + "/ca.crt",
            AREAL_CODE_RUN_ID=settings["container_prefix"] + "-" + (arm or "preflight"),
        )
    for src, dst, mode in mounts:
        args += ["-v", f"{src}:{dst}:{mode}"]
    for k, v in env.items():
        args += ["-e", k + "=" + v]
    args += ["-w", "/repo", "--entrypoint", "bash", settings["image_id"], "-lc"]
    setup = (
        "set +eu; "
        + "; ".join("source " + shlex.quote(x) for x in profile["cann_env"])
        + "; set -eu; "
    )
    setup += "export PATH=" + shlex.quote(str(Path(spec["python"]).parent)) + ":$PATH; "
    return args, setup


def validate_idle_devices(raw, devices, processes, selected):
    """Require explicit idle evidence for every selected device on shared hosts."""
    selected = {str(device) for device in selected}
    idle = set(re.findall(r"No running processes found in NPU\s+(\d+)\b", raw))
    busy = {str(process["device"]) for process in processes}
    unhealthy = {
        device for device in selected if devices.get(device, {}).get("health") != "OK"
    }
    if (
        "Process id" not in raw
        or not selected
        or not selected <= idle
        or selected & busy
        or unhealthy
    ):
        raise RuntimeError(
            "Selected NPU devices are busy, unhealthy, or lack explicit idle evidence; "
            "no experiment started"
        )


def main():
    opts = arguments()
    print(f"Loading profile: {opts.profile}", flush=True)
    profile = json.loads(opts.profile.read_text())
    if opts.render_only:
        progress = Progress(profile.get("timezone", "UTC"))
        progress.set("Generating charts")
        try:
            render(opts.render_only.resolve(), profile)
            progress.message(
                f"Charts ready: {opts.render_only.resolve() / 'index.html'}"
            )
        finally:
            progress.close()
        return
    validate_topology(opts.npu_count, opts.inference_tp, opts.inference_training_ratio)
    allocation = allocation_counts(opts.npu_count, opts.inference_training_ratio)
    repo = Path(profile["repo"]).resolve()
    root = (opts.workspace_root or Path(profile["workspace_root"])).resolve()
    if root == repo or repo in root.parents:
        raise ValueError("Workspace must be outside the source repository")
    available = profile.get("available_devices", list(range(8)))
    selected = (
        profile["devices"]
        if opts.npu_count == len(profile["devices"])
        else available[: opts.npu_count]
    )
    if len(selected) != opts.npu_count or len(set(selected)) != len(selected):
        raise ValueError("Host profile does not provide enough distinct NPU devices")
    profile["devices"] = selected
    for path in [
        repo,
        Path(profile["model"]),
        *map(Path, profile["datasets"][opts.rl].values()),
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
    root.mkdir(parents=True, exist_ok=True)
    # One process across all workspaces on this machine owns benchmark hardware.
    lockpath = Path.home() / ".cache/areal-benchmark.lock"
    lockpath.parent.mkdir(exist_ok=True)
    lock = lockpath.open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stamp = dt.datetime.now(ZoneInfo(profile.get("timezone", "UTC"))).strftime(
        "%Y%m%d-%H%M%S%z"
    )
    workspace = (
        root
        / f"{stamp}_{opts.rl}_{opts.engine}_c{opts.concurrency}_b{opts.batches}_{uuid.uuid4().hex[:6]}"
    )
    workspace.mkdir()
    print(f"WORKSPACE={workspace}", flush=True)
    settings = dict(
        rl=opts.rl,
        code_reward_scheduling=opts.code_reward_scheduling,
        code_reward_scheduling_by_arm={
            "fixed": "answer",
            "latest": opts.code_reward_scheduling,
        },
        engine=opts.engine,
        concurrency=opts.concurrency,
        max_head_offpolicyness=opts.max_head_offpolicyness,
        batches=opts.batches,
        running_rebalance=opts.running_rebalance,
        adaptive_concurrency=opts.adaptive_concurrency,
        inference_mode=opts.inference_mode,
        batch_size=opts.batch_size,
        answers=opts.answers,
        questions=opts.batch_size * opts.batches,
        seed=1,
        max_new_tokens=1024,
        timezone=profile.get("timezone", "UTC"),
        created=stamp,
        timeout=opts.timeout or max(3600, opts.batches * 600),
        container_prefix="areal-bench-" + uuid.uuid4().hex[:12],
        image=profile["engines"][opts.engine]["image"],
        devices=profile["devices"],
    )
    settings.update(
        npu_count=opts.npu_count,
        inference_tp=opts.inference_tp,
        allocation=allocation,
        inference_training_ratio=opts.inference_training_ratio,
        inference_replicas=allocation["dedicated_inference"] // opts.inference_tp,
    )
    write_json(workspace / "settings.json", settings)
    state = dict(phase="preparing", completed=[], workspace=str(workspace))
    active = None
    done = threading.Event()
    sampler_thread = None
    error = None

    def save():
        write_json(workspace / "state.json", state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    progress = Progress(profile.get("timezone", "UTC"))
    progress.message(
        f"NPU topology: devices={selected}; training={allocation['training']}; dedicated vLLM instances={allocation['dedicated_inference'] // opts.inference_tp}; TP={opts.inference_tp}"
    )
    if opts.inference_tp > 1:
        progress.message(
            "TP shared instances use per-device budgets and all-member mapping acknowledgements"
        )
    progress.set(
        f"Preparing {opts.rl} / {opts.engine} / inference={opts.inference_mode}: fixed + shared, {opts.batches} batches each, {opts.batch_size} questions/batch, {opts.answers} answers/question, concurrency {opts.concurrency}"
    )
    hooks = ResourceHooks(profile, workspace, selected, progress)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        save()
        settings["image_id"] = capture(
            ["docker", "image", "inspect", "--format", "{{.Id}}", settings["image"]]
        )
        write_json(workspace / "settings.json", settings)
        if not opts.prepare_only:
            command([profile["plot_python"], "-c", "import matplotlib, numpy, PIL"])
        progress.set("Freezing source and benchmark configuration")
        shutil.copytree(
            HERE / "templates",
            workspace / "benchmark/templates",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(HERE / "render.py", workspace / "benchmark/render.py")
        shutil.copy2(HERE / "run_bench.py", workspace / "benchmark/run_bench.py")
        write_json(
            workspace / "adapter-manifest.json",
            {
                str(f.relative_to(workspace / "benchmark")): hashlib.sha256(
                    f.read_bytes()
                ).hexdigest()
                for f in (workspace / "benchmark").rglob("*")
                if f.is_file()
            },
        )
        write_json(
            workspace / "source-manifest.json", freeze(repo, workspace / "source")
        )
        (workspace / "source.patch").write_text(
            capture(["git", "-C", repo, "diff", "HEAD"])
        )
        for arm in ["fixed", "latest"]:
            (workspace / arm).mkdir()
            (workspace / (arm + "-cache")).mkdir()
        args, setup = docker_args(profile, settings, workspace)
        progress.set(
            f"Preparing data and validating configurations; log: {workspace / 'prepare.log'}"
        )
        with (workspace / "prepare.log").open("w") as log:
            command(
                args + [setup + "python3 /tool/prepare.py"],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        if opts.prepare_only:
            state["phase"] = "prepared"
            progress.message("Preparation complete; no NPU training requested")
            return
        if opts.rl == "code":
            progress.set("Checking sandbox create / execute / delete")
            with (workspace / "sandbox-preflight.log").open("w") as log:
                command(
                    args + [setup + "python3 /tool/sandbox_check.py"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                )
        sys.path.insert(0, str(HERE / "templates"))
        from telemetry import parse_info, sampler

        hooks.start()
        progress.set("Checking NPU availability")
        raw = capture([profile["npu_smi"], "info"])
        devices, processes = parse_info(raw)
        validate_idle_devices(raw, devices, processes, profile["devices"])
        for arm in ["fixed", "latest"]:
            progress.set(f"Starting {arm} group: model initialization and training")
            event_offset, completed_batches = 0, 0
            state.update(phase="running", current=arm)
            save()
            args, setup = docker_args(profile, settings, workspace, arm)
            active = settings["container_prefix"] + "-" + arm
            # Detached container plus bounded host supervisor permits laptop disconnects.
            args.insert(2, "-d")
            command(
                args
                + [setup + f"python3 /tool/entry.py --config /work/{arm}/config.yaml"],
                stdout=subprocess.DEVNULL,
            )
            done = threading.Event()
            sampler_thread = threading.Thread(
                target=sampler,
                args=(workspace / arm, done, profile["devices"]),
                daemon=True,
            )
            sampler_thread.start()
            deadline = time.monotonic() + settings["timeout"]
            progress.message(
                f"Container: {active}; live details: docker logs -f {active}"
            )
            while running(active):
                event_offset, completed_batches = report_events(
                    workspace / arm / "phases.jsonl",
                    event_offset,
                    completed_batches,
                    opts.batches,
                    progress,
                )
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{arm} exceeded {settings['timeout']} seconds")
                time.sleep(5)
            report_events(
                workspace / arm / "phases.jsonl",
                event_offset,
                completed_batches,
                opts.batches,
                progress,
            )
            done.set()
            sampler_thread.join(timeout=35)
            with (workspace / arm / "train.log").open("w") as log:
                command(
                    ["docker", "logs", active], stdout=log, stderr=subprocess.STDOUT
                )
            code = int(
                capture(
                    ["docker", "inspect", "--format", "{{.State.ExitCode}}", active]
                )
            )
            if code:
                raise RuntimeError(
                    f"{arm} training exited {code}; see {workspace / arm / 'train.log'}"
                )
            state["completed"].append(arm)
            save()
            progress.message(
                f"{arm} group finished successfully; logs: {workspace / arm}"
            )
            active = None
        hooks.finish()
        progress.set("Generating four chart families")
        state["phase"] = "rendering"
        save()
        render(workspace, profile)
        state["phase"] = "complete"
        progress.message(f"Comparison complete: {workspace / 'index.html'}")
    except BaseException as exc:
        error = exc
        state.update(phase="failed", error=str(exc))
        progress.message(f"FAILED: {exc}; details: {workspace}")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        done.set()
        if sampler_thread:
            sampler_thread.join(timeout=35)
        try:
            if active:
                progress.set(
                    "Stopping experiment container after failure or interruption"
                )
                stopped = subprocess.run(
                    ["docker", "stop", "-t", "20", active],
                    timeout=60,
                    capture_output=True,
                )
                if stopped.returncode != 0 or running(active):
                    raise RuntimeError("Experiment container did not stop")
                with (workspace / state["current"] / "train.log").open("w") as log:
                    subprocess.run(
                        ["docker", "logs", active],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=60,
                    )
        except Exception as exc:
            state.update(phase="failed", cleanup_error=str(exc))
            error = error or exc
        finally:
            if "cleanup_error" not in state:
                try:
                    hooks.finish()
                except Exception as exc:
                    state.update(phase="failed", resource_hook_error=str(exc))
                    error = error or exc
            state["ended"] = time.time()
            save()
            progress.message(
                f"Final status: {state['phase']}; state file: {workspace / 'state.json'}"
            )
            progress.close()
            lock.close()
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
