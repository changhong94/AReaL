"""Reusable inference benchmark tooling; research policies remain opt-in."""

import fcntl
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_bench as bench


def session_call(address, action, **payload):
    # Host profiles need not install requests: urllib is sufficient.
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = json.dumps(dict(action=action, **payload)).encode()
    request = urllib.request.Request(
        "http://" + address + "/benchmark/session",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with opener.open(request, timeout=240) as response:
        return json.load(response)


def validate_session_topology(settings):
    if settings["npu_count"] != 2 or settings["inference_tp"] != 1:
        raise ValueError("Reusable sessions currently require two NPUs and TP=1")


def run(opts, profile):
    progress = bench.Progress(profile.get("timezone", "UTC"))
    active = None
    services = []
    lock = None
    workspace = None
    state = dict(
        phase="preparing",
        completed=[],
        evidence_class="development_only_not_formal_benchmark",
    )
    done = threading.Event()
    sample_thread = None
    try:
        progress.set("Preparing one immutable workload for repeated fixed/shared pairs")
        cmd = [
            sys.executable,
            str(bench.HERE / "run_bench.py"),
            "--prepare-only",
            "--profile",
            str(opts.profile),
            "--rl",
            opts.rl,
            "--code-reward-scheduling",
            getattr(opts, "code_reward_scheduling", "answer"),
            "--engine",
            opts.engine,
            "--concurrency",
            str(opts.concurrency),
            "--batches",
            str(opts.batches),
            "--batch-size",
            str(opts.batch_size),
            "--answers",
            str(opts.answers),
            "--npu-count",
            str(opts.npu_count),
            "--inference-training-ratio",
            opts.inference_training_ratio,
            "--inference-tp",
            str(opts.inference_tp),
            "--inference-mode",
            opts.inference_mode,
            "--adaptive-concurrency",
            opts.adaptive_concurrency,
            "--running-rebalance",
            opts.running_rebalance,
            "--max-head-offpolicyness",
            str(opts.max_head_offpolicyness),
        ]
        development_root = opts.workspace_root or profile.get(
            "development_workspace_root",
            str(
                Path(profile["workspace_root"]).expanduser().parent
                / "areal-development"
            ),
        )
        cmd += ["--workspace-root", str(development_root)]
        prepared = subprocess.run(cmd, text=True, capture_output=True, timeout=1800)
        if prepared.returncode:
            raise RuntimeError(prepared.stdout + prepared.stderr)
        workspace = Path(
            next(
                line.split("=", 1)[1]
                for line in prepared.stdout.splitlines()
                if line.startswith("WORKSPACE=")
            )
        )
        print("WORKSPACE=" + str(workspace), flush=True)
        (workspace / "session-prepare.log").write_text(
            prepared.stdout + prepared.stderr
        )
        settings = json.loads((workspace / "settings.json").read_text())
        validate_session_topology(settings)
        settings.update(
            repetitions=opts.repetitions,
            reuse_vllm=True,
            kv_policy=opts.kv_policy,
            evidence_class="development_only_not_formal_benchmark",
        )
        if opts.timeout:
            settings["timeout"] = opts.timeout
        bench.write_json(workspace / "settings.json", settings)
        profile = dict(profile, devices=settings["devices"])
        if opts.prepare_only:
            return
        lock = (Path.home() / ".cache/areal-benchmark.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, str(bench.HERE / "templates"))
        from telemetry import parse_info, sampler

        raw = bench.capture([profile["npu_smi"], "info"])
        devices, processes = parse_info(raw)
        bench.validate_idle_devices(raw, devices, processes, profile["devices"])
        infos = {}
        for arm in ("fixed", "latest"):
            progress.set(
                "Initializing and parking reusable " + arm + " inference services"
            )
            args, setup = bench.docker_args(profile, settings, workspace, arm)
            args[2:2] = [
                "-d",
                "-v",
                str(workspace) + ":/session:rw",
                "-e",
                "AREAL_KV_REBALANCE_POLICY=" + opts.kv_policy,
            ]
            name = settings["container_prefix"] + "-" + arm
            bench.command(
                args + [setup + "python3 /tool/reuse_service.py --arm " + arm],
                stdout=subprocess.DEVNULL,
            )
            services.append(name)
        for arm, name in zip(("fixed", "latest"), services, strict=True):
            progress.set("Validating reusable " + arm + " inference services")
            deadline = time.monotonic() + 1800
            manifest = workspace / ("services-" + arm + ".json")
            while not manifest.exists():
                if not bench.running(name):
                    raise RuntimeError(
                        "Reusable service startup failed; see service logs: " + arm
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError("Reusable service startup: " + arm)
                time.sleep(3)
            infos[arm] = json.loads(manifest.read_text())
        state["phase"] = "ready"
        bench.write_json(workspace / "session-state.json", state)
        for repetition in range(1, opts.repetitions + 1):
            pair = workspace / "repetitions" / f"{repetition:02d}"
            pair.mkdir(parents=True)
            for directory in ("source", "benchmark"):
                (pair / directory).symlink_to(
                    workspace / directory, target_is_directory=True
                )
            shutil.copytree(workspace / "data", pair / "data")
            for name in (
                "source-manifest.json",
                "adapter-manifest.json",
                "source.patch",
            ):
                shutil.copy2(workspace / name, pair / name)
            pair_settings = dict(
                settings,
                adaptive_concurrency=opts.shared_concurrency_modes[repetition - 1],
                shared_ablation=opts.shared_ablations[repetition - 1],
                shared_route_transport=opts.shared_route_transports[repetition - 1],
                container_prefix=settings["container_prefix"] + f"-r{repetition}",
            )
            bench.write_json(pair / "settings.json", pair_settings)
            for arm in ("fixed", "latest"):
                (pair / arm).mkdir()
                (pair / (arm + "-cache")).mkdir()
                shutil.copy2(
                    workspace / arm / "config.yaml", pair / arm / "config.yaml"
                )
            for arm in ("fixed", "latest"):
                progress.set(
                    f"Resetting initial DRAM weights: repetition {repetition}, {arm}"
                )
                # The other pool stays parked. Never wake both execution pools.
                trace = f"/session/repetitions/{repetition:02d}/{arm}/request-events"
                reset_started = time.monotonic()
                with ThreadPoolExecutor(max_workers=len(infos[arm])) as executor:
                    resets = list(
                        executor.map(
                            lambda row: session_call(
                                str(row["host"]) + ":" + str(row["port"]),
                                "reset",
                                trace_dir=trace,
                                seed=settings["seed"],
                            ),
                            infos[arm],
                        )
                    )
                for original, reset in zip(infos[arm], resets, strict=True):
                    before = original["capture"]["workers"]
                    after = reset["workers"]
                    if [x["worker_pid"] for x in before] != [
                        x["worker_pid"] for x in after
                    ]:
                        raise RuntimeError("Reusable worker identity changed")
                    if [x["initial_sha256"] for x in before] != [
                        x["restored_sha256"] for x in after
                    ]:
                        raise RuntimeError("Reusable initial weights changed")
                bench.write_json(
                    pair / arm / "reuse-reset.json",
                    dict(seconds=time.monotonic() - reset_started, services=resets),
                )
                (pair / arm / "npu-before-training.txt").write_text(
                    bench.capture([profile["npu_smi"], "info"])
                )
                args, setup = bench.docker_args(profile, pair_settings, pair, arm)
                servers = json.dumps(
                    [{k: row[k] for k in ("host", "port")} for row in infos[arm]]
                )
                args[2:2] = [
                    "-d",
                    "-e",
                    "AREAL_BENCH_SERVERS=" + servers,
                    "-e",
                    "AREAL_ROUTE_TRANSPORT="
                    + (
                        pair_settings["shared_route_transport"]
                        if arm == "latest"
                        else "poll"
                    ),
                    "-e",
                    "AREAL_DEV_ABLATION="
                    + (
                        pair_settings["shared_ablation"]
                        if arm == "latest"
                        else "baseline"
                    ),
                    "-e",
                    "AREAL_KV_REBALANCE_POLICY=" + opts.kv_policy,
                ]
                active = pair_settings["container_prefix"] + "-" + arm
                bench.command(
                    args
                    + [
                        setup
                        + f"python3 /tool/reuse_entry.py --config /work/{arm}/config.yaml"
                    ],
                    stdout=subprocess.DEVNULL,
                )
                state.update(phase="running", repetition=repetition, arm=arm)
                bench.write_json(workspace / "session-state.json", state)
                progress.set(
                    f"Training repetition {repetition}/{opts.repetitions}: {arm}"
                )
                done = threading.Event()
                sample_thread = threading.Thread(
                    target=sampler,
                    args=(pair / arm, done, profile["devices"]),
                    daemon=True,
                )
                sample_thread.start()
                offset = completed = 0
                deadline = time.monotonic() + settings["timeout"]
                while bench.running(active):
                    offset, completed = bench.report_events(
                        pair / arm / "phases.jsonl",
                        offset,
                        completed,
                        settings["batches"],
                        progress,
                    )
                    if time.monotonic() > deadline:
                        raise TimeoutError("Training arm exceeded timeout")
                    time.sleep(5)
                done.set()
                sample_thread.join(timeout=35)
                with (pair / arm / "train.log").open("w") as log:
                    bench.command(
                        ["docker", "logs", active], stdout=log, stderr=subprocess.STDOUT
                    )
                code = int(
                    bench.capture(
                        ["docker", "inspect", "--format", "{{.State.ExitCode}}", active]
                    )
                )
                active = None
                if code:
                    raise RuntimeError(
                        f"Training failed: repetition {repetition}, {arm}, exit {code}"
                    )
                parked = [
                    session_call(str(row["host"]) + ":" + str(row["port"]), "park")
                    for row in infos[arm]
                ]
                bench.write_json(pair / arm / "reuse-park.json", parked)
                state["completed"].append(dict(repetition=repetition, arm=arm))
                bench.write_json(workspace / "session-state.json", state)
            progress.set(f"Reporting fixed/shared pair {repetition} separately")
            bench.render(pair, profile)
        state["phase"] = "complete"
    except BaseException as exc:
        state.update(phase="failed", error=str(exc))
        raise
    finally:
        done.set()
        if sample_thread:
            sample_thread.join(timeout=35)
        for name in ([active] if active else []) + list(reversed(services)):
            subprocess.run(
                ["docker", "stop", "-t", "20", name], timeout=60, capture_output=True
            )
        if workspace:
            bench.write_json(workspace / "session-state.json", state)
        if lock:
            lock.close()
        progress.close()
