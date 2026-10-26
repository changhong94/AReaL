import fcntl
import json
import signal
import subprocess
import time
import traceback
from pathlib import Path

R = Path(__file__).resolve().parent
lock = (Path.home() / ".cache/areal-benchmark.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state = dict(phase="preflight", completed=[])
active = None


def save():
    (R / "state.json").write_text(json.dumps(state, indent=2))


def call(args):
    return subprocess.check_output(
        args, text=True, stderr=subprocess.STDOUT, timeout=120
    ).strip()


def stop(sig, frame):
    raise KeyboardInterrupt(sig)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
try:
    raw = call(["npu-smi", "info"])
    (R / "npu-before.txt").write_text(raw)
    assert raw.count("No running processes found in NPU") == 8, "NPU is occupied"
    image = call(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            "areal-timeshare:ascend-v0.26-a2-dev1",
        ]
    )
    state["image"] = image
    save()
    for label, tp in [("tp1a", 1), ("tp2", 2), ("tp1b", 1)]:
        active = "areal-r026-7b-" + label
        args = [
            "docker",
            "run",
            "-d",
            "--name",
            active,
            "--privileged",
            "--network",
            "host",
            "--shm-size",
            "16g",
            "-v",
            str(R) + ":/experiment",
            "-v",
            "/root/areal-assets/Qwen2-7B-Instruct:/model:ro",
            "-v",
            "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
        ]
        env = dict(
            ASCEND_RT_VISIBLE_DEVICES="2" if tp == 1 else "1,2",
            BENCH_TP=str(tp),
            BENCH_SESSION=label,
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            NO_PROXY="*",
            PYTHONUNBUFFERED="1",
            USE_OPTIMIZED_MODEL="0",
            OMP_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
            HCCL_CONNECT_TIMEOUT="120",
            HCCL_EXEC_TIMEOUT="300",
        )
        for k, v in env.items():
            args += ["-e", k + "=" + v]
        args += ["--entrypoint", "bash", image, "-lc", "python3 /experiment/bench.py"]
        state.update(phase=label, command=args)
        save()
        call(args)
        deadline = time.monotonic() + 2400
        with (R / (label + ".log")).open("w") as log:
            logger = subprocess.Popen(
                ["docker", "logs", "-f", active], stdout=log, stderr=subprocess.STDOUT
            )
            while (
                call(["docker", "inspect", "--format", "{{.State.Running}}", active])
                == "true"
            ):
                if time.monotonic() > deadline:
                    raise TimeoutError(label)
                time.sleep(5)
            logger.wait(timeout=30)
        assert (
            call(["docker", "inspect", "--format", "{{.State.ExitCode}}", active])
            == "0"
        ), label + " failed"
        state["completed"].append(label)
        active = None
        save()
    state["phase"] = "complete"
except BaseException as exc:
    state.update(phase="failed", error=repr(exc), traceback=traceback.format_exc())
finally:
    if active:
        subprocess.run(["docker", "stop", "-t", "20", active], timeout=60)
    state["ended"] = time.time()
    save()
