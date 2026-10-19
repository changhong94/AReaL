"""Own inference processes independently of short-lived trainer containers."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import requests

from areal.api.cli_args import GRPOConfig, load_expr_config, vLLMConfig
from areal.engine.vllm_remote import VLLMBackend
from areal.utils.network import find_free_ports, gethostip


def call(address, action, **payload):
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(
            "http://" + address + "/benchmark/session",
            json=dict(action=action, **payload),
            timeout=180,
        )
        response.raise_for_status()
        return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["fixed", "latest"], required=True)
    opts = parser.parse_args()
    root = Path("/work")
    settings = json.loads((root / "settings.json").read_text())
    config, _ = load_expr_config(
        ["--config", str(root / opts.arm / "config.yaml")], GRPOConfig
    )
    devices = settings["devices"] if opts.arm == "latest" else settings["devices"][1:]
    children, infos = [], []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for device in devices:
            args = vLLMConfig.build_args(config.vllm, tp_size=1, pp_size=1)
            args.update(
                host=gethostip(),
                port=find_free_ports(1)[0],
                worker_extension_cls="reuse_worker.ReuseWorkerExtension",
            )
            # Native Ascend sleep/wake requires ND weights for safe RL restores.
            additional = args.get("additional_config") or {}
            if isinstance(additional, str):
                additional = json.loads(additional)
            args["additional_config"] = json.dumps(dict(additional, weight_nz_mode=0))
            if opts.arm == "fixed":
                args["enable_sleep_mode"] = True
            env = VLLMBackend.build_server_env(os.environ)
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
            if opts.arm == "latest":
                env.update(
                    AREAL_AUTO_MEMORY="1",
                    AREAL_KV_SLAB_BLOCKS="16",
                    AREAL_DYNAMIC_KV="1",
                    PYTORCH_NPU_ALLOC_CONF="expandable_segments:False",
                    HCCL_NPU_SOCKET_PORT_RANGE="auto",
                    AREAL_DYNAMIC_KV_RESERVE_BYTES=str(
                        int(
                            (
                                config.colocation_memory.inference_workspace_gib
                                + config.colocation_memory.safety_margin_gib
                            )
                            * 1024**3
                        )
                    ),
                    AREAL_DYNAMIC_KV_DEVICE_BUDGET_BYTES=str(
                        int((config.colocation_memory.device_budget_gib or 0) * 1024**3)
                    ),
                )
            cmd = vLLMConfig.build_cmd_from_args(args)
            module = cmd.index("-m") + 1
            cmd[module] = "reuse_server"
            log = (root / f"service-{opts.arm}-{device}.log").open("w")
            process = subprocess.Popen(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            children.append((process, log))
            address = str(args["host"]) + ":" + str(args["port"])
            deadline = time.monotonic() + 900
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Inference service exited: " + address)
                try:
                    with requests.Session() as session:
                        session.trust_env = False
                        if (
                            session.get(
                                "http://" + address + "/health", timeout=2
                            ).status_code
                            == 200
                        ):
                            break
                except requests.RequestException:
                    pass
                if stopping or time.monotonic() > deadline:
                    raise TimeoutError("Inference service startup")
                time.sleep(2)
            capture = call(address, "capture")
            parked = call(address, "park")
            call(
                address,
                "reset",
                trace_dir="/session/service-events",
                seed=settings["seed"],
            )
            parked = call(address, "park")
            infos.append(
                dict(
                    host=args["host"],
                    port=args["port"],
                    device=device,
                    capture=capture,
                    parked=parked,
                )
            )
        (root / ("services-" + opts.arm + ".json")).write_text(
            json.dumps(infos, indent=2)
        )
        while not stopping:
            if any(p.poll() is not None for p, _ in children):
                raise RuntimeError("Persistent service exited unexpectedly")
            time.sleep(1)
    finally:
        for p, _ in children:
            if p.poll() is None:
                p.terminate()
        for p, log in children:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            log.close()


if __name__ == "__main__":
    main()
