#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Force one real vLLM KV handoff and compare deterministic tokens/logprobs."""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer

from areal.infra.kv_rebalance import initialize_pair, rpc
from areal.utils.network import find_free_ports


def post(address, endpoint, payload, routed=False):
    headers = (
        {
            "X-Areal-Request-Routing": "1",
            "X-Areal-Route-Rank": "0",
            "X-Areal-Route-Version": "0",
        }
        if routed
        else {}
    )
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(
            f"http://{address}/{endpoint}", json=payload, headers=headers, timeout=180
        )
        response.raise_for_status()
        return response.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
    if len(devices) != 2:
        raise ValueError("Probe needs exactly two devices")
    ports = find_free_ports(2)
    addresses = [f"127.0.0.1:{p}" for p in ports]
    processes = []
    try:
        for i, port in enumerate(ports):
            env = dict(
                os.environ,
                ASCEND_RT_VISIBLE_DEVICES=devices[i],
                AREAL_RUNNING_REBALANCE="kv",
                PYTORCH_NPU_ALLOC_CONF="expandable_segments:False",
                AREAL_AUTO_MEMORY="1",
                AREAL_DYNAMIC_KV="1",
                AREAL_KV_SLAB_BLOCKS="16",
                AREAL_DRAIN_METRICS="1",
                AREAL_QUEUE_REBALANCE="1",
            )
            cmd = [
                sys.executable,
                "-m",
                "areal.engine.vllm_ext.areal_vllm_server",
                "--model",
                args.model,
                "--tokenizer",
                args.model,
                "--trust-remote-code",
                "--seed",
                "1",
                "--enforce-eager",
                "--dtype",
                "bfloat16",
                "--distributed-executor-backend",
                "uni",
                "--load-format",
                "auto",
                "--max-num-batched-tokens",
                "2048",
                "--max-num-seqs",
                "64",
                "--block-size",
                "16",
                "--disable-sliding-window",
                "--max-model-len",
                "4096",
                "--no-enable-prefix-caching",
                "--gpu-memory-utilization",
                "0.4",
                "--worker-extension-cls",
                "areal.engine.vllm_ext.vllm_worker_extension.VLLMWorkerExtension",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
            log = (args.output / f"server-{i}.log").open("w")
            processes.append(
                subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            )
        deadline = time.monotonic() + 360
        for address in addresses:
            while True:
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError("Probe server exited")
                try:
                    rpc(address, "info", {})
                    break
                except requests.RequestException:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(1)
        geometry = initialize_pair(addresses, 0, 1)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompt = tokenizer.apply_chat_template(
            [
                dict(
                    role="user",
                    content="Explain step by step how to calculate the sum of integers from 1 to 100, then verify it using a second method.",
                )
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        params = dict(temperature=0.0, max_tokens=256, ignore_eos=True, logprobs=0)
        payload = dict(
            request_id="kv-probe",
            token_ids=prompt,
            sampling_params=params,
            stream=False,
        )
        baseline = post(
            addresses[0],
            "inference/v1/generate",
            dict(payload, request_id="kv-baseline"),
            True,
        )
        with ThreadPoolExecutor(max_workers=3) as executor:
            generation = executor.submit(
                post, addresses[0], "inference/v1/generate", payload, True
            )
            ticket = uuid.uuid4().hex
            deadline = time.monotonic() + 30
            while True:
                state = post(
                    addresses[0], "areal_queue_rebalance", dict(action="snapshot")
                )
                active = [
                    r
                    for r in state["active"]
                    if r["request_id"] == "kv-probe" and r["tokens"] >= 32
                ]
                if active:
                    internal = state["internal_ids"]["kv-probe"]
                    exported = rpc(
                        addresses[0], "export", dict(ticket=ticket, request_id=internal)
                    )
                    if "skipped" not in exported:
                        break
                if generation.done() or time.monotonic() > deadline:
                    raise RuntimeError("Could not freeze an in-flight request")
                time.sleep(0.005)
            data = dict(ticket=ticket, bytes=exported["bytes"], geometry=geometry)
            jobs = [
                executor.submit(
                    rpc, addresses[i], "transfer", dict(data, source=i == 0)
                )
                for i in range(2)
            ]
            for job in jobs:
                job.result()
            rpc(
                addresses[1],
                "register",
                dict(exported, ticket=ticket, external_id="kv-probe", version=0),
            )
            post(
                addresses[0],
                "areal_queue_rebalance",
                dict(action="abort_running", internal_ids=[internal]),
            )
            prefix = generation.result()
        choice = prefix["choices"][0]
        continuation = post(
            addresses[1],
            "inference/v1/generate",
            dict(
                payload,
                token_ids=prompt + choice["token_ids"],
                sampling_params=dict(params, max_tokens=256 - len(choice["token_ids"])),
            ),
            True,
        )
        b = baseline["choices"][0]
        c = continuation["choices"][0]
        tokens = choice["token_ids"] + c["token_ids"]
        logs = choice["logprobs"]["content"] + c["logprobs"]["content"]
        error = max(
            abs(x["logprob"] - y["logprob"])
            for x, y in zip(logs, b["logprobs"]["content"])
        )
        applied = rpc(addresses[1], "import_status", {})["results"][ticket]
        if applied["outcome"] != "import_applied":
            raise RuntimeError("Probe replayed instead of importing KV")
        result = dict(
            import_result=applied,
            tokens_equal=tokens == b["token_ids"],
            logprob_max_error=error,
            prefix_tokens=len(choice["token_ids"]),
            export=exported,
            total_tokens=len(tokens),
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        if not result["tokens_equal"] or error > 0.01 or len(logs) != 256:
            raise RuntimeError("KV continuation differs from deterministic baseline")
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
