#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate real vLLM KV handoffs across all directed equal-TP replica pairs."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer

from areal.infra.kv_rebalance import initialize_pairs, pair_key, rpc
from areal.utils.network import find_free_ports


def validate_logprobs(response):
    """Do not let NaN/Inf bypass numerical comparison tolerances."""
    for choice in response.get("choices", []):
        for token in (choice.get("logprobs") or {}).get("content", []):
            if not math.isfinite(token["logprob"]):
                raise RuntimeError("Generation returned a non-finite logprob")


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
        result = response.json()
        if endpoint == "inference/v1/generate":
            validate_logprobs(result)
        return result


def probe_pair(addresses, source, target, geometry, prompt, retained=False):
    params = dict(temperature=0.0, max_tokens=256, ignore_eos=True, logprobs=0)
    payload = dict(
        request_id="kv-probe",
        token_ids=prompt,
        sampling_params=params,
        stream=False,
    )
    baseline = post(
        addresses[source],
        "inference/v1/generate",
        dict(payload, request_id="kv-baseline"),
        True,
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        generation = executor.submit(
            post, addresses[source], "inference/v1/generate", payload, True
        )
        ticket = uuid.uuid4().hex
        deadline = time.monotonic() + 30
        while True:
            state = post(
                addresses[source], "areal_queue_rebalance", dict(action="snapshot")
            )
            active = [
                r
                for r in state["active"]
                if r["request_id"] == "kv-probe" and r["tokens"] >= 32
            ]
            if active:
                internal = state["internal_ids"]["kv-probe"]
                exported = rpc(
                    addresses[source],
                    "export",
                    dict(ticket=ticket, request_id=internal),
                )
                if "skipped" not in exported:
                    break
            if generation.done() or time.monotonic() > deadline:
                raise RuntimeError("Could not freeze an in-flight request")
            time.sleep(0.005)
        if retained:
            post(addresses[source], "areal_pause_generation_preserve", {})
            rpc(addresses[source], "discard", dict(ticket=ticket))
            exported = rpc(
                addresses[source],
                "export",
                dict(ticket=ticket, request_id=internal, retained_source=True),
            )
            if "skipped" in exported:
                raise RuntimeError(f"Retained export was skipped: {exported}")
        data = dict(
            ticket=ticket,
            bytes=exported["bytes"],
            geometry=geometry,
            pair_id=pair_key(source, target),
        )
        for i in (source, target):
            rpc(
                addresses[i],
                "prepare_transfer",
                dict(
                    data, source=i == source, retained_source=retained and i == source
                ),
            )
        jobs = [
            executor.submit(
                rpc,
                addresses[i],
                "transfer",
                dict(
                    data, source=i == source, retained_source=retained and i == source
                ),
            )
            for i in (source, target)
        ]
        for job in jobs:
            job.result()
        rpc(
            addresses[target],
            "register",
            dict(exported, ticket=ticket, external_id="kv-probe", version=0),
        )
        post(
            addresses[source],
            "areal_evacuate_retained" if retained else "areal_queue_rebalance",
            dict(internal_ids=[internal])
            if retained
            else dict(action="abort_running", internal_ids=[internal]),
        )
        prefix = generation.result()
    choice = prefix["choices"][0]
    continuation = post(
        addresses[target],
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
        abs(x["logprob"] - y["logprob"]) for x, y in zip(logs, b["logprobs"]["content"])
    )
    applied = rpc(addresses[target], "import_status", {})["results"][ticket]
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
    if not result["tokens_equal"] or error > 0.01 or len(logs) != 256:
        raise RuntimeError("KV continuation differs from deterministic baseline")
    rpc(addresses[source], "discard", dict(ticket=ticket))
    rpc(addresses[source], "unfreeze", dict(request_id=internal))
    if retained:
        post(addresses[source], "areal_evacuate_retained", {})
        post(addresses[source], "areal_continue_generation", {})
    return dict(
        result,
        source=source,
        target=target,
        pair_id=pair_key(source, target),
        retained=retained,
    )


def memory_roundtrip(address, prompt, model):
    payload = dict(
        request_id="graph-memory-baseline",
        token_ids=prompt,
        sampling_params=dict(
            temperature=0.0, max_tokens=128, ignore_eos=True, logprobs=0
        ),
        stream=False,
    )
    baseline = post(address, "inference/v1/generate", payload, True)["choices"][0]
    post(address, "areal_pause_generation", {})
    offloaded = post(address, "areal_colocation_memory", dict(action="offload"))
    if not offloaded or any(row["states"]["weights"] != "cpu" for row in offloaded):
        raise RuntimeError("Weight offload was not acknowledged")
    post(address, "areal_colocation_memory", dict(action="restore"))
    post(address, "areal_continue_generation", {})
    after = post(
        address,
        "inference/v1/generate",
        dict(payload, request_id="graph-memory-restored"),
        True,
    )["choices"][0]
    equal = baseline["token_ids"] == after["token_ids"]
    error = max(
        abs(a["logprob"] - b["logprob"])
        for a, b in zip(
            baseline["logprobs"]["content"], after["logprobs"]["content"], strict=True
        )
    )
    if not equal or error > 0.01:
        raise RuntimeError("Graph output changed after storage restoration/reload")
    post(address, "areal_pause_generation", {})
    post(address, "areal_update_weights", dict(model_path=model))
    post(address, "areal_continue_generation", {})
    reloaded = post(
        address,
        "inference/v1/generate",
        dict(payload, request_id="graph-reloaded"),
        True,
    )["choices"][0]
    reload_equal = baseline["token_ids"] == reloaded["token_ids"]
    reload_error = max(
        abs(a["logprob"] - b["logprob"])
        for a, b in zip(
            baseline["logprobs"]["content"],
            reloaded["logprobs"]["content"],
            strict=True,
        )
    )
    if not reload_equal or reload_error > 0.01:
        raise RuntimeError("Graph output changed after checkpoint reload")
    return dict(
        tokens_equal=equal,
        logprob_max_error=error,
        reload_tokens_equal=reload_equal,
        reload_logprob_max_error=reload_error,
        offloaded=offloaded,
    )


def preserved_roundtrip(address, prompt, output_dir):
    payload = dict(
        token_ids=(prompt * (9000 // len(prompt) + 1))[:9000],
        sampling_params=dict(
            temperature=0.0, max_tokens=512, ignore_eos=True, logprobs=0
        ),
        stream=False,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        baseline_jobs = [
            executor.submit(
                post,
                address,
                "inference/v1/generate",
                dict(payload, request_id=f"control-{i}"),
                True,
            )
            for i in range(1)
        ]
        baselines = [job.result()["choices"][0] for job in baseline_jobs]
    with ThreadPoolExecutor(max_workers=1) as executor:
        jobs = [
            executor.submit(
                post,
                address,
                "inference/v1/generate",
                dict(payload, request_id=f"preserve-{i}"),
                True,
            )
            for i in range(1)
        ]
        deadline = time.monotonic() + 90
        while True:
            state = post(address, "areal_queue_rebalance", dict(action="snapshot"))
            active = [
                x for x in state["active"] if x["request_id"].startswith("preserve-")
            ]
            if len(active) == 1 and min(x["tokens"] for x in active) >= 256:
                break
            if time.monotonic() > deadline or any(j.done() for j in jobs):
                raise RuntimeError("Unable to pause a full live-KV workload")
            time.sleep(0.02)
        post(address, "areal_pause_generation_preserve", {})
        before = post(address, "areal_colocation_memory", dict(action="status"))
        # Idle allocator pages may satisfy the first reclaim. Increase the
        # target one physical page group at a time until live KV is exercised.
        reclaimed = before
        try:
            for _ in range(before[0]["dynamic_kv"]["mapped_slabs"] + 2):
                row = reclaimed[0]
                pages = row["dynamic_kv"]
                if pages["swapped_slabs"]:
                    break
                target = (
                    row["free_bytes"]
                    + pages["physical_page_bytes"] * pages["tensor_count"]
                )
                reclaimed = post(
                    address,
                    "areal_colocation_memory",
                    dict(action="reclaim", target_free_bytes=target, member_rank=0),
                )
            if reclaimed[0]["dynamic_kv"]["swapped_slabs"] < 1:
                raise RuntimeError("No live KV page was actually offloaded")
        except BaseException:
            post(address, "areal_continue_generation", {})
            raise
        if any(row["dynamic_kv"]["swapped_slabs"] for row in reclaimed[1:]):
            raise RuntimeError("Unselected TP member unexpectedly offloaded KV")
        restored = post(address, "areal_colocation_memory", dict(action="restore"))
        if any(row["dynamic_kv"]["cpu_backup_bytes"] for row in restored):
            raise RuntimeError("Live KV backup remains after explicit restoration")
        post(address, "areal_continue_generation", {})
        outputs = [job.result()["choices"][0] for job in jobs]
    differences = []
    for i, (o, baseline) in enumerate(zip(outputs, baselines, strict=True)):
        mismatch = next(
            (
                j
                for j, (a, b) in enumerate(
                    zip(o["token_ids"], baseline["token_ids"], strict=True)
                )
                if a != b
            ),
            None,
        )
        error = max(
            abs(a["logprob"] - b["logprob"])
            for a, b in zip(
                o["logprobs"]["content"], baseline["logprobs"]["content"], strict=True
            )
        )
        differences.append(
            dict(request=i, first_mismatch=mismatch, logprob_max_error=error)
        )
    result = dict(
        tokens_equal=all(x["first_mismatch"] is None for x in differences),
        logprob_max_error=max(x["logprob_max_error"] for x in differences),
        requests=1,
        reclaimed=reclaimed,
        restored=restored,
        prompt_tokens=len(payload["token_ids"]),
        differences=differences,
    )
    (output_dir / "preserved-details.json").write_text(
        json.dumps(
            dict(
                result=result,
                before=before,
                baselines=baselines,
                outputs=outputs,
                active_at_pause=active,
            ),
            indent=2,
        )
    )
    if not result["tokens_equal"] or result["logprob_max_error"] > 0.01:
        raise RuntimeError(f"Live-KV graph continuation differs: {differences[:4]}")
    return result


def changed_weights_probe(address, prompt, original_model, replacement_model):
    """The replacement checkpoint must have a zeroed tied output embedding."""
    payload = dict(
        token_ids=prompt,
        sampling_params=dict(
            temperature=0.0, max_tokens=16, ignore_eos=True, logprobs=0
        ),
        stream=False,
    )
    before = post(
        address,
        "inference/v1/generate",
        dict(payload, request_id="weights-before"),
        True,
    )["choices"][0]
    post(address, "areal_pause_generation", {})
    post(address, "areal_update_weights", dict(model_path=replacement_model))
    post(address, "areal_continue_generation", {})
    changed = post(
        address,
        "inference/v1/generate",
        dict(payload, request_id="weights-changed"),
        True,
    )["choices"][0]
    post(address, "areal_pause_generation", {})
    post(address, "areal_update_weights", dict(model_path=original_model))
    post(address, "areal_continue_generation", {})
    restored = post(
        address,
        "inference/v1/generate",
        dict(payload, request_id="weights-restored"),
        True,
    )["choices"][0]
    if (
        changed["token_ids"] != [0] * 16
        or before["token_ids"] == changed["token_ids"]
        or restored["token_ids"] != before["token_ids"]
    ):
        raise RuntimeError("Graph did not observe changed/restored weights")
    return dict(changed_tokens=changed["token_ids"], restored_tokens_equal=True)


def prefill_checks(addresses, prompt):
    """Compare stable single-request outputs across capture and chunk boundaries."""
    rows = []
    for length in [31, 127, 129, 513, 2049, 4097, 9000]:
        payload = dict(
            token_ids=(prompt * (length // len(prompt) + 1))[:length],
            sampling_params=dict(
                temperature=0.0, max_tokens=32, ignore_eos=True, logprobs=0
            ),
            stream=False,
        )
        outputs = [
            post(
                a,
                "inference/v1/generate",
                dict(payload, request_id=f"prefill-{length}-{i}"),
                True,
            )["choices"][0]
            for i, a in enumerate(addresses)
        ]
        equal = all(o["token_ids"] == outputs[0]["token_ids"] for o in outputs)
        error = max(
            abs(a["logprob"] - b["logprob"])
            for o in outputs
            for a, b in zip(
                o["logprobs"]["content"], outputs[0]["logprobs"]["content"], strict=True
            )
        )
        rows.append(
            dict(
                prompt_tokens=length,
                tokens_equal=equal,
                logprob_max_error=error,
                outputs=outputs,
            )
        )
    return rows


def main():
    from areal.engine.vllm_ext.graph_compat import full_graph_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-roundtrip", action="store_true")
    parser.add_argument("--prefill-check", action="store_true")
    parser.add_argument(
        "--prefill-reference",
        type=Path,
        help="Native control prefill-details.json for adapter equivalence",
    )
    parser.add_argument(
        "--native-control",
        action="store_true",
        help="Disable sharing adapters; record the vendor graph/eager numeric baseline",
    )
    parser.add_argument("--control-inference-mode", choices=["graph", "eager"])
    parser.add_argument("--preserved-roundtrip", action="store_true")
    parser.add_argument("--retained-handoff", action="store_true")
    parser.add_argument(
        "--replacement-model", help="Test-only zero-output-embedding checkpoint"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inference-mode", choices=["graph", "eager"], default="graph")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
    if (
        args.tp < 1
        or args.replicas < 2
        or args.replicas % 2
        or len(devices) != args.replicas * args.tp
    ):
        raise ValueError("Probe needs an even number of equal-TP device groups")
    ports = find_free_ports(args.replicas)
    addresses = [f"127.0.0.1:{p}" for p in ports]
    processes = []
    try:
        for i, port in enumerate(ports):
            mode = (
                args.control_inference_mode
                if i == len(ports) - 1 and args.control_inference_mode
                else args.inference_mode
            )
            env = dict(
                os.environ,
                ASCEND_RT_VISIBLE_DEVICES=",".join(
                    devices[i * args.tp : (i + 1) * args.tp]
                ),
                AREAL_RUNNING_REBALANCE="kv",
                PYTORCH_NPU_ALLOC_CONF="expandable_segments:False",
                AREAL_AUTO_MEMORY="1",
                AREAL_DYNAMIC_KV="1",
                AREAL_KV_SLAB_BLOCKS="16",
                AREAL_DRAIN_METRICS="1",
                AREAL_QUEUE_REBALANCE="1",
            )
            if args.native_control:
                env.update(
                    AREAL_RUNNING_REBALANCE="off",
                    AREAL_AUTO_MEMORY="0",
                    AREAL_DYNAMIC_KV="0",
                    AREAL_KV_SLAB_BLOCKS="0",
                )
            if mode == "graph" and env.get("TASK_QUEUE_ENABLE") == "2":
                env["TASK_QUEUE_ENABLE"] = "1"
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
                "--dtype",
                "bfloat16",
                "--distributed-executor-backend",
                "mp" if args.tp > 1 else "uni",
                "--tensor-parallel-size",
                str(args.tp),
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
                "16384",
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
            cmd += (
                ["--enforce-eager"]
                if mode == "eager"
                else [
                    "--compilation-config",
                    json.dumps(full_graph_config(2048, 64)),
                ]
            )
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
                    if args.native_control:
                        with requests.Session() as session:
                            session.trust_env = False
                            session.get(
                                f"http://{address}/health", timeout=5
                            ).raise_for_status()
                    else:
                        rpc(address, "info", {})
                    break
                except requests.RequestException:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(1)
        half = args.replicas // 2
        peers = (
            {}
            if args.native_control
            else initialize_pairs(
                addresses, list(range(half)), list(range(half, args.replicas))
            )
        )
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
        results = [
            probe_pair(
                addresses,
                source,
                target,
                geometry,
                prompt,
                retained=args.retained_handoff,
            )
            for (source, target), geometry in peers.items()
        ]
        result = (
            results[0]
            if args.replicas == 2 and results
            else dict(tp=args.tp, replicas=args.replicas, pairs=results)
        )
        if args.prefill_check:
            checks = prefill_checks(addresses, prompt)
            (args.output / "prefill-details.json").write_text(
                json.dumps(checks, indent=2)
            )
            if args.prefill_reference:
                reference = json.loads(args.prefill_reference.read_text())
                if len(reference) != len(checks):
                    raise RuntimeError("Prefill reference has different cases")
                for observed, expected in zip(checks, reference, strict=True):
                    if observed["prompt_tokens"] != expected["prompt_tokens"]:
                        raise RuntimeError("Prefill reference lengths differ")
                    for actual, control in zip(
                        observed["outputs"], expected["outputs"], strict=True
                    ):
                        if actual["token_ids"] != control["token_ids"] or any(
                            abs(a["logprob"] - b["logprob"]) > 1e-5
                            for a, b in zip(
                                actual["logprobs"]["content"],
                                control["logprobs"]["content"],
                                strict=True,
                            )
                        ):
                            raise RuntimeError(
                                "Sharing adapter differs from native prefill reference"
                            )
                result["native_reference_equal"] = True
            elif not all(
                row["tokens_equal"]
                and (args.native_control or row["logprob_max_error"] <= 0.05)
                for row in checks
            ):
                raise RuntimeError(
                    "Prefill output comparison failed; inspect prefill-details.json and compare a native control"
                )
            result["prefill_checks"] = [
                {k: v for k, v in row.items() if k != "outputs"} for row in checks
            ]
        if args.memory_roundtrip:
            result["memory_roundtrip"] = [
                memory_roundtrip(a, prompt, args.model) for a in addresses
            ]
        if args.preserved_roundtrip:
            result["preserved_roundtrip"] = preserved_roundtrip(
                addresses[0], prompt, args.output
            )
        if args.replacement_model:
            result["changed_weights"] = changed_weights_probe(
                addresses[0], prompt, args.model, args.replacement_model
            )
        if args.prefill_check:
            evidence = []
            for i in range(args.replicas):
                mode = (
                    args.control_inference_mode
                    if i == args.replicas - 1 and args.control_inference_mode
                    else args.inference_mode
                )
                text = (args.output / f"server-{i}.log").read_text()
                expected = "FULL" if mode == "graph" else "NONE"
                phases = {
                    phase: any(
                        f"AREAL_GRAPH_{phase}" in line and f"mode={expected}" in line
                        for line in text.splitlines()
                    )
                    for phase in ("PREFILL_MIXED", "DECODE")
                }
                if not all(phases.values()):
                    raise RuntimeError(
                        f"Missing actual {expected} dispatch evidence: {i}: {phases}"
                    )
                evidence.append(dict(replica=i, mode=expected, phases=phases))
            result["execution_evidence"] = evidence
        result["inference_mode"] = args.inference_mode
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
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
