import hashlib
import json
import os
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

R = Path("/experiment")
tp = int(os.environ["BENCH_TP"])
label = os.environ["BENCH_SESSION"]
tok = AutoTokenizer.from_pretrained("/model")
old = json.loads((R / "workload-text.json").read_text())["requests"]
tails = []
for row in old:
    prompt = tok.encode(row["prompt"], add_special_tokens=False)
    completion = tok.encode(row["completion"], add_special_tokens=False)
    assert len(prompt) == row["prompt_len"] and len(completion) == row["output_tokens"]
    tails.append(
        dict(
            ids=prompt + completion[: row["prefix_output_tokens"]],
            tokens=row["remaining_tokens"],
        )
    )
longest = max(old, key=lambda row: row["historical_finish_s"])
single = tails[old.index(longest)]
# Fixed lengths prevent output variation from masquerading as a speedup.
base = single["ids"][-1536:]
cases = {"historical_tail_group": tails, "historical_last_single": [single]}
for count in [1, 4, 16, 32]:
    cases["fixed512_c" + str(count)] = [
        dict(ids=base, tokens=512) for _ in range(count)
    ]
manifest = {
    k: [
        dict(
            input_tokens=len(v["ids"]),
            output_tokens=v["tokens"],
            input_sha256=hashlib.sha256(json.dumps(v["ids"]).encode()).hexdigest(),
        )
        for v in vs
    ]
    for k, vs in cases.items()
}
(R / (label + "-workload.json")).write_text(json.dumps(manifest, indent=2))
start = time.perf_counter()
llm = LLM(
    model="/model",
    tensor_parallel_size=tp,
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=64,
    gpu_memory_utilization=0.6,
    seed=1,
    enforce_eager=False,
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    enable_prefix_caching=True,
    disable_log_stats=True,
)
init = time.perf_counter() - start
result = dict(tp=tp, initialization_s=init, graph_mode="FULL_DECODE_ONLY", cases=[])
for name, items in cases.items():
    prompts = [{"prompt_token_ids": x["ids"]} for x in items]
    params = [
        SamplingParams(
            temperature=0,
            max_tokens=x["tokens"],
            min_tokens=x["tokens"],
            ignore_eos=True,
            seed=1,
        )
        for x in items
    ]
    for rep in range(-2, 3):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        rows = []
        for expected, o in zip(items, outputs):
            ids = o.outputs[0].token_ids
            assert len(ids) == expected["tokens"]
            m = vars(o.metrics) if o.metrics else {}
            first = m.get("first_token_time")
            last = m.get("last_token_time")
            arrival = m.get("arrival_time")
            rows.append(
                dict(
                    input_tokens=len(expected["ids"]),
                    output_tokens=len(ids),
                    cached_tokens=getattr(o, "num_cached_tokens", None),
                    metrics=m,
                    ttft_s=first - arrival
                    if first is not None and arrival is not None
                    else None,
                    tpot_s=(last - first) / (len(ids) - 1)
                    if last is not None and first is not None and len(ids) > 1
                    else None,
                    output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                )
            )
        row = dict(
            case=name,
            repeat=rep,
            warmup=rep < 0,
            elapsed_s=elapsed,
            output_tokens=sum(x["tokens"] for x in items),
            requests=rows,
        )
        result["cases"].append(row)
        (R / (label + "-results.json")).write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in row.items() if k != "requests"}), flush=True)
