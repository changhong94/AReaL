# R026 model screening: Qwen2-7B-Instruct (2026-10-20)

[中文](2026-10-20-r026-qwen2-7b-tp-screen.zh.md)

**Outcome: TP2 did not accelerate the tested long tails.** This is inference-only
screening, not a cross-TP KV migration implementation or an RL performance result.

## Model and method

The original candidate Qwen2.5-7B-Instruct had no local weights and external download
was slow. We instead copied Qwen2-7B-Instruct from the host model store. All four weight
shards match the official Hugging Face LFS SHA-256 hashes. Metadata and source identity
are recorded in the evidence manifest. The default RL model is unchanged.

Ascend 910B2; TP1 on NPU2, TP2 on NPUs1/2. BF16, eager, prefix caching enabled, max
model length4096, max sequences64, memory utilization0.6. Order: TP1a, TP2, TP1b, with
independent engine startup per arm, two warmups and three measured repetitions per
condition. TP1 values combine six measurements; TP2 uses three. Initialization and
warmups are excluded. Identical token inputs and output counts are asserted.

Historical prompts include previously generated prefixes, reconstructed as token inputs;
this is cached-prefix replay, not live transferred KV. Fixed512 cases use one identical
prefix repeated at concurrency1/4/16/32 with exactly512 generated tokens. These
workloads test service cost, not model answer quality.

| Workload               | TP1 median(s) | TP2 median(s) | TP2 slowdown | Reserved NPU-seconds ratio |
| ---------------------- | ------------: | ------------: | -----------: | -------------------------: |
| historical_tail_group  |        16.668 |        19.537 |       17.21% |                      2.34x |
| historical_last_single |        15.466 |        18.130 |       17.23% |                      2.34x |
| fixed512_c1            |        28.615 |        33.740 |       17.91% |                      2.36x |
| fixed512_c4            |        31.152 |        35.253 |       13.17% |                      2.26x |
| fixed512_c16           |        31.295 |        36.095 |       15.34% |                      2.31x |
| fixed512_c32           |        31.994 |        36.574 |       14.32% |                      2.29x |

## Interpretation and limits

For these workloads, the estimated migration gain is already negative before any KV
transfer/resharding/resumption cost. Qwen2-7B-Instruct in this eager stack is therefore
not a validated positive R026 candidate. This does not reject larger TP for other
models, kernels, graph execution modes or hardware. No profiler attribution was
performed: communication/launch/smaller-kernel overhead remains a hypothesis.

Offline RequestOutput.metrics was absent with log statistics disabled, so TTFT and pure
decode TPOT are not reported as measured. Completion time includes residual prefix
processing. Reserved NPU-seconds counts two allocated cards for TP2; it is not measured
energy. Different generated token content is possible despite matched lengths. This is a
single TP1–TP2–TP1 screening sequence, not broad statistical proof.

## Evidence

[Summary](evidence/r026-20261020/summary.json),
[plan](evidence/r026-20261020/plan.json),
[model identity](evidence/r026-20261020/model-manifest.json),
[integrity manifest](evidence/r026-20261020/manifest.json). Raw logs and scripts:
`/root/areal-development/r026-qwen7b-tp-20261020/`. Prepared model:
`/root/areal-assets/Qwen2-7B-Instruct/`. No live KV migration was added. A positive
decode-only prerequisite is needed before investing in cross-TP migration correctness
and cost-aware integration.
