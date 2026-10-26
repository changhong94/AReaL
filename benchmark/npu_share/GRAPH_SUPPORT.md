# Ascend inference graph support

## Execution contract

`run_bench.py --inference-mode graph` selects `FULL` for both fixed and shared arms.
This is the default. Prefill, mixed prefill/decode, and decode batches are eligible for
ACLGraph replay. Training execution is unchanged.

`--inference-mode eager` disables inference graph capture. The coverage matrix uses
graph for 21 cases and eager for G01/G02. Low-level explicit `FULL_DECODE_ONLY` remains
accepted for decode-only comparisons.

Capture sizes cover the scheduler token limit, with dense small decode buckets and
sparse larger prefill buckets. With the default token limit, the largest capture is 2048
tokens. Requests longer than this are handled by chunked prefill; this is not a
2048-token context limit.

The worker logs the actual dispatch mode once for each phase/mode combination:

- `AREAL_GRAPH_PREFILL_MIXED ... mode=FULL`
- `AREAL_GRAPH_DECODE ... mode=FULL`
- `mode=NONE` reports an eager fallback.

These markers show dispatch selection; successful request completion is also required to
establish successful graph execution.

## Memory and publication safety

- Weight, buffer, and sparse KV virtual addresses remain stable across replay.
- Physical pages can be reclaimed only through the existing pause/fence protocol.
- Graph workspaces stay resident in the vendor allocator. Their allocation is reflected
  in measured free HBM; they are not discarded as idle training cache.
- Capture requires an idle, restored KV pool. Dummy block tables reference the permanent
  null block and slot mappings are invalidated before capture.
- Weight updates validate storage before and after publication. A storage change
  invalidates the graph and fails closed. TP members agree on errors before entering
  weight-update collectives.
- Inference children normalize `TASK_QUEUE_ENABLE=2` to `1`, because the tested
  torch_npu stack rejects Level 2 during graph capture. Training environment values are
  preserved.

Scope remains the existing Qwen2 BF16 full-attention uni/mp implementation, with
equal-TP migration peers. This does not add support for LoRA, MoE, context parallelism,
native vLLM KV transfer, or arbitrary attention backends.

## Validation, 2026-10-20

Hardware: Ascend 910B2; model: Qwen2.5-1.5B-Instruct; vLLM 0.26 Ascend runtime.
Development evidence root: `/root/areal-development/graph-compat-20261020`.

- Decode-only FSDP two-device and Megatron eight-device TP=2 fixed/shared comparisons
  completed three batches per arm.
- FULL prefill TP=1 and TP=2 probes passed real request-local HCCL KV migration, weight
  RAM round-trip, checkpoint reload, and live KV partial reclaim/restore.
- Prompt lengths 31, 127, 129, 513, 2049, 4097, and 9000 exercised capture boundaries
  and chunked prefill. Matched graph replicas produced identical tokens/logprobs.
- The retained-KV test uses one 9000-token prompt and 512 generated tokens. Only rank 0
  reclaims a physical KV page group; other TP members remain mapped. Resumed outputs and
  logprobs matched the uninterrupted control exactly.
- A test-only checkpoint with zeroed tied output embeddings changed graph outputs to
  token 0; restoring the original checkpoint restored the original outputs.
- Graph versus eager generated identical tokens in the seven prompt tests, but their
  logprobs differed by up to 0.09116137. The same difference occurred with sharing
  adapters disabled. Within each execution mode, adapter/native outputs and logprobs
  matched exactly. Do not require bitwise graph/eager equivalence. RL importance ratios
  use these probabilities; this check establishes adapter non-regression, not equivalent
  learning dynamics between graph and eager.
- FULL prefill FSDP two-device fixed/shared integration passed three batches per arm,
  with actual FULL prefill/decode dispatch and no observed eager fallback.
- FULL prefill Megatron eight-device TP=2 fixed/shared integration passed three batches
  per arm, including multiple inference instances and shared publication. Actual FULL
  prefill/decode dispatch was observed without eager fallback.

Machine-readable evidence:
[graph-validation-2026-10-20.json](graph-validation-2026-10-20.json).

The earlier 32-request retained-KV comparison was not a valid isolated control: outputs
diverged before the pause because the submitted requests formed different execution
batches. The single-request long-context test removes that confound.

## Reproducing the focused probes

Run inside the configured Ascend runtime with model and writable output mounts:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python3 benchmark/npu_share/kv_handoff_probe.py \
  --model /model --output /work/full-tp2 --inference-mode graph --tp 2 \
  --prefill-check --memory-roundtrip --preserved-roundtrip
```

A native graph/eager reference can be recorded with
`--native-control --control-inference-mode eager --prefill-check`. An adapter probe can
then pass `--prefill-reference /work/native/prefill-details.json` to require equivalence
within each mode rather than treating intrinsic graph/eager numeric differences as an
adapter regression.

These are correctness/compatibility checks, not evidence of a performance gain.
