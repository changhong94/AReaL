# R030: same-state asynchronous HF save replay (2026-10-26)

The original 11.549 s foreground call contains a 4.308 s rank-0 engine RPC, including
4.272 s staging. Only 0.036 s is inside that RPC but outside staging. Approximately
7.241 s is outside the engine RPC. Existing logs cannot retrospectively attribute every
millisecond of that outer interval. It is not a seven-second DRAM or optimizer fence.
This corrects the earlier unverified fence hypothesis.

The replay confirms expensive tokenizer metadata handling: per-rank archive/
serialization, tokenizer reconstruction, and writer-side deepcopy. The change captures
save_pretrained files once per save, passes immutable base64 strings through RPC, and
writes them in the background. No tokenizer cache is used: caller mutations are captured
on every save. FSDP async dispatch also avoids importing Archon first. Independent CPU
weight ownership and all state/version fences remain unchanged. Unsafe/reserved
auxiliary paths fail during capture, before weight staging.

## Matched comparison

Same final model state; two CPU/Gloo ranks; production codecs and FSDP save method. Pipe
transport replaces HTTP. No active training, NPU gather, or standby contention is
reproduced. This is a saving-path replay, not a new end-to-end RL measurement. One CPU
thread per worker, unchanged across arms. Order: new, old, old, new.

| Metric (two-run mean)                  |     Old |     New |
| -------------------------------------- | ------: | ------: |
| Foreground, including metadata capture | 6.044 s | 2.298 s |
| Final write join                       | 2.306 s | 2.157 s |
| Total, including file completion       | 8.350 s | 4.455 s |

Foreground reduction: 62.0%; total reduction: 46.7%. Final write waits are included. No
fsync/power-loss durability claim is made.

All four output checkpoints contain 339 tensors exactly equal to the input, and matching
tokenizer vocabulary, special tokens, chat template and sample encodings. 58 focused
tests pass. Source-side optimizer state is not saved by this HF-only path. Timing now
records dispatch, codec, worker queue, individual snapshot fences, DRAM mapping,
independent CPU copy, metadata copy and writer submission. The remaining mandatory CPU
snapshot copy is retained; removing it without pinning an immutable generation would
introduce a race.

Evidence: summary.json, timing-details.json, matched-results.json, correctness.json. No
NPU was used; make-npu-busy was left running. No commit/push.
