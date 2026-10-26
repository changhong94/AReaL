# R030: latest fixed/shared full RL comparison (2026-10-26)

Same code-RL workload as the old/new saving test: FSDP, two NPUs, TP1 graph, 192
questions, batch32, eight answers/question, concurrency16, scoring16, six batches,
offpolicy1, seed1. Development vLLM reuse with verified initial weight reset. Fixed
retains its original synchronous saving and answer-level scoring; shared uses the latest
policies and optimized async HF saving.

| Metric                                     |   Fixed |  Shared |
| ------------------------------------------ | ------: | ------: |
| RL loop seconds (final file wait included) | 461.807 | 429.981 |
| Generated tokens                           |  361789 |  371932 |
| Tokens / RL loop second                    | 783.421 | 864.996 |
| Save foreground seconds                    |  10.869 |   4.062 |
| Final save join seconds                    |   0.000 |   3.033 |
| Training-engine initialization seconds     | 179.687 | 411.651 |
| Initialization + RL seconds                | 641.494 | 841.632 |

RL loop saving: 31.826 s (6.89%). Token throughput gain: 10.41%. Shared generates more
tokens, but content differs and this is one pair; do not attribute the complete shared
gain to the final saving change. The short run is slower when training-engine
initialization is included. These totals still exclude the earlier reusable inference
service preparation, so are not full job wall times.

Both arms completed all 1536 answers and scores with no code_reward_error events. The
three-batch full correctness gate passed before benchmarking. All six timed checkpoints
across both experiments have matching tokenizer vocabulary, special tokens, chat
template and sample encodings. Performance full readback was disabled as before; the
final write completion is inside train.end.

Four figure families (11 PNG files) are under figures/. All timings, scoring work, stage
arrays and input hashes are in measurements.json. make-npu-busy restored.
