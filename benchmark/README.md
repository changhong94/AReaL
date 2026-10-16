# Benchmark

This directory contains guides for reproducing AReaL's throughput and algorithmic
performance benchmarks, as reported in the paper.

As of January 2026, AReaL has been fully refactored from the original `realhf` directory
to the new `areal` directory. The legacy configurations and instructions are no longer
compatible with the current codebase. For reference, you can still access the legacy
configs and run experiments within `realhf` by reverting to commit `820ca49` or earlier.

Updated benchmark results for the new codebase will be available soon.

## Ascend fixed/shared comparison

See [the multi-NPU comparison runner](npu_share/README.md) for arithmetic or code RL
benchmarks with FSDP or Megatron, timestamped workspaces outside the repository, and
four families of performance charts. Host profiles and credentials remain outside the
repository.
