#!/usr/bin/env python3
"""Development-only warm-session comparison; not valid as a formal benchmark."""

import argparse
import json
import sys

import run_bench
from session_bench import run


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--repetitions", type=run_bench.positive, default=2)
    parser.add_argument(
        "--kv-policy",
        choices=["conservative", "aggressive", "exploratory"],
        default="exploratory",
    )
    special, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    opts = run_bench.arguments()
    opts.repetitions = special.repetitions
    opts.kv_policy = special.kv_policy
    if opts.render_only:
        raise ValueError("Use the original benchmark renderer for existing workspaces")
    if opts.npu_count != 2 or opts.inference_tp != 1:
        raise ValueError("Development reuse currently supports two NPUs and TP=1")
    print(
        "DEVELOPMENT ONLY: warm-session results are not formal benchmark evidence",
        flush=True,
    )
    run(opts, json.loads(opts.profile.read_text()))


if __name__ == "__main__":
    main()
