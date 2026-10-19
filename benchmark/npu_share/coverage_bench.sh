#!/usr/bin/env bash
# Run selected representative comparisons sequentially.
set -euo pipefail

mode=prepare
suite=core
selected=""
profile="${HOME}/.config/areal-benchmark/profile.json"
runner="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_bench.py"
usage() {
    cat <<'HELP'
Usage: coverage_bench.sh [--suite NAME] [--mode plan|prepare|run]
                         [--cases 'C01 C02 ...'] [--profile FILE]
Suites: core (default), load, boundary, topology, long, all
Default mode: prepare (no NPU training or sandbox creation).
--cases selects explicit IDs instead of a suite. Cases run sequentially, stopping
on the first failure. Each run_bench invocation compares fixed and shared modes.
HELP
}
while (($#)); do
    case "$1" in
        --suite|--mode|--cases|--profile)
            (($# >= 2)) || { usage >&2; exit 2; }
            case "$1" in
                --suite) suite="$2";; --mode) mode="$2";;
                --cases) selected="$2";; --profile) profile="$2";;
            esac
            shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
    esac
done
case "$mode" in plan|prepare|run) ;; *) echo "Invalid mode: $mode" >&2; exit 2;; esac
case "$suite" in
    core) defaults="C01 C02 C03 C04";;
    load) defaults="C05 C06";;
    boundary) defaults="E00 E01 E02 E03 E04";;
    topology) defaults="T01 T02 T03 T04";;
    long) defaults="L01";;
    all) defaults="C01 C02 C03 C04 C05 C06 E00 E01 E02 E03 E04 T01 T02 T03 T04 L01";;
    *) echo "Invalid suite: $suite" >&2; exit 2;;
esac
selected="${selected:-$defaults}"
declare -A matrix=(
    [C01]="math fsdp     16 6   2 1 2"
    [C02]="math megatron 16 6   2 1 2"
    [C03]="code fsdp     16 3   2 1 2"
    [C04]="code megatron 16 3   2 1 2"
    [C05]="math fsdp     32 6   2 1 2"
    [C06]="math megatron 32 6   2 1 2"
    [E00]="math fsdp     16 6   2 1 0"
    [E04]="math megatron 16 3   2 1 0"
    [E01]="math fsdp      1 1   2 1 1"
    [E02]="math fsdp     16 6   2 1 1"
    [E03]="math fsdp     16 6   2 1 3"
    [T01]="math fsdp     16 3   4 1 2"
    [T02]="math fsdp     16 3   4 2 2"
    [T03]="math megatron 32 3   8 1 2"
    [T04]="math megatron 32 3   8 2 2"
    [L01]="math fsdp     16 100 2 1 2"
)
read -r -a ids <<< "$selected"
declare -A seen=()
for id in "${ids[@]}"; do
    [[ "$id" =~ ^[CETL][0-9]{2}$ ]] || { echo "Invalid case ID: $id" >&2; exit 2; }
    [[ -v "matrix[$id]" && ! -v "seen[$id]" ]] || {
        echo "Unknown or duplicate case ID: $id" >&2; exit 2;
    }
    seen[$id]=1
done
output="$(python3 - "$profile" <<'PY'
import datetime
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
p = json.loads(Path(sys.argv[1]).read_text())
stamp = datetime.datetime.now(ZoneInfo(p.get("timezone", "UTC"))).strftime("%Y%m%d-%H%M%S%z")
print(Path(p["workspace_root"]).resolve() / ("coverage-" + stamp))
PY
)"
if [[ "$mode" != plan ]]; then
    mkdir -p -- "$output"
    printf 'case\trl\tengine\tconcurrency\tbatches\tnpu_count\ttp\toffpolicy\texit_code\n' > "$output/cases.tsv"
fi
echo "Coverage mode=$mode; cases=$selected; output=$output"
for id in "${ids[@]}"; do
    read -r rl engine concurrency batches npus tp offpolicy <<< "${matrix[$id]}"
    cmd=(python3 "$runner" --profile "$profile" --workspace-root "$output"
         --rl "$rl" --engine "$engine" --concurrency "$concurrency"
         --batches "$batches" --npu-count "$npus" --inference-tp "$tp"
         --max-head-offpolicyness "$offpolicy")
    [[ "$mode" != prepare ]] || cmd+=(--prepare-only)
    echo "[$id] $rl / $engine; c=$concurrency b=$batches NPU=$npus TP=$tp offpolicy=$offpolicy"
    printf '%q ' "${cmd[@]}"; printf '\n'
    [[ "$mode" != plan ]] || continue
    set +e
    "${cmd[@]}" 2>&1 | tee "$output/$id.log"
    statuses=("${PIPESTATUS[@]}")
    set -e
    rc="${statuses[0]}"
    [[ "${statuses[1]}" == 0 ]] || rc=1
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$id" "$rl" "$engine" "$concurrency" "$batches" "$npus" "$tp" "$offpolicy" "$rc" >> "$output/cases.tsv"
    if [[ "$rc" != 0 ]]; then
        echo "[$id] Failed; stopping suite. See $output/$id.log" >&2
        exit "$rc"
    fi
done
echo "Coverage finished: $output"
