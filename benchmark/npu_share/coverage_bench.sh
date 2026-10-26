#!/usr/bin/env bash
# Run selected representative comparisons sequentially.
set -euo pipefail

mode=prepare
suite=core
inference_mode=""
selected=""
profile="${HOME}/.config/areal-benchmark/profile.json"
reporter="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/coverage_summary.py"
runner="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_bench.py"
usage() {
    cat <<'HELP'
Usage: coverage_bench.sh [--suite NAME] [--mode plan|prepare|run]
                         [--cases 'C01 C02 ...'] [--profile FILE]
                         [--inference-mode graph|eager]
Suites: core (default), load, boundary, topology, ratio, execution, workload, long, all
Inference defaults to graph; execution suite also includes eager controls.
--inference-mode overrides the mode for every selected case.
Default mode: prepare (no NPU training or sandbox creation).
--cases selects explicit IDs instead of a suite. Cases run sequentially and continue after individual failures.
The suite exits nonzero if any case fails. Each run_bench invocation compares fixed and shared modes.
HELP
}
while (($#)); do
    case "$1" in
        --suite|--mode|--cases|--profile|--inference-mode)
            (($# >= 2)) || { usage >&2; exit 2; }
            case "$1" in
                --suite) suite="$2";; --mode) mode="$2";;
                --cases) selected="$2";; --profile) profile="$2";;
                --inference-mode) inference_mode="$2";;
            esac
            shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
    esac
done
case "$mode" in plan|prepare|run) ;; *) echo "Invalid mode: $mode" >&2; exit 2;; esac
case "$inference_mode" in ""|graph|eager) ;; *) echo "Invalid inference mode" >&2; exit 2;; esac
case "$suite" in
    core) defaults="C01 C02 C03 C04";;
    load) defaults="C05 C06";;
    boundary) defaults="E00 E01 E02 E03 E04";;
    topology) defaults="T01 T02 T03 T04";;
    ratio) defaults="R01 R02 R03 R04";;
    execution) defaults="C01 C02 G01 G02";;
    workload) defaults="C01 W01 W02 W03 W04 W05";;
    long) defaults="L01 L02";;
    all) defaults="C01 C02 C03 C04 C05 C06 E00 E01 E02 E03 E04 T01 T02 T03 T04 R01 R02 R03 R04 G01 G02 W01 W02 W03 W04 W05 L01 L02";;
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
    [R01]="math fsdp     16 3   3 1 2 2:1"
    [R02]="math fsdp     16 3   4 1 2 3:1"
    [R03]="math megatron 16 3   6 2 2 2:1"
    [R04]="math megatron 16 3   8 2 2 3:1"
    [G01]="math fsdp     16 6   2 1 2 1:1 eager"
    [G02]="math megatron 16 6   2 1 2 1:1 eager"
    [W01]="math fsdp     16 6   2 1 2 1:1 graph 16 8"
    [W02]="math fsdp     16 6   2 1 2 1:1 graph 64 8"
    [W03]="math fsdp     16 6   2 1 2 1:1 graph 32 4"
    [W04]="math fsdp     16 6   2 1 2 1:1 graph 32 16"
    [W05]="math fsdp     16 6   2 1 2 1:1 graph 64 16"
    [L01]="math fsdp     16 100 2 1 2"
    [L02]="math megatron 32 100 2 1 2 1:1"
)
read -r -a ids <<< "$selected"
declare -A seen=()
for id in "${ids[@]}"; do
    [[ "$id" =~ ^[CETLRGW][0-9]{2}$ ]] || { echo "Invalid case ID: $id" >&2; exit 2; }
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
    printf 'case\trl\tengine\tconcurrency\tbatches\tnpu_count\ttp\toffpolicy\tinference_training_ratio\tadaptive_concurrency\tinference_mode\tbatch_size\tanswers\texit_code\n' > "$output/cases.tsv"
fi
case_parameters() {
    read -r rl engine concurrency batches npus tp offpolicy ratio case_mode batch_size answers <<< "${matrix[$id]}"
    batch_size="${batch_size:-32}"
    answers="${answers:-8}"
    ratio="${ratio:-1:1}"
    case_mode="${inference_mode:-${case_mode:-graph}}"
}
if [[ "$mode" != plan ]]; then
    head -1 "$output/cases.tsv" > "$output/planned-cases.tsv"
    for id in "${ids[@]}"; do
        case_parameters
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n' \
            "$id" "$rl" "$engine" "$concurrency" "$batches" "$npus" "$tp" "$offpolicy" "$ratio" "capacity" "$case_mode" "$batch_size" "$answers" >> "$output/planned-cases.tsv"
    done
    report_on_exit() {
        local rc=$?
        trap - EXIT
        python3 "$reporter" "$output" --final || { echo "Coverage summary generation failed" >&2; rc=1; }
        echo "Coverage summary: $output/index.html (English), $output/index.zh.html (Chinese)"
        exit "$rc"
    }
    trap report_on_exit EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    python3 "$reporter" "$output"
fi
failed=0
echo "Coverage mode=$mode; cases=$selected; output=$output"
for id in "${ids[@]}"; do
    case_parameters
    cmd=(python3 "$runner" --profile "$profile" --workspace-root "$output"
         --rl "$rl" --engine "$engine" --concurrency "$concurrency"
         --batches "$batches" --npu-count "$npus" --inference-tp "$tp"
         --max-head-offpolicyness "$offpolicy"
         --inference-training-ratio "$ratio" --adaptive-concurrency capacity
         --inference-mode "$case_mode" --batch-size "$batch_size" --answers "$answers")
    [[ "$mode" != prepare ]] || cmd+=(--prepare-only)
    echo "[$id] $rl / $engine; c=$concurrency b=$batches NPU=$npus TP=$tp offpolicy=$offpolicy inference:training=$ratio adaptive=capacity inference=$case_mode batch-size=$batch_size answers=$answers"
    printf '%q ' "${cmd[@]}"; printf '\n'
    [[ "$mode" != plan ]] || continue
    : > "$output/$id.log"
    python3 "$reporter" "$output"
    set +e
    "${cmd[@]}" 2>&1 | tee "$output/$id.log"
    statuses=("${PIPESTATUS[@]}")
    set -e
    rc="${statuses[0]}"
    [[ "${statuses[1]}" == 0 ]] || rc=1
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$id" "$rl" "$engine" "$concurrency" "$batches" "$npus" "$tp" "$offpolicy" "$ratio" "capacity" "$case_mode" "$batch_size" "$answers" "$rc" >> "$output/cases.tsv"
    python3 "$reporter" "$output"
    if [[ "$rc" != 0 ]]; then
        failed=$((failed + 1))
        echo "[$id] Failed; continuing to the next case. See $output/$id.log" >&2
    fi
done
echo "Coverage finished: $output; failed cases=$failed"
[[ "$failed" == 0 ]] || exit 1
