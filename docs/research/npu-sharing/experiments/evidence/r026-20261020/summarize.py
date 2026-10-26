import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parent
all_data = {
    label: json.loads((R / (label + "-results.json")).read_text())
    for label in ["tp1a", "tp2", "tp1b"]
}
workloads = [
    json.loads((R / (label + "-workload.json")).read_text()) for label in all_data
]
assert all(w == workloads[0] for w in workloads)
result = dict(
    metrics_note=json.loads((R / "plan.json").read_text())["request_metrics_note"],
    model=json.loads((R / "model-manifest.json").read_text())["model"],
    initialization_s={k: v["initialization_s"] for k, v in all_data.items()},
    cases={},
)
for name in workloads[0]:
    arms = {}
    for label, data in all_data.items():
        rows = [r for r in data["cases"] if r["case"] == name and not r["warmup"]]
        arms[label] = dict(
            times_s=[r["elapsed_s"] for r in rows],
            output_tokens=rows[0]["output_tokens"],
            requests=len(rows[0]["requests"]),
            ttft_s=[
                x["ttft_s"]
                for r in rows
                for x in r["requests"]
                if x["ttft_s"] is not None
            ],
            tpot_s=[
                x["tpot_s"]
                for r in rows
                for x in r["requests"]
                if x["tpot_s"] is not None
            ],
        )
    a = statistics.median(arms["tp1a"]["times_s"] + arms["tp1b"]["times_s"])
    b = statistics.median(arms["tp2"]["times_s"])
    result["cases"][name] = dict(
        arms=arms,
        tp1_median_s=a,
        tp2_median_s=b,
        saved_s=a - b,
        speedup=a / b,
        time_reduction_pct=100 * (1 - b / a),
        reserved_npu_seconds_ratio=2 * b / a,
    )
(R / "summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
