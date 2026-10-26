import json
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parent
subprocess.run(
    ["python3", str(R / "summarize.py")], check=True, stdout=subprocess.DEVNULL
)
g = json.loads((R / "summary.json").read_text())
e = json.loads((R.parent / "r026-qwen7b-tp-20261020" / "summary.json").read_text())
rows = []
for name, v in g["cases"].items():
    old = e["cases"][name]
    rows.append(
        dict(
            case=name,
            eager_tp1_s=old["tp1_median_s"],
            eager_tp2_s=old["tp2_median_s"],
            graph_tp1_s=v["tp1_median_s"],
            graph_tp2_s=v["tp2_median_s"],
            graph_tp2_savings_pct=v["time_reduction_pct"],
            graph_tp1_savings_vs_eager_pct=100
            * (1 - v["tp1_median_s"] / old["tp1_median_s"]),
            graph_tp2_savings_vs_eager_pct=100
            * (1 - v["tp2_median_s"] / old["tp2_median_s"]),
        )
    )
(R / "comparison.json").write_text(json.dumps(rows, indent=2))
for x in rows:
    print(x)
