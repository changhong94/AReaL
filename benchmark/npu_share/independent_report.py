"""Report independent arms using external wall-clock events, never invented traces."""

import html
import json
from pathlib import Path


def render(root):
    root = Path(root)
    settings = json.loads((root / "settings.json").read_text())
    status = json.loads((root / "arm-status.json").read_text())
    times = {}
    for arm in settings["arms"]:
        events = [
            json.loads(line)
            for line in (root / arm / "phases.jsonl").read_text().splitlines()
            if line.strip()
        ]
        starts = [x["time"] for x in events if x["event"] == "train.begin"]
        ends = [x["time"] for x in events if x["event"] == "train.end"]
        if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
            if arm == "colocated":
                status[arm] = {
                    "status": "analysis_failed",
                    "reason": "Missing or invalid train timing pair",
                }
                continue
            raise ValueError(f"Missing completed training interval: {arm}")
        times[arm] = ends[0] - starts[0]

    def gain(a, b):
        return (
            100 * (times[a] - times[b]) / times[a]
            if a in times and b in times
            else None
        )

    summary = dict(
        settings=settings,
        arm_status=status,
        loop_seconds=times,
        reduction_percent=gain("fixed", "latest"),
        colocated_reduction_percent=gain("fixed", "colocated"),
        shared_vs_colocated_percent=gain("colocated", "latest"),
        source_provenance=settings["source_provenance"],
        figures=[],
        trace_status="Community detailed traces require a separate compatible reader; no synthetic traces.",
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# 独立源码运行结果",
        "",
        "| 组别 | RL 循环秒数 | 源码 commit | 状态 |",
        "| --- | ---: | --- | --- |",
    ]
    labels = {"fixed": "固定", "colocated": "colocated", "latest": "共享"}
    for arm in settings.get("requested_arms", settings["arms"]):
        duration = f"{times[arm]:.3f}" if arm in times else "—"
        commit = settings["source_provenance"][arm]["head"]
        lines.append(
            f"| {labels[arm]} | {duration} | {commit} | {status.get(arm, {}).get('status', 'unknown')} |"
        )
    lines += [
        "",
        "耗时来自各组入口的 train.begin/end。当前不合成社区基线缺失的微阶段、请求或 token 图；运行完成不等于完整数值正确性验证。",
    ]
    for key, label in [
        ("reduction_percent", "共享相对固定缩短"),
        ("colocated_reduction_percent", "colocated 相对固定缩短"),
        ("shared_vs_colocated_percent", "共享相对 colocated 缩短"),
    ]:
        if summary[key] is not None:
            lines.append(f"{label}：{summary[key]:.2f}%")
    text = "\n".join(lines) + "\n"
    (root / "summary.zh.md").write_text(text)
    (root / "index.html").write_text(
        '<meta charset="utf-8"><pre>' + html.escape(text) + "</pre>"
    )
