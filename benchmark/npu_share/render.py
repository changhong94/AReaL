"""Render four chart families from retained benchmark evidence; no NPU access."""

import argparse
import datetime as dt
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text


def rows(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load(root, arm, settings=None):
    settings = settings or {}
    replicas = settings.get("npu_count", 2) // settings.get("inference_tp", 1)
    offset = replicas // 2 if arm == "fixed" else 0

    def instance(rank):
        return offset + rank + 1

    p = root / arm
    phases = rows(p / "phases.jsonl")
    base = next(e["time"] for e in phases if e["event"] == "train.begin")
    end = next(e["time"] for e in phases if e["event"] == "train.end") - base
    events = sorted(
        (e for f in (p / "request-events").glob("*.jsonl") for e in rows(f)),
        key=lambda e: e["time"],
    )
    batches = {
        task: e["batch"]
        for e in events
        if e["event"] == "batch_consumed"
        for task in e["task_ids"]
    }
    ready = {
        e["batch"]: e["time"] - base
        for e in events
        if e["event"] == "train_batch_answers_ready"
    }
    starts = {e["request_id"]: e for e in events if e["event"] == "answer_start"}
    ends = {e["request_id"]: e for e in events if e["event"] == "answer_end"}
    if starts.keys() != ends.keys():
        raise ValueError(
            f"{arm}: incomplete request evidence; retain logs and inspect before rendering a completed comparison"
        )
    native = {e["internal_id"]: e for e in events if e["event"] == "native_id"}
    byrid = defaultdict(list)
    for e in events:
        if "request_id" in e:
            byrid[e["request_id"].removeprefix("generate-tokens-")].append(e)
        for rid in e.get("request_ids", []):
            byrid[rid].append(e)
    # Fixed engines use their rollout worker rank; routed shared requests carry
    # the selected server rank explicitly. Recover old arrivals from send records.
    sends_by_request = defaultdict(list)
    for event in events:
        if event["event"] == "segment_send":
            if event.get("server_rank") is None:
                event["server_rank"] = event["rank"]
            sends_by_request[event["request_id"]].append(event)
    pidcard = {}
    for event in events:
        if event["event"] != "server_arrival":
            continue
        rank = event.get("server_rank")
        if rank is None:
            candidates = [
                send
                for send in sends_by_request[event["request_id"]]
                if send["time"] <= event["time"]
            ]
            if not candidates:
                raise ValueError("Server arrival has no matching send evidence")
            rank = candidates[-1]["server_rank"]
        card = instance(rank)
        previous = pidcard.setdefault(event["pid"], card)
        if previous != card:
            raise ValueError("Server process mapped to conflicting instances")
    wb = {
        (e["pid"], e["step_id"]): e for e in events if e["event"] == "device_step_begin"
    }
    we = {
        (e["pid"], e["step_id"]): e for e in events if e["event"] == "device_step_end"
    }
    if not wb or wb.keys() != we.keys():
        raise ValueError(f"{arm}: missing device trace windows")
    wins = []
    reqwins = defaultdict(list)
    iidwins = defaultdict(list)
    for key, e in wb.items():
        cards = {pidcard[native[i]["pid"]] for i in e["internal_ids"]}
        if len(cards) != 1:
            raise ValueError("Ambiguous device window")
        w = dict(
            start=e["time"] - base,
            end=we[key]["time"] - base,
            card=cards.pop(),
            scheduled=e["scheduled_tokens"],
        )
        wins.append(w)
        for iid in e["internal_ids"]:
            rid = native[iid]["request_id"].removeprefix("generate-tokens-")
            reqwins[rid].append(w)
            iidwins[iid].append(w)
    req = []
    for rid, s in starts.items():
        if s["task_id"] not in batches:
            raise ValueError("Unconsumed request: exact-workload admission failed")
        w = sorted(reqwins[rid], key=lambda w: w["start"])
        sends = [e for e in byrid[rid] if e["event"] == "segment_send"]
        segments = []
        for i, send in enumerate(sends):
            limit = sends[i + 1]["time"] if i + 1 < len(sends) else float("inf")
            ns = [
                e
                for e in byrid[rid]
                if e["event"] == "native_id" and send["time"] <= e["time"] < limit
            ]
            sw = sorted(
                (z for n in ns for z in iidwins[n["internal_id"]]),
                key=lambda z: z["start"],
            )
            card = instance(send["server_rank"])
            ret = next(
                (
                    e
                    for e in byrid[rid]
                    if e["event"] == "segment_return"
                    and e["segment"] == send["segment"]
                ),
                None,
            )
            queued = next(
                (
                    e["time"] - base
                    for e in byrid[rid]
                    if e["event"] == "route_queue" and e["segment"] == send["segment"]
                ),
                send["time"] - base,
            )
            segments.append(
                dict(
                    card=card,
                    queue=queued,
                    send=send["time"] - base,
                    first=sw[0]["start"] if sw else None,
                    last=max((z["end"] for z in sw), default=None),
                    end=ret["time"] - base if ret else ends[rid]["time"] - base,
                    reason=ret.get("reason") if ret else "unfinished",
                )
            )
        pauses = []
        for pe in byrid[rid]:
            if pe["event"] != "native_pause_request":
                continue
            card = pidcard.get(pe["pid"])
            resume = [
                e["time"] - base
                for e in byrid[rid]
                if e["event"] == "native_resume"
                and e["pid"] == pe["pid"]
                and e["time"] > pe["time"]
            ]
            resume += [
                send["time"] - base
                for send in sends
                if send["time"] > pe["time"] and (instance(send["server_rank"])) != card
            ]
            if not resume:
                continue
            a = pe["time"] - base
            z = min(resume)
            # Do not hide the execution window already in flight at pause request.
            a = max(
                [a]
                + [
                    ww["end"]
                    for ww in w
                    if ww["card"] == card and ww["start"] <= a <= ww["end"]
                ]
            )
            if z > a:
                pauses.append(dict(card=card, start=a, end=z))
        req.append(
            dict(
                id=rid,
                task=s["task_id"],
                batch=batches[s["task_id"]],
                start=s["time"] - base,
                end=ends[rid]["time"] - base,
                tokens=ends[rid]["output_tokens"],
                segments=segments,
                pauses=pauses,
                card=w[-1]["card"] if w else None,
                reason=ends[rid].get("reason"),
            )
        )
    origin = json.loads((p / "clock-master.json").read_text())["origin_wall_seconds"]
    trace = rows(p / "logs/root/npu-share-ab/run/perf_tracer/master/traces-r0.jsonl")
    stages = [
        dict(
            name=e["name"][6:],
            start=origin + e["ts"] / 1e6 - base,
            end=origin + (e["ts"] + e["dur"]) / 1e6 - base,
        )
        for e in trace
        if e.get("ph") == "X" and e["name"].startswith("train.")
    ]
    last = {
        b: max((r for r in req if r["batch"] == b), key=lambda r: r["end"])
        for b in ready
    }
    return dict(
        arm=arm,
        base=base,
        end=end,
        events=events,
        phases=phases,
        batches=ready,
        requests=req,
        last=last,
        wins=wins,
        native=native,
        byrid=byrid,
        stages=stages,
        samples=rows(p / "npu-samples.jsonl"),
    )


def merged(spans):
    result = []
    for a, z in sorted(spans):
        if result and a <= result[-1][1] + 0.015:
            result[-1][1] = max(result[-1][1], z)
        else:
            result.append([a, z])
    return result


def pairs(d, begin, end, key="version"):
    waiting = {}
    result = []
    for e in d["events"]:
        if e["event"] == begin:
            waiting[e.get(key)] = e["time"] - d["base"]
        elif e["event"] == end and e.get(key) in waiting:
            result.append((waiting.pop(e.get(key)), e["time"] - d["base"]))
    return result


def translate_phase_text(value):
    """Translate phase-chart text while preserving every numeric measurement."""
    labels = {
        "Shared-group inference": "共享组推理",
        "Log-prob recomputation": "重算对数概率",
        "Advantages": "计算优势",
        "PPO update": "PPO 参数更新",
        "Checkpoint": "保存检查点",
        "Training-group snapshot export": "训练组导出权重快照",
        "Shared-group memory switch": "共享组显存切换",
        "Shared-group weight load": "共享组加载权重",
        "Dedicated-group inference": "固定推理组推理",
        "Dedicated-group weight publication": "固定推理组权重发布",
        "Batch step (odd)": "批次步骤（奇数）",
        "Batch step (even)": "批次步骤（偶数）",
        "Seconds since training loop started": "训练循环开始后的时间（秒）",
        "Rn, Xs: X = time since previous readiness node; R1 is measured from loop start. Dotted line = controller receives the complete batch.": "Rn，X秒：X 为相邻收齐节点的时间间隔；R1 从训练开始计算。虚线表示控制器取到完整批次。",
        "Shared Rn, Xs, saved +/-Ys: Y = fixed interval X - shared interval X. Positive = time saved in this interval; negative = extra time, no saving.": "共享组 Rn，X秒，节省 ±Y秒：Y = 固定组该段间隔 − 共享组该段间隔；正数为节省，负数为额外耗时。",
        "Bn, duration: acquisition entry to step-tail end; batch bars can overlap, so do not sum. Stage blocks: start-end, duration.": "Bn 后的时长：开始获取该批次至步骤收尾完成；批次条可以重叠，不能相加。阶段方块：开始−结束，时长。",
        "Stars: last consumed answer ends. Blue: execution windows (gaps <=15ms merged). Each run starts at 0s; initialization excluded.": "星号：该批最后一个被消费回答结束。蓝色：执行窗口（合并不超过15毫秒的间隙）。两组各自从0秒计时，不含初始化。",
    }
    value = "\n".join(labels.get(line, line) for line in value.split("\n"))
    value = re.sub(
        r"(\S+) \| (math|code) RL \| (\d+) questions x 6 answers \| concurrency (\d+) \| (\d+) batches",
        lambda m: (
            f"{m[1]} | {'算术' if m[2] == 'math' else '代码'} RL | {m[3]}题 × 每题6个回答 | 并发{m[4]}题 | {m[5]}个批次"
        ),
        value,
    )
    value = re.sub(r"^fixed \|", "固定组 |", value)
    value = re.sub(r"^latest \|", "共享组 |", value)
    value = value.replace("Rendered ", "生成日期 ").replace("saved ", "节省 ")
    return re.sub(r"(?<=\d)s\b", "秒", value)


def phase_chinese_font():
    """Use an installed CJK font without changing other chart families."""
    for path in sorted(font_manager.findSystemFonts()):
        name = Path(path).name.lower()
        if any(
            token in name
            for token in [
                "droidsansfallback",
                "notosanscjk",
                "sourcehansans",
                "msyh",
                "simhei",
            ]
        ):
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=path).get_name()
    raise RuntimeError(
        "Chinese phase charts require an installed CJK font (Droid Sans Fallback or Noto Sans CJK)"
    )


def readiness_intervals(nodes):
    """Return each readiness interval, with the first measured from loop start."""
    previous = 0.0
    result = {}
    for batch, time in sorted(nodes.items()):
        result[batch] = time - previous
        previous = time
    return result


def batch_step_spans(data):
    """Pair ordered batch acquisitions with serialized step-tail completion.

    Shared-mode acquisition may overlap the previous tail. These intervals are
    inclusive step lifetimes, not exclusive time slices and must not be summed.
    """
    starts = sorted(s["start"] for s in data["stages"] if s["name"] == "rollout")
    ends = sorted(
        e["time"] - data["base"]
        for e in data["phases"]
        if e["event"] == "_finish_step.end"
    )
    batches = sorted(data["batches"])
    if len(starts) != len(batches) or len(ends) != len(batches):
        raise ValueError(
            "Cannot label batch durations: acquisition/tail event counts differ"
        )
    result = []
    for batch, start, end in zip(batches, starts, ends):
        if end < start or not start <= data["batches"][batch] <= end + 0.01:
            raise ValueError(f"Inconsistent batch timing for batch {batch}")
        result.append(
            dict(
                batch=batch,
                start=start,
                end=end,
                duration=end - start,
                answers_ready=data["batches"][batch],
                last_answer=data["last"][batch]["end"],
            )
        )
    return result


def write_summary_markdown(root, summary):
    """Render English and Chinese reports from exactly the same measurements."""
    labels = {
        "Benchmark comparison": "基准测试对比",
        "Run": "实验目录",
        "Configuration": "实验配置",
        "Main comparison": "主要指标比较",
        "Metric": "指标",
        "Fixed": "固定组",
        "Shared": "共享组",
        "Training loop (s)": "训练循环耗时（秒）",
        "Questions / loop second": "题目吞吐量（题/秒）",
        "Answers / loop second": "回答吞吐量（条/秒）",
        "Reported generated tokens": "日志记录的生成 token 总数",
        "Reported tokens / loop second": "生成吞吐量（token/秒）",
        "Shared loop time reduction": "共享组训练循环耗时降低",
        "Speedup (fixed/shared)": "加速比（固定组耗时/共享组耗时）",
        "Sampled device utilization": "设备利用率采样统计",
        "Device metric (%)": "设备指标（%）",
        "Fixed mean": "固定组均值",
        "Shared mean": "共享组均值",
        "Measurement scope": "统计口径",
        "Charts": "图表",
        "Machine-readable results": "机器可读结果",
        "Token audit": "Token 核对",
        "N/A": "无数据",
        "Timing covers the measured training loop, not environment preparation or chart rendering.": "耗时覆盖实测训练循环，不包括环境准备和图表生成。",
        "Throughput divides counts by loop duration; it is not active-decode throughput.": "吞吐量使用数量除以训练循环耗时，不是仅在解码执行期间的吞吐量。",
        "Generated answers and token counts can differ between arms. Speedup alone does not demonstrate equal model quality.": "两组生成的回答及 token 数可能不同；加速比本身不能证明模型质量相同。",
        "Utilization is the arithmetic mean of recorded samples in the renderer's measurement window, not a time-weighted average. HBM includes reservations and caches.": "利用率是图表统计窗口内采样值的算术平均，不是时间加权平均；显存占用包括预留空间和缓存。",
        "Token charts reconstruct decode-window events; audit discrepancies are retained below.": "Token 图根据解码窗口事件重建，核对差异保留在下方。",
    }
    config_labels = {
        "rl": "RL 任务",
        "engine": "训练引擎",
        "concurrency": "题目并发数",
        "batches": "批次数",
        "questions": "题目数",
        "answers": "每题回答数",
        "npu_count": "NPU 数量",
        "inference_tp": "推理 TP",
        "max_head_offpolicyness": "最大头部离策略限制",
        "running_rebalance": "共享组运行中请求再均衡",
        "seed": "随机种子",
        "created": "创建时间",
    }
    notes = [
        "Timing covers the measured training loop, not environment preparation or chart rendering.",
        "Throughput divides counts by loop duration; it is not active-decode throughput.",
        "Generated answers and token counts can differ between arms. Speedup alone does not demonstrate equal model quality.",
        "Utilization is the arithmetic mean of recorded samples in the renderer's measurement window, not a time-weighted average. HBM includes reservations and caches.",
        "Token charts reconstruct decode-window events; audit discrepancies are retained below.",
    ]
    settings = summary["settings"]
    times = summary["loop_seconds"]
    arms = ["fixed", "latest"]
    for language, filename in [("en", "summary.md"), ("zh", "summary.zh.md")]:

        def tr(text):
            return labels.get(text, text) if language == "zh" else text

        def rate(count, seconds):
            return (
                f"{count / seconds:.3f}"
                if count is not None and seconds > 0
                else tr("N/A")
            )

        lines = [
            f"# {tr('Benchmark comparison')}",
            "",
            "[English](summary.md) | [中文](summary.zh.md)",
            "",
            f"{tr('Run')}: `{root.name}`",
            "",
            f"## {tr('Configuration')}",
            "",
        ]
        for key in config_labels:
            label = f"{config_labels[key]} (`{key}`)" if language == "zh" else key
            value = settings.get(key, tr("N/A"))
            lines.append(f"- {label}: {value}")
        lines += [
            "",
            f"## {tr('Main comparison')}",
            "",
            f"| {tr('Metric')} | {tr('Fixed')} | {tr('Shared')} |",
            "| --- | ---: | ---: |",
        ]

        def row(label, values):
            lines.append(f"| {tr(label)} | {values[0]} | {values[1]} |")

        row("Training loop (s)", [f"{times[a]:.3f}" for a in arms])
        row(
            "Questions / loop second",
            [rate(summary["questions"], times[a]) for a in arms],
        )
        row("Answers / loop second", [rate(summary["answers"], times[a]) for a in arms])
        counts = [summary["token_audit"][a].get("reported") for a in arms]
        row(
            "Reported generated tokens",
            [str(c) if c is not None else tr("N/A") for c in counts],
        )
        row(
            "Reported tokens / loop second",
            [rate(c, times[a]) for a, c in zip(arms, counts)],
        )
        lines += [
            "",
            f"{tr('Shared loop time reduction')}: **{summary['reduction_percent']:.2f}%**.",
            f"{tr('Speedup (fixed/shared)')}: **{rate(times['fixed'], times['latest'])}x**.",
            "",
            f"## {tr('Sampled device utilization')}",
            "",
            f"| {tr('Device metric (%)')} | {tr('Fixed mean')} | {tr('Shared mean')} |",
            "| --- | ---: | ---: |",
        ]
        utilization = summary["utilization_sample_means"]
        for key in sorted(set().union(*(utilization[a] for a in arms))):
            row(
                key,
                [
                    f"{utilization[a][key]:.2f}" if key in utilization[a] else tr("N/A")
                    for a in arms
                ],
            )
        lines += ["", f"## {tr('Measurement scope')}", ""]
        lines += [f"- {tr(note)}" for note in notes]
        lines += ["", f"## {tr('Token audit')}", ""]
        for arm, label in zip(arms, ["Fixed", "Shared"]):
            lines.append(
                f"- {tr(label)}: `{json.dumps(summary['token_audit'][arm], sort_keys=True)}`"
            )
        lines += [
            "",
            f"[{tr('Charts')}](index.html) | [{tr('Machine-readable results')}](summary.json)",
            "",
        ]
        (root / filename).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    root = args.workspace.resolve()
    settings = (
        json.loads((root / "settings.json").read_text())
        if (root / "settings.json").exists()
        else dict(
            engine="unspecified",
            rl="unspecified",
            concurrency="unknown",
            timezone="UTC",
        )
    )
    data = [load(root, a, settings) for a in ["fixed", "latest"]]
    devices = settings.get("devices", [1, 2])
    tp = settings.get("inference_tp", 1)
    instances = list(range(1, len(devices) // tp + 1))
    half = len(instances) // 2

    def instance_label(card):
        return f"Instance {card} (NPU {devices[(card - 1) * tp : card * tp]}, TP={tp})"

    n = len(data[0]["batches"])
    questions = len({r["task"] for r in data[0]["requests"]})
    if n != settings.get("batches", n) or questions != settings.get(
        "questions", questions
    ):
        raise ValueError("Completed workload differs from requested workload")
    for d in data:
        if len(d["batches"]) != n or len(d["requests"]) != questions * 6:
            raise ValueError("Unequal completed workload")
        if set(Counter(r["task"] for r in d["requests"]).values()) != {6}:
            raise ValueError("Each question must have six completed answers")
    hashes = []
    for arm in ["fixed", "latest"]:
        hashes.append(
            sorted(
                (x["task_id"], x["data_sha256"])
                for x in json.loads((root / arm / "submitted.json").read_text())
            )
        )
    if any(
        len(h) != questions or len(set(x[0] for x in h)) != questions for h in hashes
    ):
        raise ValueError("Submitted workload is incomplete or duplicated")
    if hashes[0] != hashes[1]:
        raise ValueError("Question hashes differ between arms")
    output = root / "figures"
    output.mkdir(exist_ok=True)
    date = dt.datetime.now(ZoneInfo(settings.get("timezone", "UTC"))).strftime(
        "%Y-%m-%d"
    )
    title = f"{settings['engine'].upper()} | {settings['rl']} RL | {questions} questions x 6 answers | concurrency {settings['concurrency']} | {n} batches"
    extent = math.ceil(max(d["end"] for d in data) / 10) * 10
    cmap = plt.get_cmap("tab20")
    palette = ["#2477b8", "#d68622", "#8b58aa", "#278b69", "#bd4d65", "#826342"]

    def color(b):
        return (
            palette[(b - 1) % 6]
            if n <= 6
            else cmap(((b - 1) * 2) % 20 + (1 if b > 10 else 0))
            if n <= 20
            else plt.get_cmap("turbo")((b - 1) / max(1, n - 1))
        )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    files = []

    def save(fig, name, note):
        h = fig.get_figheight()
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1 - 0.12 / h)
        fig.text(
            0.5,
            0.15 / h,
            f"Rendered {date} | {root.name}\n" + note,
            ha="center",
            fontsize=9 if name.startswith("phases/") else 8,
        )
        fig.tight_layout(
            rect=(
                0,
                (1.25 if name.startswith("phases/") else 0.7) / h,
                1,
                1 - (1.5 if getattr(fig, "request_chart", False) else 0.7) / h,
            )
        )
        path = output / (date + "_" + name + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
        files.append(str(path.relative_to(root)))
        if name.startswith("phases/"):
            cjk_font = phase_chinese_font()
            for artist in fig.findobj(match=Text):
                artist.set_text(translate_phase_text(artist.get_text()))
                artist.set_fontproperties(
                    font_manager.FontProperties(
                        family=["DejaVu Sans", cjk_font],
                        size=artist.get_fontsize(),
                        weight=artist.get_fontweight(),
                    )
                )
            for axis in fig.axes:
                axis.set_yticklabels(
                    [
                        translate_phase_text(tick.get_text())
                        for tick in axis.get_yticklabels()
                    ],
                    fontproperties=font_manager.FontProperties(
                        family=["DejaVu Sans", cjk_font], size=9
                    ),
                )
            fig.tight_layout(rect=(0, 1.25 / h, 1, 1 - 0.7 / h))
            chinese_path = path.with_name(path.stem + ".zh.png")
            fig.savefig(chinese_path, dpi=140)
            files.append(str(chinese_path.relative_to(root)))
        plt.close(fig)

    def markers(ax, d):
        for b, t in d["batches"].items():
            ax.axvline(t, color=color(b), ls=":", alpha=0.4, lw=0.8)
        ax.axvline(d["end"], color="black", ls="--", lw=0.7)

    def stage_spans(d):
        values = [
            merged((w["start"], w["end"]) for w in d["wins"] if w["card"] <= half),
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            merged((w["start"], w["end"]) for w in d["wins"] if w["card"] > half),
            [],
        ]
        for i, name in [
            (1, "recompute_logp"),
            (2, "compute_advantage"),
            (3, "ppo_update"),
            (4, "save"),
        ]:
            values[i] = [
                (s["start"], s["end"]) for s in d["stages"] if s["name"] == name
            ]
        for i, b, e in [
            (5, "shared_snapshot_export_begin", "shared_snapshot_export_end"),
            (6, "shared_memory_switch_begin", "shared_memory_switch_end"),
            (7, "shared_model_load_begin", "shared_model_load_end"),
        ]:
            values[i] = pairs(d, b, e)
        if d["arm"] == "fixed":
            values[9] = [
                (s["start"], s["end"])
                for s in d["stages"]
                if s["name"] == "update_weights"
            ][: n - 1]
        else:
            starts = [
                e["time"] - d["base"]
                for e in d["phases"]
                if e["event"] == "_publish_fixed_weights.begin"
            ]
            ends = [
                e["time"] - d["base"]
                for e in d["phases"]
                if e["event"] == "_publish_fixed_weights.end"
            ]
            values[9] = list(zip(starts, ends))[: n - 1]
        return values

    # Overview plus six-batch detail pages for long runs.
    pages = [(0, extent, "overview", n <= 6)]
    if n > 6:
        for first in range(1, n + 1, 6):
            lo = 0 if first == 1 else min(d["batches"][first - 1] for d in data)
            last = min(n, first + 5)
            hi = max(d["end"] if last == n else d["batches"][last + 1] for d in data)
            pages.append((lo, hi, f"batches_{first:03d}_{last:03d}", True))
    for lo, hi, label, labels in pages:
        fig, axs = plt.subplots(2, 1, figsize=(24, 12))
        names = [
            "Shared-group inference",
            "Log-prob recomputation",
            "Advantages",
            "PPO update",
            "Checkpoint",
            "Training-group snapshot export",
            "Shared-group memory switch",
            "Shared-group weight load",
            "Dedicated-group inference",
            "Dedicated-group weight publication",
        ]
        for ax, d in zip(axs, data):
            for y, spans in enumerate(stage_spans(d)):
                for a, z in spans:
                    if z < lo or a > hi:
                        continue
                    ax.broken_barh(
                        [(a, z - a)],
                        (y - 0.18, 0.36),
                        facecolors=(
                            "#3286b7"
                            if y in [0, 8]
                            else "#e1a146"
                            if y in [1, 2, 3]
                            else "#926ab5"
                        ),
                    )
                    if labels and z - a > 0.25:
                        ax.text(
                            (a + z) / 2,
                            y - 0.25,
                            f"{a:.2f}-{z:.2f}, {z - a:.2f}s",
                            ha="center",
                            fontsize=6,
                        )
            markers(ax, d)
            if labels:
                previous_ready = 0.0
                for b, t in sorted(d["batches"].items()):
                    interval = t - previous_ready
                    if lo <= t <= hi:
                        readiness_label = f"R{b}, {interval:.2f}s"
                        if d["arm"] == "latest":
                            fixed_interval = readiness_intervals(data[0]["batches"])[b]
                            saved = fixed_interval - interval
                            readiness_label += f", saved {saved:+.2f}s"
                        ax.text(t, -0.55, readiness_label, fontsize=7, ha="center")
                    previous_ready = t
            for step in batch_step_spans(d):
                a, z, b = step["start"], step["end"], step["batch"]
                if z < lo or a > hi:
                    continue
                y = -2 if b % 2 else -1
                ax.broken_barh(
                    [(a, z - a)], (y - 0.28, 0.56), facecolors=color(b), alpha=0.25
                )
                if labels:
                    ax.text(
                        (max(a, lo) + min(z, hi)) / 2,
                        y,
                        f"B{b}, {step['duration']:.2f}s\n{a:.2f}-{z:.2f}s",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            for b, r in d["last"].items():
                ax.plot(
                    r["end"],
                    0 if r["card"] is not None and r["card"] <= half else 8,
                    "*",
                    color="#d7a414",
                    ms=7,
                )
            ax.set(
                yticks=[-2, -1, *range(len(names))],
                yticklabels=["Batch step (odd)", "Batch step (even)", *names],
                xlim=(lo, hi),
                ylim=(9.5, -2.6),
                xlabel="Seconds since training loop started",
                title=f"{d['arm']} | {d['end']:.2f}s",
            )
            ax.grid(axis="x", alpha=0.2)
        save(
            fig,
            "phases/" + label,
            "Rn, Xs: X = time since previous readiness node; R1 is measured from loop start. Dotted line = controller receives the complete batch.\n"
            "Shared Rn, Xs, saved +/-Ys: Y = fixed interval X - shared interval X. Positive = time saved in this interval; negative = extra time, no saving.\n"
            "Bn, duration: acquisition entry to step-tail end; batch bars can overlap, so do not sum. Stage blocks: start-end, duration.\n"
            "Stars: last consumed answer ends. Blue: execution windows (gaps <=15ms merged). Each run starts at 0s; initialization excluded.",
        )
    # Request pages preserve question groups and show both cards on each page.
    for d in data:
        groups = defaultdict(list)
        for r in d["requests"]:
            groups[r["task"]].append(r)
        order = sorted(groups, key=lambda q: min(r["start"] for r in groups[q]))
        page_questions = questions if n <= 6 else 16
        for page, start in enumerate(range(0, len(order), page_questions), 1):
            rs = []
            labels = {}
            for q in order[start : start + page_questions]:
                for i, r in enumerate(sorted(groups[q], key=lambda r: r["start"]), 1):
                    rs.append(r)
                    labels[r["id"]] = f"Q{q} A{i} B{r['batch']}"
            heights = [
                max(10, sum(any(s["card"] == c for s in r["segments"]) for r in rs))
                for c in instances
            ]
            fig, axs = plt.subplots(
                len(instances),
                1,
                figsize=(22, max(15, 5 + sum(heights) * 0.115)),
                gridspec_kw={"height_ratios": heights},
            )
            fig.request_chart = True
            legend = (
                [Patch(facecolor=color(b), label=f"B{b}") for b in range(1, n + 1)]
                if n <= 10
                else []
            )
            legend += [
                Patch(facecolor="#2477b8", alpha=0.18, label="Queue / routing"),
                Patch(facecolor="#2477b8", alpha=0.45, label="Native queue"),
                Patch(facecolor="#2477b8", label="Execution span"),
                Patch(
                    facecolor="white",
                    edgecolor="#2477b8",
                    linestyle="--",
                    label="Logged pause",
                ),
                Line2D(
                    [],
                    [],
                    marker="*",
                    color="#c69b10",
                    ls="",
                    label="Last consumed answer",
                ),
            ]
            fig.legend(
                handles=legend,
                ncol=4,
                loc="upper center",
                bbox_to_anchor=(0.5, 1 - 0.45 / fig.get_figheight()),
                frameon=False,
                fontsize=9,
            )
            for card, ax in enumerate(axs, 1):
                rr = [r for r in rs if any(s["card"] == card for s in r["segments"])]
                for y, r in enumerate(rr):
                    ax.plot(
                        [r["start"], r["end"]], [y, y], color="#bbc3cc", ls=":", lw=0.5
                    )
                    segs = [s for s in r["segments"] if s["card"] == card]
                    for i, s in enumerate(segs):
                        c = color(r["batch"])
                        ax.broken_barh(
                            [(s["queue"], max(0, s["end"] - s["queue"]))],
                            (y - 0.3, 0.6),
                            facecolors=c,
                            alpha=0.18,
                        )
                        if s["first"] is not None:
                            ax.broken_barh(
                                [(s["send"], max(0, s["first"] - s["send"]))],
                                (y - 0.3, 0.6),
                                facecolors=c,
                                alpha=0.45,
                            )
                            ax.broken_barh(
                                [(s["first"], s["last"] - s["first"])],
                                (y - 0.3, 0.6),
                                facecolors=c,
                            )
                        if s["reason"] == "abort":
                            ax.plot(s["end"], y, marker=">", mfc="white", mec=c, ms=4)
                        if i + 1 < len(segs):
                            ax.broken_barh(
                                [(s["end"], max(0, segs[i + 1]["queue"] - s["end"]))],
                                (y - 0.3, 0.6),
                                facecolors="white",
                                edgecolors=c,
                                linestyle="--",
                            )
                    for pause in r["pauses"]:
                        if pause["card"] == card:
                            ax.broken_barh(
                                [(pause["start"], pause["end"] - pause["start"])],
                                (y - 0.3, 0.6),
                                facecolors="white",
                                edgecolors=color(r["batch"]),
                                linestyle="--",
                                linewidth=0.6,
                            )
                    ax.plot(r["start"], y, "D", ms=2, color="#465260")
                    if r["card"] == card:
                        ax.plot(
                            r["end"],
                            y,
                            "*" if d["last"][r["batch"]]["id"] == r["id"] else "o",
                            ms=7 if d["last"][r["batch"]]["id"] == r["id"] else 2,
                            color="#c69b10"
                            if d["last"][r["batch"]]["id"] == r["id"]
                            else "#374353",
                        )
                markers(ax, d)
                ax.set(
                    yticks=range(len(rr)),
                    yticklabels=[labels[r["id"]] for r in rr],
                    ylim=(max(1, len(rr)), -1),
                    xlim=(0, extent),
                    title=f"{d['arm']} | {instance_label(card)} | page {page}",
                    xlabel="Seconds since training loop started",
                )
                ax.tick_params(axis="y", labelsize=6)
                ax.grid(axis="x", alpha=0.15)
                if not rr:
                    ax.text(
                        0.5,
                        0.5,
                        "No inference requests on this card for these questions",
                        ha="center",
                        transform=ax.transAxes,
                    )
            save(
                fig,
                f"requests/{d['arm']}_{page:03d}",
                "Color: batch. Light: queue/routing; medium: native queue; dark: execution-window span; hollow: logged pause (not exclusive NPU occupancy). Arrow: interrupted segment. Star: last consumed answer; dotted lines: batch ready. All logical answers completed.",
            )
    fig, axs = plt.subplots(
        2 * len(devices), 2, figsize=(20, 7 * len(devices)), sharex=True, sharey=True
    )
    utilization = {}
    for k, d in enumerate(data):
        samples = [
            r
            for r in d["samples"]
            if r.get("parse_ok")
            and 0 <= (r["start"] + r["end"]) / 2 - d["base"] <= d["end"]
        ]
        if not samples:
            raise ValueError("No NPU utilization samples")
        times = [(r["start"] + r["end"]) / 2 - d["base"] for r in samples]
        utilization[d["arm"]] = {}
        for device_index, card in enumerate(devices):
            for j, metric in enumerate(["aicore", "hbm_used_pct"]):
                ax = axs[k * len(devices) + device_index, j]
                v = [r["devices"][str(card)][metric] for r in samples]
                ax.plot(times, v, ".-", lw=1, ms=2)
                markers(ax, d)
                ax.set(
                    xlim=(0, extent),
                    ylim=(0, 105),
                    title=f"{d['arm']} | Card {card} | {metric} | mean {np.mean(v):.1f}%",
                    ylabel="Percent",
                    xlabel="Seconds since loop start",
                )
                ax.grid(alpha=0.15)
                utilization[d["arm"]][f"{card}_{metric}"] = float(np.mean(v))
    save(
        fig,
        "utilization/overview",
        "Approximately 2-second samples; short peaks may be missed. HBM includes allocated/cached/reserved memory; AI Core activity is not achieved FLOPS.",
    )
    fig, axs = plt.subplots(
        2 * len(instances),
        1,
        figsize=(22, 7 * len(instances)),
        sharex=True,
        sharey=True,
    )
    audit = {}
    bins = int(extent) + 1
    for k, d in enumerate(data):
        versions = {}
        for iid, e in d["native"].items():
            rid = e["request_id"].removeprefix("generate-tokens-")
            send = max(
                (
                    s
                    for s in d["byrid"][rid]
                    if s["event"] == "segment_send" and s["time"] <= e["time"]
                ),
                key=lambda s: s["time"],
            )
            versions[iid] = send["version"]
        hist = defaultdict(lambda: np.zeros(bins, dtype=int))
        counts = Counter()
        for w in d["wins"]:
            for iid, tokens in w["scheduled"].items():
                if tokens == 1:
                    counts[iid] += 1
                    hist[(w["card"], versions[iid])][int(w["end"])] += 1
        summaries = {
            e["internal_id"]: e["tokens"]
            for e in d["events"]
            if e["event"] == "token_summary"
        }
        diffs = {
            i: counts[i] - summaries.get(i, 0)
            for i in counts.keys() | summaries.keys()
            if counts[i] != summaries.get(i, 0)
        }
        audit[d["arm"]] = dict(
            reconstructed=sum(counts.values()),
            reported=sum(r["tokens"] for r in d["requests"]),
            segment_differences=diffs,
        )
        for card in instances:
            ax = axs[k * len(instances) + card - 1]
            bottom = np.zeros(bins)
            for version in sorted({v for c, v in hist if c == card}):
                v = hist[(card, version)]
                ax.bar(
                    np.arange(bins),
                    v,
                    width=1,
                    align="edge",
                    bottom=bottom,
                    color=color(version + 1),
                    label=f"V{version}",
                )
                bottom += v
            markers(ax, d)
            ax.set(
                xlim=(0, extent),
                title=f"{d['arm']} | {instance_label(card)} | {int(sum(bottom)):,} reconstructed tokens",
                ylabel="Tokens / second",
                xlabel="Seconds since loop start",
            )
            ax.grid(axis="y", alpha=0.15)
            if n <= 10 and ax.get_legend_handles_labels()[0]:
                ax.legend(ncol=min(n, 6), fontsize=7, loc="upper right")
    save(
        fig,
        "tokens/overview",
        "One-second bins from single-token decode-window ends, not exact emission times. Colors identify weight version. Prompt/replay excluded; count discrepancies are recorded in summary.json.",
    )
    summary = dict(
        settings=settings,
        questions=questions,
        answers=questions * 6,
        batches=n,
        loop_seconds={d["arm"]: d["end"] for d in data},
        reduction_percent=100 * (1 - data[1]["end"] / data[0]["end"]),
        batch_step_timing={d["arm"]: batch_step_spans(d) for d in data},
        token_audit=audit,
        utilization_sample_means=utilization,
        figures=files,
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_markdown(root, summary)
    html = (
        '<html><head><meta charset="utf-8"><title>AReaL comparison</title></head><body><h1>'
        + title
        + "</h1><p>Open images for full resolution. Four chart families; large runs are paginated.</p>"
    )
    html += '<p><a href="summary.md">English summary</a> | <a href="summary.zh.md">中文总结</a></p>'
    for f in files:
        html += f'<p><a href="{f}">{f}</a></p>'
    (root / "index.html").write_text(html + "</body></html>")
    print(f"Generated {len(files)} PNG images in {output}", flush=True)


if __name__ == "__main__":
    main()
