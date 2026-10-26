"""Build dependency-free, bilingual coverage reports without using NPUs."""

import argparse
import csv
import html
import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def collect(root, final=False):
    planned = read_rows(root / "planned-cases.tsv")
    finished = read_rows(root / "cases.tsv")
    rows = {row["case"]: dict(row) for row in planned}
    for row in finished:
        rows.setdefault(row["case"], {}).update(row)
    # Legacy suites did not persist a full plan; do not invent missing cases.
    for log in sorted(root.glob("*.log")):
        if re.fullmatch(r"[A-Z]\d\d", log.stem):
            rows.setdefault(log.stem, {"case": log.stem})
    results = []
    for case, row in rows.items():
        log = root / f"{case}.log"
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        matches = re.findall(r"^WORKSPACE=(.+)$", text, re.MULTILINE)
        workspace = root / Path(matches[-1].strip()).name if matches else None
        if workspace and not workspace.is_dir():
            workspace = None
        state = read_json(workspace / "state.json") if workspace else {}
        summary = read_json(workspace / "summary.json") if workspace else {}
        settings = summary.get("settings", {})
        if workspace:
            settings = read_json(workspace / "settings.json") | settings
        phase = state.get("phase")
        rc = row.get("exit_code", "")
        if rc and rc != "0":
            status = "failed"
        elif phase == "failed":
            status = "failed"
        elif rc == "0":
            status = "complete" if phase == "complete" else "prepared"
        elif phase == "complete":
            status = "complete"
        elif log.exists():
            status = "interrupted" if final else "running"
        else:
            status = "not_run"
        times = summary.get("loop_seconds", {})
        fixed, shared = times.get("fixed"), times.get("latest")
        gain = (
            (100 * (fixed - shared) / fixed)
            if (status == "complete" and number(fixed) and fixed > 0 and number(shared))
            else None
        )
        error = state.get("error", "")
        if status == "failed" and not error:
            error = next(
                (line for line in reversed(text.splitlines()) if line.strip()), ""
            )
        results.append(
            dict(
                case=case,
                status=status,
                parameters=row | settings,
                workspace=workspace.name if workspace else None,
                fixed_seconds=fixed,
                shared_seconds=shared,
                reduction_percent=gain,
                tokens={
                    a: summary.get("token_audit", {}).get(a, {}).get("reported")
                    for a in ["fixed", "latest"]
                },
                error=error[:500],
                log=f"{case}.log" if log.exists() else None,
            )
        )
    return results


def write_reports(root, final=False):
    results = collect(root, final)
    counts = dict(Counter(row["status"] for row in results))
    (root / "summary.json").write_text(
        json.dumps(
            {"scope": "coverage", "counts": counts, "cases": results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    labels = {
        "complete": "完成",
        "failed": "失败",
        "prepared": "预检查通过",
        "running": "运行中",
        "interrupted": "已中断",
        "not_run": "未运行",
    }
    for lang in ["en", "zh"]:
        zh = lang == "zh"
        title = "Coverage 测试汇总" if zh else "Coverage benchmark summary"
        headers = (
            [
                "用例",
                "配置",
                "状态",
                "固定耗时(s)",
                "共享耗时(s)",
                "节省比例",
                "Tokens 固定/共享",
                "详情",
            ]
            if zh
            else [
                "Case",
                "Configuration",
                "Status",
                "Fixed (s)",
                "Shared (s)",
                "Reduction",
                "Tokens fixed/shared",
                "Details",
            ]
        )
        note = (
            "耗时仅指训练循环，不含初始化。正节省值表示共享更快，负值表示更慢；失败项不计性能收益。生成 token 数可能不同，性能结果不代表训练质量提升。"
            if zh
            else "Times cover training loops only, excluding initialization. Positive reduction means shared is faster; negative means slower. Failed cases have no performance claim. Token counts may differ; speed does not establish training quality."
        )

        def esc(v):
            return str(v).replace("|", "\\|").replace("\n", " ")

        def fmt(v):
            return f"{v:.3f}" if number(v) else "—"

        markdown_rows, html_rows = [], []
        for row in results:
            p = row["parameters"]
            config = " / ".join(
                str(p.get(k, "?")) for k in ["rl", "engine", "inference_mode"]
            )
            config += " · " + " ".join(
                f"{label}={p.get(k, '?')}"
                for label, k in [
                    ("NPU", "npu_count"),
                    ("TP", "inference_tp" if "inference_tp" in p else "tp"),
                    ("ratio", "inference_training_ratio"),
                    ("c", "concurrency"),
                    ("batch", "batch_size"),
                    ("answers", "answers"),
                    ("batches", "batches"),
                    (
                        "offpolicy",
                        "max_head_offpolicyness"
                        if "max_head_offpolicyness" in p
                        else "offpolicy",
                    ),
                ]
            )
            gain = row["reduction_percent"]
            cells = [
                row["case"],
                config,
                labels[row["status"]] if zh else row["status"],
                fmt(row["fixed_seconds"]),
                fmt(row["shared_seconds"]),
                f"{gain:+.2f}%" if number(gain) else "—",
                " / ".join(
                    str(row["tokens"][a]) if row["tokens"][a] is not None else "—"
                    for a in ["fixed", "latest"]
                ),
            ]
            links = []
            if row["workspace"]:
                folder = root / row["workspace"]
                for filename, label in [
                    ("summary.zh.md" if zh else "summary.md", "Summary"),
                    ("index.html", "Charts"),
                    ("state.json", "State"),
                ]:
                    if (folder / filename).exists():
                        links.append((label, quote(row["workspace"] + "/" + filename)))
            if row["log"]:
                links.append(("Log", quote(row["log"])))
            markdown_rows.append(
                "| "
                + " | ".join(map(esc, cells))
                + " | "
                + " · ".join(f"[{label}]({url})" for label, url in links)
                + " |"
            )
            details = " · ".join(
                f'<a href="{html.escape(url)}">{label}</a>' for label, url in links
            )
            if row["error"]:
                details += f"<details><summary>Error</summary><pre>{html.escape(row['error'])}</pre></details>"
            html_rows.append(
                "<tr>"
                + "".join(f"<td>{html.escape(c)}</td>" for c in cells)
                + f"<td>{details}</td></tr>"
            )
        status_line = " · ".join(
            f"{labels.get(k, k) if zh else k}: {v}" for k, v in counts.items()
        )
        suffix = ".zh" if zh else ""
        md = f"# {title}\n\n[English](summary.md) | [中文](summary.zh.md) | [HTML](index{suffix}.html)\n\n{status_line}\n\n{note}\n\n"
        md += (
            "| "
            + " | ".join(headers)
            + " |\n| "
            + " | ".join(["---"] * len(headers))
            + " |\n"
            + "\n".join(markdown_rows)
            + "\n"
        )
        document = f'''<!doctype html><html lang="{lang}"><meta charset="utf-8"><title>{title}</title>
<style>body{{font:15px system-ui;margin:30px;color:#24324a}}table{{border-collapse:collapse;width:100%}}th,td{{padding:12px;border:1px solid #ddd;text-align:left}}th{{background:#edf2f7}}tr:nth-child(even){{background:#f8fafc}}pre{{white-space:pre-wrap;max-width:400px}}a{{color:#0969da}}</style>
<h1>{title}</h1><p><a href="index.html">English</a> | <a href="index.zh.html">中文</a> | <a href="summary.json">JSON</a></p><p>{html.escape(status_line)}</p><p>{note}</p><table><tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr>{"".join(html_rows)}</table></html>'''
        (root / f"summary{suffix}.md").write_text(md, encoding="utf-8")
        (root / f"index{suffix}.html").write_text(document, encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--final", action="store_true", help="Mark unfinished started cases interrupted"
    )
    args = parser.parse_args()
    write_reports(args.directory.resolve(), args.final)
