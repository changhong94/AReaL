"""CPU-only checks for bilingual benchmark report parity."""

import tempfile
import unittest
from pathlib import Path

from render import (
    batch_step_spans,
    readiness_intervals,
    translate_phase_text,
    write_summary_markdown,
)


class SummaryTests(unittest.TestCase):
    def fixture(self):
        return dict(
            settings=dict(
                rl="math", batch_size=32, answers=8, max_head_offpolicyness=3
            ),
            questions=192,
            answers=1536,
            loop_seconds=dict(fixed=120.0, latest=100.0),
            reduction_percent=100 * (1 - 100 / 120),
            token_audit=dict(fixed=dict(reported=139706), latest=dict(reported=137748)),
            utilization_sample_means=dict(
                fixed={"1_aicore": 23.4}, latest={"1_aicore": 33.9}
            ),
        )

    def test_reports_share_measurements_and_link_to_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_summary_markdown(root, self.fixture())
            en = (root / "summary.md").read_text(encoding="utf-8")
            zh = (root / "summary.zh.md").read_text(encoding="utf-8")
            for report in [en, zh]:
                for value in [
                    "120.000",
                    "100.000",
                    "139706",
                    "137748",
                    "16.67%",
                    "1.200x",
                    "23.40",
                    "33.90",
                ]:
                    self.assertIn(value, report)
                self.assertIn("(summary.zh.md)", report)
                self.assertIn("(summary.md)", report)
            self.assertIn("训练循环耗时", zh)
            self.assertIn("每题回答数", zh)
            self.assertIn("not a time-weighted average", en)
            self.assertIn("不是时间加权平均", zh)

    def test_missing_old_settings_and_token_counts_are_not_invented(self):
        summary = self.fixture()
        summary["settings"].pop("max_head_offpolicyness")
        summary["token_audit"]["latest"] = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_summary_markdown(root, summary)
            en = (root / "summary.md").read_text(encoding="utf-8")
            zh = (root / "summary.zh.md").read_text(encoding="utf-8")
            self.assertIn("max_head_offpolicyness: N/A", en)
            self.assertIn("(`max_head_offpolicyness`): 无数据", zh)
            self.assertIn("| Reported generated tokens | 139706 | N/A |", en)

    def test_sandbox_retry_marks_performance_as_affected(self):
        summary = self.fixture()
        summary["settings"]["rl"] = "code"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "fixed" / "request-events"
            events.mkdir(parents=True)
            (events / "events.jsonl").write_text(
                '{"event":"code_sandbox_upload_missing","retry_wait_seconds":0.4}\n'
            )
            write_summary_markdown(root, summary)
            self.assertIn(
                "不能作为共享性能收益结论", (root / "summary.zh.md").read_text()
            )
            self.assertIn("0.400s", (root / "summary.zh.md").read_text())
            self.assertTrue((root / "infrastructure_audit.json").exists())

    def test_zero_duration_does_not_divide_by_zero(self):
        summary = self.fixture()
        summary["loop_seconds"]["latest"] = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_summary_markdown(root, summary)
            self.assertIn("N/A", (root / "summary.md").read_text(encoding="utf-8"))


class BatchTimingTests(unittest.TestCase):
    def test_background_tail_overlap_is_preserved(self):
        data = dict(
            base=100,
            stages=[dict(name="rollout", start=0), dict(name="rollout", start=8)],
            phases=[
                dict(event="_finish_step.end", time=110),
                dict(event="_finish_step.end", time=118),
            ],
            batches={1: 3, 2: 8.1},
            last={1: dict(end=2.9), 2: dict(end=7)},
        )
        spans = batch_step_spans(data)
        self.assertEqual([s["duration"] for s in spans], [10, 10])
        self.assertLess(spans[1]["start"], spans[0]["end"])
        self.assertEqual(spans[1]["last_answer"], 7)
        data["phases"].pop()
        with self.assertRaisesRegex(ValueError, "event counts differ"):
            batch_step_spans(data)


class ReadinessIntervalTests(unittest.TestCase):
    def test_saving_compares_intervals_and_preserves_negative_values(self):
        fixed = readiness_intervals({3: 53.0, 1: 25.0, 2: 40.0})
        shared = readiness_intervals({1: 19.0, 2: 32.0, 3: 48.0})
        self.assertEqual(fixed, {1: 25.0, 2: 15.0, 3: 13.0})
        self.assertEqual([fixed[b] - shared[b] for b in fixed], [6.0, 2.0, -3.0])
        self.assertEqual(readiness_intervals({}), {})


class PhaseTranslationTests(unittest.TestCase):
    def test_workload_title_uses_configured_answer_count(self):
        for answers in [4, 6, 8, 16]:
            text = f"FSDP | math RL | 192 questions x {answers} answers | concurrency 16 | 6 batches"
            translated = translate_phase_text(text)
            self.assertIn(f"每题{answers}个回答", translated)
            self.assertIn("并发16题", translated)
            self.assertIn("6个批次", translated)

    def test_translation_preserves_signed_times_and_localizes_labels(self):
        self.assertEqual(
            translate_phase_text("R3, 15.63s, saved -2.15s"),
            "R3, 15.63秒, 节省 -2.15秒",
        )
        self.assertEqual(translate_phase_text("Shared-group inference"), "共享组推理")
        self.assertEqual(translate_phase_text("latest | 109.65s"), "共享组 | 109.65秒")
        self.assertEqual(
            translate_phase_text("B1, 35.30s\n0.02-35.32s"), "B1, 35.30秒\n0.02-35.32秒"
        )


if __name__ == "__main__":
    unittest.main()
