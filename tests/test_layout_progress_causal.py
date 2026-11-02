"""CPU-only tests: no historical benchmarks and no engine/case labels."""

import importlib.util
import unittest
from pathlib import Path


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "areal/trainer" / (name + ".py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


predict = module("layout_horizon").predict_layout
progress = module("layout_progress").estimate_progress


class TestCausalLayout(unittest.TestCase):
    def test_unfinished_sibling_is_censored_not_discarded(self):
        tail = module("layout_progress").conditional_tail
        self.assertEqual(tail([2, 4], [7], 1, 10)[-1], (9, 1.0))
        self.assertEqual(tail([2, 4], [], 1, 10)[-1], (3, 1.0))

    def test_busy_unknown_drain_prefers_proven_split_but_terminal_does_not(self):
        d = predict(
            "b",
            {"a": 55, "b": 20},
            {"a": 4, "b": 1},
            {"a": 2, "b": 4},
            4,
            1,
            100,
            None,
            False,
            baseline="a",
            retains_inference={"a": True, "b": False},
        )
        self.assertEqual(
            (d["world_size"], d["reason"]),
            ("a", "protect_live_inference_unpriced_alternative"),
        )

    def test_partial_competitor_is_dominated_by_compute_alone(self):
        d = predict(
            "b",
            {"a": 70, "b": 35},
            {"a": None, "b": 4},
            {"a": 0, "b": 0},
            0,
            None,
            0,
            None,
            True,
            baseline="a",
            retains_inference={"a": True, "b": False},
        )
        self.assertEqual((d["world_size"], d["reason"]), ("b", "dominates_lower_bound"))

    def test_cpu_score_overlap_does_not_remove_unfinished_generation(self):
        p = progress({"completed": {"q": range(3)}, "active": []}, None, 1, 4)
        self.assertEqual(p["unfinished_fraction"], 0.25)
        d = predict(
            "a",
            {"a": 60, "b": 20},
            {"a": 0, "b": 0},
            {"a": 0, "b": 0},
            1,
            0.025,
            0,
            None,
            False,
            baseline="a",
            retains_inference={"a": True, "b": False},
            progress=p,
        )
        self.assertEqual(d["candidates"]["b"]["total"], 40)
        p["unfinished_fraction"] = 1.0
        self.assertEqual(
            predict(
                "a",
                {"a": 60, "b": 20},
                {"a": 0, "b": 0},
                {"a": 0, "b": 0},
                1,
                0.025,
                0,
                None,
                False,
                baseline="a",
                retains_inference={"a": True, "b": False},
                progress=p,
            )["world_size"],
            "a",
        )

    def test_maximum_of_two_residuals_not_max_of_means(self):
        e = [
            dict(
                active=[dict(request_id=k, tokens=1) for k in ("x", "y")],
                window={"decode": [dict(concurrency=2, seconds=1, count=1)]},
            )
        ]
        route = dict(
            active=[
                dict(
                    request_id=k,
                    rank=0,
                    metrics=dict(question_id=k, generated_tokens=0, max_new_tokens=10),
                )
                for k in ("x", "y")
            ],
            completed={},
            window_observations=[
                dict(done=True, question_id="z", length=n) for n in (2, 4)
            ],
        )
        # Independently draw 1 or 3; max equals 3 in three of four equally likely cases.
        self.assertAlmostEqual(progress(route, e, 1, 2)["drain_seconds"], 2.5)
        e[0]["active"].pop()
        self.assertIsNone(progress(route, e, 1, 2)["drain_seconds"])

    def test_ambiguous_keeps_current(self):
        d = predict(
            "b",
            {"a": None, "b": 20},
            {"a": 1, "b": 1},
            {"a": 0, "b": 0},
            0,
            None,
            0,
            None,
            True,
            baseline="a",
            retains_inference={"a": True, "b": False},
        )
        self.assertEqual(d["world_size"], "b")


if __name__ == "__main__":
    unittest.main()
