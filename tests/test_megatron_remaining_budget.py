"""Remaining-work decisions must not charge already completed CPU work."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "entry_budget", Path(__file__).parents[1] / "areal/trainer/idle_entry.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TestRemainingBudget(unittest.TestCase):
    def setUp(self):
        self.b = m.EntryBudget()
        self.b.observe(1, 1, "compute", 68)
        self.b.observe(2, 2, "compute", 37)
        self.b.observe(2, 2, "prepare", 45)
        self.b.observe(1, 1, "prepare", 0)
        self.b.observe(1, 1, "return", 0)
        self.b.observe(1, 2, "return", 0)

    def test_completed_work_is_not_charged_again(self):
        self.assertEqual(self.b.decide(3, 0, terminal=True)["world_size"], 1)
        d = self.b.decide(3, 0, terminal=True, remaining_prepare={1: 20, 2: 1})
        self.assertEqual(d["world_size"], 2)
        self.assertEqual(
            d["candidates"][2]["remaining_prepare"]
            - d["candidates"][1]["remaining_prepare"],
            -19,
        )

    def test_expensive_remaining_transfer_rejects_switch(self):
        self.assertEqual(
            self.b.decide(3, 0, terminal=True, remaining_prepare={1: 1, 2: 40})[
                "world_size"
            ],
            1,
        )

    def test_generation_still_has_priority(self):
        self.assertEqual(
            self.b.decide(3, 1, remaining_prepare={1: 20, 2: 0})["world_size"], 1
        )

    def test_invalid_estimate_is_rejected(self):
        with self.assertRaises(ValueError):
            self.b.decide(3, 0, remaining_prepare={1: -1, 2: 0})


if __name__ == "__main__":
    unittest.main()
