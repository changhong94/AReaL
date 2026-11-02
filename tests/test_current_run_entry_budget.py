"""Evidence-based decisions must not infer benefits from a backend or prior."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "budget", Path(__file__).parents[1] / "areal/trainer/idle_entry.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Budget = module.EntryBudget


class TestCurrentRunBudget(unittest.TestCase):
    def measured(self, single=60, dual=30):
        b = Budget()
        for world, compute in [(1, single), (2, dual)]:
            for phase, seconds in [
                ("compute", compute),
                ("prepare", 2),
                ("return", 5),
                ("tail", 0),
            ]:
                b.observe(1, world, phase, seconds)
        return b

    def test_cold_start_is_conservative(self):
        self.assertEqual(Budget().decide(1, 0)["world_size"], 1)

    def test_one_observation_per_path_is_enough(self):
        self.assertEqual(self.measured().decide(2, 0)["world_size"], 2)

    def test_no_layout_is_preferred_when_it_is_slower(self):
        self.assertEqual(self.measured(30, 60).decide(2, 0)["world_size"], 1)

    def test_missing_compute_is_not_invented_from_speedup(self):
        b = self.measured()
        b.samples = type(b.samples)(s for s in b.samples if s[1:3] != (2, "compute"))
        d = b.decide(2, 0)
        self.assertEqual(d["world_size"], 1)
        self.assertIn("compute", d["missing"][2])

    def test_missing_baseline_cost_prevents_comparison(self):
        b = self.measured()
        b.samples = type(b.samples)(s for s in b.samples if s[1:3] != (1, "return"))
        self.assertEqual(b.decide(2, 0)["world_size"], 1)

    def test_empty_queue_does_not_inherit_busy_drain(self):
        b = self.measured()
        b.observe_drain(1, 2, 16, 100)
        d = b.decide_paths(
            2, current_layout=2, drain_work={1: 0, 2: 0}, include_tail=True
        )
        self.assertEqual(d["world_size"], 2)
        self.assertEqual(d["candidates"][2]["drain"], 0)
        self.assertEqual(d["candidates"][2]["publication"], 5)

    def test_busy_queue_uses_matching_current_observation(self):
        b = self.measured()
        b.observe_drain(1, 2, 16, 100)
        self.assertEqual(
            b.decide_paths(2, current_layout=2, drain_work={1: 0, 2: 16})["world_size"],
            1,
        )

    def test_unobserved_request_context_is_unknown(self):
        b = self.measured()
        b.observe_drain(1, 2, 16, 1)
        d = b.decide_paths(2, current_layout=1, drain_work={1: 0, 2: 32})
        self.assertIn("drain", d["missing"][2])
        self.assertEqual(d["world_size"], 1)

    def test_new_run_does_not_inherit_previous_budget(self):
        old = self.measured()
        self.assertEqual(old.decide(2, 0)["world_size"], 2)
        self.assertEqual(Budget().decide(2, 0)["world_size"], 1)

    def test_unknown_remaining_work_is_not_zero(self):
        d = self.measured().decide(2, 0, remaining_prepare={1: 0, 2: None})
        self.assertEqual(d["world_size"], 1)
        self.assertIn("remaining_prepare", d["missing"][2])

    def test_completed_work_is_structural_zero(self):
        d = self.measured(30, 29).decide(2, 0, remaining_prepare={1: 2, 2: 0})
        self.assertEqual(d["world_size"], 2)

    def test_layouts_are_executor_supplied_not_backend_rules(self):
        b = Budget(
            baseline_layout="original", candidate_layouts=("original", "available")
        )
        for world, compute in [("original", 60), ("available", 20)]:
            for phase, seconds in [("compute", compute), ("prepare", 0), ("return", 0)]:
                b.observe(1, world, phase, seconds)
        self.assertEqual(b.decide(2, 0)["world_size"], "available")

    def test_ties_keep_baseline(self):
        self.assertEqual(self.measured(30, 30).decide(2, 0)["world_size"], 1)

    def test_terminal_flag_does_not_erase_actual_publication(self):
        b = self.measured()
        b.observe(1, 2, "return", 100)
        self.assertEqual(b.decide(2, 0, terminal=True)["world_size"], 1)

    def test_unmeasured_handoff_prevents_switch(self):
        d = self.measured().decide_paths(2, current_layout=1, include_handoff=True)
        self.assertEqual(d["world_size"], 1)
        self.assertIn("handoff", d["missing"][2])

    def test_unknown_lease_count_is_not_an_empty_queue(self):
        d = self.measured().decide_paths(
            2, current_layout=1, drain_work={1: 0, 2: None}
        )
        self.assertEqual(d["world_size"], 1)
        self.assertIn("drain", d["missing"][2])

    def test_invalid_samples_are_not_evidence(self):
        for value in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                Budget().observe(1, 1, "compute", value)


if __name__ == "__main__":
    unittest.main()
