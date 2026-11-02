"""CPU checks for the all-rank native allocator guard; no device needed."""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch


class Headroom(unittest.TestCase):
    def exercise(self, free_before, free_after, peer_free):
        tree = ast.parse(
            (Path(__file__).parents[1] / "areal/engine/megatron_engine.py").read_text()
        )
        m = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_idle_optimizer_headroom"
        )
        calls = []
        samples = iter((free_before, free_after))

        def gather(rows, row, group):
            rows[:] = [row, dict(row, rank=1, after=peer_free)]

        ns = dict(
            os=os,
            current_platform=NS(
                synchronize=lambda: None, empty_cache=lambda: calls.append("released")
            ),
            dist=NS(all_gather_object=gather),
        )
        exec(compile(ast.Module(body=[m], type_ignores=[]), "guard", "exec"), ns)
        engine = NS(
            rank=0,
            world_size=2,
            cpu_group="cpu",
            logger=NS(info=lambda *a: None),
            _shared_memory=NS(backend=NS(free_memory=lambda: (next(samples), 0))),
        )
        with patch.dict(os.environ, AREAL_IDLE_COLLECTIVE_MARGIN_BYTES="100"):
            ns[m.name](engine)
        return calls

    def test_recheck_reclaimable_not_estimated_cache(self):
        self.assertEqual(self.exercise(30, 130, 120), ["released"])

    def test_other_rank_rejection_reaches_healthy_rank(self):
        with self.assertRaises(RuntimeError):
            self.exercise(160, 160, 80)

    def test_enough_on_all_ranks_does_not_flush(self):
        self.assertEqual(self.exercise(160, 160, 140), [])
