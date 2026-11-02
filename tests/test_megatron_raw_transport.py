"""Actual Torch CPU round trips for leased raw state transport."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

path = Path(__file__).parents[1] / "areal/engine/megatron_utils/idle_state.py"
spec = importlib.util.spec_from_file_location("raw_contract", path)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class TestRawTransport(unittest.TestCase):
    def test_roundtrip_nonzero_offset_bf16_empty_scalar(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"AREAL_IDLE_RAW_TRANSPORT": "1"}),
        ):
            source = dict(
                a=torch.arange(40, dtype=torch.float32)[7:31],
                b=torch.arange(8, dtype=torch.bfloat16),
                empty=torch.empty(0),
                scalar=torch.tensor(7),
                meta=[3, {"version": 4}],
            )
            p = Path(d) / "state.pt"
            result = m.save_transport(source, p)
            restored = m.load_transport(p)
            self.assertEqual(result["format"], "raw")
            for k in ("a", "b", "empty", "scalar"):
                self.assertTrue(torch.equal(source[k], restored[k]))
                self.assertEqual(source[k].shape, restored[k].shape)
            self.assertEqual(source["meta"], restored["meta"])
            source["a"].fill_(0)
            self.assertEqual(restored["a"][0].item(), 7)

    def test_recycled_capacity_preserves_exact_logical_size(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"AREAL_IDLE_RAW_TRANSPORT": "1"}),
        ):
            root = Path(d)
            pool = root / "recycled-raw"
            pool.mkdir()
            slot = pool / "tensor-old.bin"
            slot.write_bytes(b"x" * 4096)
            generation = root / "generation-1"
            generation.mkdir()
            p = generation / "state.pt"
            x = torch.arange(17, dtype=torch.bfloat16)
            with patch(
                "os.posix_fallocate",
                side_effect=AssertionError("unexpected allocation"),
            ):
                result = m.save_transport({"x": x}, p)
            self.assertTrue(result["recycled"])
            self.assertEqual(next(generation.glob("*.bin")).stat().st_size, 4096)
            self.assertTrue(torch.equal(m.load_transport(p)["x"], x))
            self.assertFalse(slot.exists())

    def test_recycle_large_small_large_exact_bytes(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"AREAL_IDLE_RAW_TRANSPORT": "1"}),
        ):
            root = Path(d)
            pool = root / "recycled-raw"
            pool.mkdir()
            for version, size in enumerate([1024, 17, 1000]):
                generation = root / f"generation-{version}"
                generation.mkdir()
                p = generation / "state.pt"
                x = torch.arange(size, dtype=torch.float32) + version
                result = m.save_transport({"x": x}, p)
                self.assertEqual(result["recycled"], version > 0)
                loaded = m.load_transport(p)
                self.assertTrue(torch.equal(loaded["x"], x))
                del loaded
                for f in generation.glob("*.bin"):
                    f.rename(pool / f.name)

    def test_undersized_slot_not_claimed(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"AREAL_IDLE_RAW_TRANSPORT": "1"}),
        ):
            root = Path(d)
            pool = root / "recycled-raw"
            pool.mkdir()
            slot = pool / "tensor-small.bin"
            slot.write_bytes(b"x")
            generation = root / "generation-1"
            generation.mkdir()
            p = generation / "state.pt"
            result = m.save_transport({"x": torch.ones(9)}, p)
            self.assertFalse(result["recycled"])
            self.assertTrue(slot.exists())
            self.assertTrue(torch.equal(m.load_transport(p)["x"], torch.ones(9)))

    def test_concurrent_writers_claim_slot_once(self):
        from concurrent.futures import ThreadPoolExecutor

        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"AREAL_IDLE_RAW_TRANSPORT": "1"}),
        ):
            root = Path(d)
            pool = root / "recycled-raw"
            pool.mkdir()
            (pool / "tensor-old.bin").write_bytes(b"x" * 4096)
            generation = root / "generation-1"
            generation.mkdir()

            def write(i):
                p = generation / f"rank-{i}.pt"
                result = m.save_transport({"x": torch.full((100,), i)}, p)
                return result, m.load_transport(p)["x"]

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(write, [1, 2]))
            self.assertEqual(sum(r[0]["recycled"] for r in results), 1)
            for i, (_, tensor) in enumerate(results, 1):
                self.assertTrue(torch.equal(tensor, torch.full((100,), i)))

    def test_truncated_bytes_rejected(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"AREAL_IDLE_RAW_TRANSPORT": "1"}),
        ):
            p = Path(d) / "state.pt"
            m.save_transport({"x": torch.ones(9)}, p)
            data = next(Path(d).glob("*.bin"))
            data.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "Truncated"):
                m.load_transport(p)

    def test_async_missing_or_wrong_ready_rejected(self):
        import json

        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"AREAL_IDLE_RAW_TRANSPORT": "1"}),
        ):
            p = Path(d) / "state.pt"
            m.save_transport(
                dict(async_export=True, rank=0, world=1, version=4, x=torch.ones(3)), p
            )
            with self.assertRaises(FileNotFoundError):
                m.load_transport(p)
            ready = Path(d) / "ready-0.json"
            ready.write_text(json.dumps(dict(rank=0, world=1, version=3)))
            with self.assertRaisesRegex(RuntimeError, "generation"):
                m.load_transport(p)
            ready.write_text(json.dumps(dict(rank=0, world=1, version=4)))
            self.assertEqual(m.load_transport(p)["version"], 4)

    def test_noncontiguous_rejected(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"AREAL_IDLE_RAW_TRANSPORT": "1"}),
        ):
            with self.assertRaisesRegex(ValueError, "contiguous"):
                m.save_transport({"x": torch.ones(4, 4).t()}, Path(d) / "state.pt")
            self.assertFalse(list(Path(d).glob("*.partial")))

    def test_legacy_transport_compatible(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"AREAL_IDLE_RAW_TRANSPORT": "0"}),
        ):
            p = Path(d) / "state.pt"
            x = torch.arange(9)
            m.save_transport({"x": x}, p)
            self.assertTrue(torch.equal(m.load_transport(p)["x"], x))


if __name__ == "__main__":
    unittest.main()
