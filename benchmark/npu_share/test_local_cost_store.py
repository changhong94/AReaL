import ast
import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

here = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, here / "templates" / (name + ".py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


c = load("local_cost_store")
ret = load("deployment_calibration")


class LocalCosts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = dict(
            engine="m",
            npu_count=2,
            inference_tp=1,
            allocation={"training": 1},
            inference_training_ratio="1:1",
            inference_replicas=1,
            inference_mode="graph",
            devices=[1, 2],
        )
        self.row = dict(
            probes={w: dict(seconds=2.0, tokens=8) for w in ("1", "2")},
            costs={w: dict(entry=1.0, publication=3.0) for w in ("1", "2")},
            actual={"1": {"compute_per_position": 0.8}},
        )

    def ctx(self, host="A", model="m", cfg="actor: megatron:d1p1t1"):
        return c.context(
            "training-layout", host, model, cfg, self.settings, "runtime", {}
        )

    def test_card_indices_not_identity_but_host_model_parallelism_are(self):
        ctx = self.ctx()
        self.settings["devices"] = [3, 4]
        self.assertEqual(ctx, self.ctx())
        for other in [
            self.ctx(host="B"),
            self.ctx(model="other"),
            self.ctx(cfg="actor: megatron:d2p1t1"),
        ]:
            self.assertNotEqual(ctx, other)
        self.settings["inference_tp"] = 2
        self.assertNotEqual(ctx, self.ctx())

    def test_revision_restore_keeps_cold_and_actual_evidence(self):
        s = c.Store(self.ctx(), self.root)
        self.assertIsNone(s.read())
        cold = copy.deepcopy(self.row)
        cold["actual"] = {}
        first = s.put(cold, source="deployment_probe", details={"workspace": "cold"})
        second = s.put(
            self.row,
            source="mixed",
            details={"origins": {"1:compute_per_position": {"batch": 2}}},
        )
        self.assertEqual(s.snapshot()["previous"], first)
        self.assertEqual(s.read()["actual"], self.row["actual"])
        self.assertEqual(s.revision(first)["data"], cold)
        invalid = s.put(None, source="invalidate", details={"reason": "clock changed"})
        self.assertIsNone(s.read())
        s.put(
            s.revision(second)["data"],
            source="rollback",
            details={"reason": "checked", "restored_revision": second},
        )
        self.assertEqual(s.snapshot()["previous"], invalid)

    def test_deployment_declares_noncanonical_layouts(self):
        self.settings.update(npu_count=6, allocation={"training": 3})
        store = c.Store(self.ctx(), self.root)
        with self.assertRaises(ValueError):
            store.put(self.row, source="mixed", details={"workspace": "old"})
        for key in ("probes", "costs", "actual"):
            self.row[key] = {
                {"1": "3", "2": "6"}[w]: v for w, v in self.row[key].items()
            }
        store.put(self.row, source="mixed", details={"workspace": "new"})
        c.adjust_layout_probe(store, "6", "entry", 3.0, "fixture check")

    def test_foreign_and_damaged_records_do_not_load(self):
        a = c.Store(self.ctx(), self.root)
        rid = a.put(self.row, source="mixed", details={"workspace": "a"})
        b = c.Store(self.ctx(host="B"), self.root)
        import shutil

        b.path.parent.mkdir(parents=True)
        shutil.copy2(a.path, b.path)
        shutil.copytree(a.path.with_suffix(".history"), b.path.with_suffix(".history"))
        self.assertIsNone(b.read())
        f = a.path.with_suffix(".history") / (rid + ".json")
        x = json.loads(f.read_text())
        x["data"]["costs"]["2"]["entry"] = 999
        f.write_text(json.dumps(x))
        self.assertIsNone(a.read())

    def test_real_inference_pair_replaces_all_probe_samples(self):
        v = ret.ObservedCosts([{"publish": 99, "restore": 88}])
        v.append({"publish": 4, "restore": 1})
        self.assertEqual(list(v), [{"publish": 4, "restore": 1}])

    def test_actual_compute_only_replaces_observed_layout_on_cache_hit(self):
        src = here.parents[1] / "areal/trainer/megatron_idle_training.py"
        cls = next(
            n for n in ast.parse(src.read_text()).body if isinstance(n, ast.ClassDef)
        )
        method = next(n for n in cls.body if getattr(n, "name", "") == "commission")
        ns = dict(
            os=os,
            json=json,
            Path=Path,
            request_trace=SimpleNamespace(emit=lambda *a, **k: None),
        )
        exec(compile(ast.Module([method], []), str(src), "exec"), ns)
        (self.root / "layout-calibration-in.json").write_text(json.dumps(self.row))
        (self.root / "calibration-inputs.json").write_text(
            json.dumps({"training-layout": {"revision": "r", "origins": {}}})
        )
        ro = SimpleNamespace(
            get_version=lambda: 0,
            staleness_manager=SimpleNamespace(
                get_stats=lambda: SimpleNamespace(accepted=0)
            ),
        )
        pool = SimpleNamespace(critical=True, trainer=SimpleNamespace(rollout=ro))
        with patch.dict(os.environ, AREAL_BENCH_WORKSPACE=str(self.root)):
            ns["commission"](pool)
        self.assertEqual(pool.live_compute, {1: 0.8})
        self.assertEqual(pool.probes[2], self.row["probes"]["2"])

    def test_adjustment_preserves_actual_and_prior_probe(self):
        s = c.Store(self.ctx(), self.root)
        old = s.put(
            self.row,
            source="mixed",
            details={
                "origins": {"1:compute_per_position": {"source": "runtime_observation"}}
            },
        )
        c.adjust_layout_probe(s, "1", "compute_seconds", 7.0, "checked probe bias")
        self.assertEqual(s.read()["actual"], self.row["actual"])
        self.assertEqual(
            s.snapshot()["details"]["origins"]["1:compute_per_position"]["source"],
            "runtime_observation",
        )
        self.assertEqual(s.revision(old)["data"]["probes"]["1"]["seconds"], 2.0)
        self.assertEqual(s.read()["probes"]["1"]["seconds"], 7.0)
        with self.assertRaises(ValueError):
            c.adjust_layout_probe(s, "2", "entry", 1.0, "")

    def test_nonfinite_measurement_never_replaces_valid_revision(self):
        s = c.Store(self.ctx(), self.root)
        rev = s.put(self.row, source="mixed", details={"workspace": "a"})
        self.row["actual"]["1"]["entry"] = float("nan")
        with self.assertRaises(ValueError):
            s.put(self.row, source="mixed", details={"workspace": "b"})
        self.assertEqual(s.snapshot()["revision"], rev)


if __name__ == "__main__":
    unittest.main()
