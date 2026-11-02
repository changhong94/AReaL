"""CPU coverage of parameter-relative DP resharding contracts."""

import importlib.util
import sys
import unittest
from pathlib import Path

path = Path(__file__).parents[1] / "areal/engine/megatron_utils/idle_state.py"
spec = importlib.util.spec_from_file_location("megatron_idle_state_contract", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
R = module.ShardRange


class TestResharding(unittest.TestCase):
    def test_uneven_dp1_to_dp2_and_back(self):
        original = list(range(13))
        shards = [R(0, 7, 13), R(7, 13, 13)]
        pieces = []
        for target in shards:
            copies = module.intersections([R(0, 13, 13)], target)
            out = [None] * (target.end - target.start)
            for _, source_offset, target_offset, size in copies:
                out[target_offset : target_offset + size] = original[
                    source_offset : source_offset + size
                ]
            pieces.append(out)
        restored = [None] * 13
        for index, src, dst, size in module.intersections(shards, R(0, 13, 13)):
            restored[dst : dst + size] = pieces[index][src : src + size]
        self.assertEqual(restored, original)

    def test_changed_bucket_boundaries(self):
        self.assertEqual(
            module.intersections([R(0, 3, 10), R(3, 10, 10)], R(2, 8, 10)),
            [(0, 2, 0, 1), (1, 0, 1, 5)],
        )

    def test_reject_overlap_gap_and_wrong_total(self):
        for ranges in [
            [R(0, 6, 10), R(5, 10, 10)],
            [R(0, 4, 10), R(5, 10, 10)],
            [R(0, 10, 11)],
        ]:
            with self.assertRaises(ValueError):
                module.intersections(ranges, R(0, 10, 10))

    def test_empty_owner(self):
        self.assertEqual(module.intersections([R(0, 0, 2), R(0, 2, 2)], R(0, 0, 2)), [])

    def test_invalid_destination(self):
        with self.assertRaises(ValueError):
            module.intersections([R(0, 2, 2)], R(-1, 2, 2))


class TestMetadataExtraction(unittest.TestCase):
    def test_parked_parameters_are_not_read_or_copied(self):
        from types import SimpleNamespace

        class ParkedParameter:
            def __deepcopy__(self, memo):
                raise AssertionError("Parked parameter storage was accessed")

        groups = [
            {"params": [ParkedParameter()], "step": 9, "lr": 1e-5},
            {"params": [], "step": 0, "lr": 1e-5},
        ]
        optimizer = SimpleNamespace(
            grad_scaler=None, optimizer=SimpleNamespace(param_groups=groups)
        )
        actual = module.optimizer_metadata(optimizer)
        self.assertEqual(
            actual,
            {
                "optimizer": {
                    "param_groups": [{"step": 9, "lr": 1e-5}, {"step": 9, "lr": 1e-5}]
                }
            },
        )
        self.assertEqual(groups[1]["step"], 0)

    def test_inconsistent_steps_rejected(self):
        from types import SimpleNamespace

        optimizer = SimpleNamespace(
            grad_scaler=None,
            optimizer=SimpleNamespace(
                param_groups=[
                    {"params": [object()], "step": 1},
                    {"params": [object()], "step": 2},
                ]
            ),
        )
        with self.assertRaises(RuntimeError):
            module.optimizer_metadata(optimizer)


class TestTransport(unittest.TestCase):
    def test_ram_transport_preserves_values_without_global_crc_change(self):
        import os

        try:
            import torch  # noqa: F401 -- check runtime availability
        except ImportError:
            self.skipTest("torch is required for transport roundtrip")
        if not hasattr(os, "memfd_create"):
            self.skipTest("Linux RAM file descriptor required")
        before = torch.serialization.get_crc32_options()
        descriptor = os.memfd_create("megatron-state-test")
        try:
            path = f"/proc/self/fd/{descriptor}"
            value = {
                "master": torch.arange(257, dtype=torch.float32),
                "moment": torch.tensor([0.1, -0.7]),
                "step": 7,
                "scalar": torch.tensor(3.0),
            }
            module.save_transport(value, path)
            loaded = torch.load(path, weights_only=False, mmap=True)
            self.assertTrue(module._metadata_equal(loaded, value))
            self.assertEqual(torch.serialization.get_crc32_options(), before)
        finally:
            os.close(descriptor)


class TestSchedulerRestore(unittest.TestCase):
    def test_repeated_load_replaces_instead_of_accumulates_steps(self):
        try:
            import torch  # noqa: F401 -- check runtime availability
        except ImportError:
            self.skipTest("torch is required for metadata comparisons")

        class AdditiveScheduler:
            num_steps = 7

            def load_state_dict(self, state):
                self.num_steps += state["num_steps"]

            def state_dict(self):
                return {"num_steps": self.num_steps}

        scheduler = AdditiveScheduler()
        for step in (9, 10, 9):
            module.restore_scheduler(scheduler, {"num_steps": step})
            self.assertEqual(scheduler.num_steps, step)


class TestMetadata(unittest.TestCase):
    def test_optimizer_progress_is_not_just_tensor_state(self):
        import copy

        try:
            import torch  # noqa: F401 -- check runtime availability
        except ImportError:
            self.skipTest("torch is required for metadata tensor checks")
        original = {
            "optimizer": {
                "param_groups": [
                    {"step": 7, "lr": 1e-5, "betas": (0.9, 0.999), "weight_decay": 0.01}
                ]
            },
            "scaler": torch.tensor(1.0),
        }
        self.assertTrue(module._metadata_equal(original, copy.deepcopy(original)))
        for field, value in [
            ("step", 6),
            ("lr", 2e-5),
            ("betas", (0.8, 0.999)),
            ("weight_decay", 0.02),
        ]:
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed["optimizer"]["param_groups"][0][field] = value
                self.assertFalse(module._metadata_equal(original, changed))
        changed = copy.deepcopy(original)
        changed["scaler"] += 1
        self.assertFalse(module._metadata_equal(original, changed))


class TestInPlaceOptimizerMetadata(unittest.TestCase):
    def test_preserves_storage_and_group_identity(self):
        import types

        try:
            import torch  # noqa: F401 -- check runtime availability
        except ImportError:
            self.skipTest("torch unavailable")
        parameter = torch.ones(2)
        moment = torch.zeros(2)
        group = dict(
            params=[parameter],
            wd_mult=1,
            lr_mult=1,
            is_expert_parallel=False,
            is_decoupled_lr=False,
            step=1,
            lr=0.1,
        )
        mirror = torch.tensor(1)
        inner = types.SimpleNamespace(
            param_groups=[group],
            state={parameter: {"exp_avg": moment}},
            _get_group_step_tensor=lambda value: mirror,
        )
        outer = types.SimpleNamespace(optimizer=inner, grad_scaler=None)
        metadata = module.optimizer_metadata(outer)
        metadata["optimizer"]["param_groups"][0].update(step=7, lr=0.02)
        params, state, pointer = group["params"], inner.state, moment.data_ptr()
        module.restore_optimizer_metadata(outer, metadata)
        self.assertIs(inner.param_groups[0], group)
        self.assertIs(group["params"], params)
        self.assertIs(inner.state, state)
        self.assertEqual(moment.data_ptr(), pointer)
        self.assertEqual(mirror.item(), 7)
        self.assertTrue(
            module._metadata_equal(module.optimizer_metadata(outer), metadata)
        )


class TestParkedStepMirror(unittest.TestCase):
    def test_missing_mirror_rejects_without_allocating(self):
        import types

        try:
            import torch  # noqa: F401 -- check runtime availability
        except ImportError:
            self.skipTest("torch unavailable")
        group = {"params": [torch.ones(1)]}
        inner = types.SimpleNamespace(_step_tensors={})
        with self.assertRaisesRegex(RuntimeError, "preallocated"):
            module._prepared_step_mirror(inner, group, object())
        mirror = torch.tensor(4)
        inner._step_tensors[(id(group), "cpu", None)] = mirror
        self.assertIs(module._prepared_step_mirror(inner, group, object()), mirror)


if __name__ == "__main__":
    unittest.main()
