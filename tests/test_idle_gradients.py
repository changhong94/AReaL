"""Gradient-prefix correctness across independent DP1/DP2 process groups."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from areal.engine.idle_gradients import add_prefix_gradients, export_prefix_gradients


def gradient_worker(rank, world, rendezvous, root, prefix):
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=world
    )
    torch.manual_seed(5)
    initial = torch.randn(7, 5, dtype=torch.float64)
    x = torch.randn(11, 5, dtype=torch.float64)
    y = torch.randn(11, 7, dtype=torch.float64)
    weights = torch.tensor([1, 2, 0, 4, 1, 3, 2, 1, 0, 5, 1], dtype=torch.float64)

    def compute(weight, start, stop):
        return (
            ((x[start:stop] @ weight.T - y[start:stop]) ** 2)
            * weights[start:stop, None]
        ).sum() / weights.sum()

    reference = torch.nn.Parameter(initial.clone())
    opt = torch.optim.AdamW([reference], lr=0.01, foreach=False)
    compute(reference, 0, 11).backward()
    full_grad = reference.grad.clone()
    opt.step()
    model = torch.nn.Linear(5, 7, bias=False, dtype=torch.float64)
    model.weight = torch.nn.Parameter(initial.clone())
    engine = SimpleNamespace(
        model=model, rank=rank, world_size=world, get_version=lambda: 3
    )
    if world == 1:
        compute(model.weight, 0, prefix).backward()
        export_prefix_gradients(engine, root, "batch-2")
    else:
        # Compute each rank's suffix, then reproduce DP mean + DP loss multiplier.
        begin = prefix + rank
        loss = (
            (
                ((x[begin:11:world] @ model.weight.T - y[begin:11:world]) ** 2)
                * weights[begin:11:world, None]
            ).sum()
            / weights.sum()
            * world
        )
        loss.backward()
        dist.all_reduce(model.weight.grad)
        model.weight.grad.div_(world)
        suffix = model.weight.grad.clone()
        mesh = init_device_mesh("cpu", (world,))
        model.weight = torch.nn.Parameter(
            distribute_tensor(initial.clone(), mesh, [Shard(0)])
        )
        model.weight.grad = distribute_tensor(suffix, mesh, [Shard(0)])
        add_prefix_gradients(engine, root, "batch-2")
        expected = distribute_tensor(full_grad, mesh, [Shard(0)])
        torch.testing.assert_close(
            model.weight.grad.to_local(), expected.to_local(), rtol=1e-12, atol=1e-12
        )
        actual_opt = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        actual_opt.step()
        expected_weight = distribute_tensor(reference.detach(), mesh, [Shard(0)])
        torch.testing.assert_close(
            model.weight.to_local(), expected_weight.to_local(), rtol=1e-12, atol=1e-12
        )
        for key in ["exp_avg", "exp_avg_sq"]:
            target = distribute_tensor(opt.state[reference][key], mesh, [Shard(0)])
            torch.testing.assert_close(
                actual_opt.state[model.weight][key].to_local(),
                target.to_local(),
                rtol=1e-12,
                atol=1e-12,
            )
        assert actual_opt.state[model.weight]["step"].item() == 1
        with pytest.raises(RuntimeError, match="already been added"):
            add_prefix_gradients(engine, root, "batch-2")
    dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.parametrize("prefix", [1, 5, 9])
def test_prefix_handoff_preserves_gradients_adam_and_sample_weights(tmp_path, prefix):
    with tempfile.TemporaryDirectory(prefix="areal-gradient-", dir="/dev/shm") as temp:
        root = str(Path(temp) / "prefix")
        for world in [1, 2]:
            mp.spawn(
                gradient_worker,
                args=(world, str(tmp_path / str(world)), root, prefix),
                nprocs=world,
                join=True,
            )


def test_prefix_only_gradient_and_wrong_generation():
    with tempfile.TemporaryDirectory(prefix="areal-gradient-", dir="/dev/shm") as temp:
        model = torch.nn.Linear(3, 5, bias=False)
        model.weight.grad = torch.ones_like(model.weight)
        engine = SimpleNamespace(
            model=model, rank=0, world_size=1, get_version=lambda: 7
        )
        path = str(Path(temp) / "prefix")
        export_prefix_gradients(engine, path, "b7")
        model.weight.grad = None
        with pytest.raises(RuntimeError, match="generation mismatch"):
            add_prefix_gradients(engine, path, "b8")
        assert model.weight.grad is None
        add_prefix_gradients(engine, path, "b7")
        torch.testing.assert_close(
            model.weight.grad, torch.ones_like(model.weight), rtol=0, atol=0
        )


def test_diagnostic_replay_preserves_original_gradient(tmp_path, monkeypatch):
    from areal.engine.idle_gradients import replay_segments
    from areal.utils import request_trace

    records = []
    monkeypatch.setattr(
        request_trace, "emit", lambda event, **kw: records.append((event, kw))
    )
    dist.init_process_group(
        "gloo", init_method="file://" + str(tmp_path / "group"), rank=0, world_size=1
    )
    try:
        model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
        model.weight.grad = torch.full_like(model.weight, 2.0)
        engine = SimpleNamespace(
            model=model,
            rank=0,
            cpu_group=dist.group.WORLD,
            parallel_helper=SimpleNamespace(dp_size=1),
            optimizer_zero_grad=lambda: model.zero_grad(set_to_none=True),
            _idle_recorded_segments=[3.0],
        )

        def replay(chunks, multiplier):
            (model.weight.sum() * chunks[0] * multiplier).backward()

        replay_segments(engine, replay, "diagnostic")
        torch.testing.assert_close(
            model.weight.grad, torch.full_like(model.weight, 2.0), rtol=0, atol=0
        )
        assert records[0][1]["relative_l2"] == pytest.approx(0.5)
        assert records[0][1]["norm_ratio"] == pytest.approx(1.5)
    finally:
        dist.destroy_process_group()


def test_diagnostic_restore_recreates_exact_gradient_coverage():
    from areal.engine.idle_gradients import restore_gradient_snapshot

    model = torch.nn.Linear(3, 2)
    saved = {"weight": torch.ones_like(model.weight)}
    model.weight.grad = None
    model.bias.grad = torch.ones_like(model.bias)
    restore_gradient_snapshot(SimpleNamespace(model=model), saved)
    torch.testing.assert_close(model.weight.grad, saved["weight"], rtol=0, atol=0)
    assert model.bias.grad is None
