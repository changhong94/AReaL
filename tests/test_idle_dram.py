"""CPU distributed correctness checks for the RAM-only state transport."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from areal.engine.idle_state import DramState, manifest, tensor_slice


class CPUPlatform:
    def synchronize(self):
        pass

    def current_device(self):
        return 0

    def set_device(self, device):
        pass


def worker(rank, world, rendezvous, root, phase):
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=world
    )
    torch.manual_seed(1)
    model = torch.nn.Linear(5, 7, bias=False)
    model.register_buffer("scalar", torch.tensor(3.0))
    if world > 1:
        mesh = init_device_mesh("cpu", (world,))
        model.weight = torch.nn.Parameter(
            distribute_tensor(model.weight.detach(), mesh, [Shard(0)])
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for value in model.parameters():
        optimizer.state[value] = dict(
            step=torch.tensor(0.0),
            exp_avg=torch.zeros_like(value),
            exp_avg_sq=torch.zeros_like(value),
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    engine = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        rank=rank,
        world_size=world,
        cpu_group=dist.group.WORLD,
        get_version=lambda: 0,
        set_version=lambda _: None,
    )
    store = DramState(engine, str(Path(root) / phase))
    if phase != "single":
        result = store.load(
            str(Path(root) / ("single" if phase == "dual" else "dual")), 1
        )
        assert result["verified_bytes"] > 0
    if phase == "dual":
        with torch.no_grad():
            for value in model.parameters():
                value.add_(1.0)
                optimizer.state[value]["exp_avg"].add_(2.0)
                optimizer.state[value]["exp_avg_sq"].add_(3.0)
                optimizer.state[value]["step"].add_(1.0)
        scheduler.last_epoch = 1
    if phase == "returned":
        torch.manual_seed(1)
        expected = torch.nn.Linear(5, 7, bias=False).weight.detach() + 1
        torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
        state = optimizer.state[model.weight]
        assert torch.equal(state["exp_avg"], torch.full_like(model.weight, 2.0))
        assert torch.equal(state["exp_avg_sq"], torch.full_like(model.weight, 3.0))
        assert state["step"].item() == 1
        assert scheduler.last_epoch == 1
    with patch("areal.infra.platforms.current_platform", CPUPlatform()):
        store.start()
        store.wait()
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.slow
def test_single_dual_single_preserves_all_state(tmp_path):
    with tempfile.TemporaryDirectory(prefix="areal-dram-test-", dir="/dev/shm") as root:
        for phase, world in [("single", 1), ("dual", 2), ("returned", 1)]:
            mp.spawn(
                worker,
                args=(world, str(tmp_path / phase), root, phase),
                nprocs=world,
                join=True,
            )


def test_manifest_has_aligned_disjoint_ranges():
    values = {"a": torch.zeros(7, 5), "b": torch.tensor(2.0)}
    result = manifest(values)
    assert result["entries"]["b"]["offset"] % 64 == 0
    assert result["entries"]["b"]["offset"] >= 7 * 5 * 4


def test_noncontiguous_state_is_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        tensor_slice(torch.zeros(3, 4).t(), 0, 1)


def test_parked_backup_view_uses_tensor_offset_and_rejects_unpreserved():
    from areal.engine.idle_state import parked_storage_view

    backup = torch.zeros(64, dtype=torch.uint8)
    allocation = SimpleNamespace(
        tag="actor_weights", handle=(0, 64, 1000), cpu_backup_tensor=backup
    )
    memory = SimpleNamespace(
        broken=False,
        inventory=lambda: {1000: allocation},
        _missing={1000: (allocation, "cpu")},
    )
    view = parked_storage_view(memory, 1016, 8, torch.float32, (2,))
    view.copy_(torch.tensor([3.0, 4.0]))
    assert torch.equal(backup[16:24].view(torch.float32), torch.tensor([3.0, 4.0]))
    assert not backup[:16].any() and not backup[24:].any()
    with pytest.raises(RuntimeError, match="unique"):
        parked_storage_view(memory, 1060, 8, torch.float32, (2,))
    memory._missing[1000] = (allocation, "discarded")
    with pytest.raises(RuntimeError, match="preserved"):
        parked_storage_view(memory, 1016, 8, torch.float32, (2,))
    memory._missing[1000] = (allocation, "cpu")
    allocation.cpu_backup_tensor = None
    with pytest.raises(RuntimeError, match="Invalid"):
        parked_storage_view(memory, 1016, 8, torch.float32, (2,))


def test_export_poll_is_nonblocking_and_propagates_failure():
    from concurrent.futures import Future

    state = object.__new__(DramState)
    state.generation = 3
    state.future = Future()
    assert state.poll() == {"generation": 3, "done": False, "snapshot": None}
    state.future.set_exception(RuntimeError("copy failed"))
    with pytest.raises(RuntimeError, match="copy failed"):
        state.poll()
