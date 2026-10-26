from types import SimpleNamespace

import pytest
import torch

from areal.engine.idle_plan import segment_bounds, select_microbatches


def test_segment_boundaries_preserve_even_rounds_and_every_item():
    counts = [3, 4, 1, 8, 5, 2, 1]
    bounds = segment_bounds(counts, 8)
    assert bounds == [(0, 4), (4, 7)]
    assert [i for a, b in bounds for i in range(a, b)] == list(range(len(counts)))
    assert all(a % 2 == 0 for a, b in bounds)


def test_odd_plan_has_zero_loss_dummy_without_modifying_real_data():
    items = [
        dict(input_ids=torch.tensor([i]), loss_mask=torch.ones(1, 2)) for i in range(3)
    ]
    plan = SimpleNamespace(
        mbs=items,
        padded_mbs=items.copy(),
        group_lens=[2] * 3,
        padding_lengths=[0] * 3,
        padded_to_lengths=[2] * 3,
        old_cu_seqlens_list=None,
        forward_indices=[0, 1, 2],
        backward_indices=[0, 1, 2],
    )
    a = select_microbatches(plan, 0, 3, 0, 2)
    b = select_microbatches(plan, 0, 3, 1, 2)
    assert [x["input_ids"].item() for x in a.mbs] == [0, 2]
    assert b.mbs[0] is items[1]
    assert a.data == b.data == {}
    assert len(a.mbs) == len(b.mbs) == 2
    assert b.mbs[1]["loss_mask"].count_nonzero() == 0
    assert b.padded_mbs[1]["loss_mask"].count_nonzero() == 0
    assert all(x["loss_mask"].count_nonzero() == 2 for x in items)
    assert sum(x["loss_mask"].sum().item() for x in a.mbs + b.mbs) == 6
    with pytest.raises(ValueError):
        select_microbatches(plan, 2, 1, 0, 2)
