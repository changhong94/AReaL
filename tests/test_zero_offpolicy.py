# SPDX-License-Identifier: Apache-2.0
"""No-lookahead admission must advance only with published model versions."""

from types import SimpleNamespace

import pytest

from areal.infra.borrowed_rollout import BorrowedRolloutPool
from areal.infra.staleness_manager import StalenessManager


@pytest.mark.parametrize("batch_size", [1, 16])
def test_zero_admission_waits_for_publication_across_batches(batch_size):
    version = [0]
    manager = StalenessManager(
        SimpleNamespace(get_version=lambda: version[0]), 32, batch_size, 0
    )
    pool = BorrowedRolloutPool(batch_size)
    assert manager.get_pending_limit() == batch_size
    for epoch in range(3):
        assert manager.get_capacity() == batch_size
        for _ in range(batch_size):
            manager.on_rollout_enqueued()
            manager.on_rollout_submitted()
            manager.on_rollout_accepted()
        assert manager.get_capacity() == 0
        pool.batch_ready(batch_size)
        assert manager.get_capacity() == 0
        version[0] += 1
        assert manager.get_capacity() == batch_size
        pool.release(manager.get_stats().accepted)
        assert pool.training_prefetch_target() == (epoch + 1) * batch_size + max(
            1, batch_size - 1
        )


def test_zero_rejection_replaces_work_and_recovery_preserves_one_batch():
    provider = SimpleNamespace(get_version=lambda: 5)
    manager = StalenessManager(provider, 32, 16, 0)
    manager.on_version_recovered(5)
    assert manager.get_capacity() == 16
    manager.on_rollout_enqueued()
    manager.on_rollout_submitted()
    assert manager.get_capacity() == 15
    manager.on_rollout_rejected()
    assert manager.get_capacity() == 16
