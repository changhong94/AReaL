# SPDX-License-Identifier: Apache-2.0
"""Admission control for one inference replica borrowed by its resident trainer."""

from threading import Lock


class BorrowedRolloutPool:
    """Withdraw when only one workflow remains before a complete training batch.

    Rank zero owns the training state. Rank one remains available for inference.
    This class only gates new workflows; in-flight work is stopped separately by
    the worker executor and the inference server's acknowledged pause operation.
    """

    def __init__(
        self,
        batch_size: int,
        preserve_requests: bool = False,
        replicas_per_group: int = 1,
    ):
        if batch_size < 1:
            raise ValueError("Borrowed rollout requires a positive batch size")
        if replicas_per_group < 1:
            raise ValueError("Replica groups must be nonempty")
        self.replicas_per_group = replicas_per_group
        self.batch_size = batch_size
        self.preserve_requests = preserve_requests
        self.consumed = 0
        self.state = "inference"
        self._next_rank = 0
        self._lock = Lock()

    def _withdraw_target(self) -> int:
        # Caller holds _lock. Keep a one-item batch usable until it completes.
        # A frozen request may own the last answer needed by this batch.
        remaining = (
            self.batch_size if self.preserve_requests else max(1, self.batch_size - 1)
        )
        return self.consumed + remaining

    def training_prefetch_target(self) -> int:
        with self._lock:
            return self._withdraw_target()

    def _withdraw_if_ready(self, accepted: int) -> None:
        threshold = self._withdraw_target()
        if self.state == "inference" and accepted >= threshold:
            self.state = "draining"

    def choose(self, accepted: int) -> tuple[int, bool]:
        with self._lock:
            old_state = self.state
            self._withdraw_if_ready(accepted)
            if self.state != "inference":
                rank = (
                    self.replicas_per_group + self._next_rank % self.replicas_per_group
                )
                self._next_rank += 1
                return rank, self.state != old_state
            rank = self._next_rank % (2 * self.replicas_per_group)
            self._next_rank += 1
            return rank, False

    def batch_ready(self, count: int) -> None:
        with self._lock:
            if count != self.batch_size:
                raise ValueError(
                    "Borrowed rollout requires fixed-size complete batches"
                )
            self.consumed += count
            self.state = "training"

    def withdraw(self) -> None:
        with self._lock:
            self.state = "training"

    def release(self, accepted: int) -> None:
        with self._lock:
            self.state = "inference"
            self._withdraw_if_ready(accepted)
