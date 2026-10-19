"""Benchmark-only immutable host snapshot; separate from mutable offload backups."""

import hashlib


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(
            value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes()
        )
    return digest.hexdigest()


class InitialWeights:
    def __init__(self, model):
        # clone() is mandatory even if model tensors are already on the CPU.
        self._tensors = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        self.digest = tensor_digest(self._tensors)
        self.bytes = sum(t.numel() * t.element_size() for t in self._tensors.values())

    def restore(self, model):
        import torch

        current = model.state_dict()
        if current.keys() != self._tensors.keys():
            raise RuntimeError("Model schema changed during reusable benchmark")
        if tensor_digest(self._tensors) != self.digest:
            raise RuntimeError("Immutable initial snapshot was modified")
        with torch.no_grad():
            for name, original in self._tensors.items():
                target = current[name]
                if target.shape != original.shape or target.dtype != original.dtype:
                    raise RuntimeError("Model tensor schema changed: " + name)
                target.copy_(original)
        observed = tensor_digest(model.state_dict())
        if observed != self.digest:
            raise RuntimeError("Restored model differs from initial snapshot")
        return dict(
            initial_sha256=self.digest,
            restored_sha256=observed,
            initial_dram_bytes=self.bytes,
        )
