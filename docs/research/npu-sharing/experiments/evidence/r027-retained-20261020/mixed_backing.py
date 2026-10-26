import json

import torch

from areal.engine.vllm_ext.dynamic_kv import AscendSparseBackend, SparseKVPages

torch.npu.set_device(0)
backend = AscendSparseBackend(torch.device("npu:0"))
size = backend.page * 3
block = backend.page // 4
raw = [backend.allocate(size + backend.page) for _ in range(2)]
tensors = [
    t[(-t.data_ptr()) % backend.page : ((-t.data_ptr()) % backend.page) + size]
    for t in raw
]
pool = SparseKVPages(backend, tensors, block, 12, 0)
pool.reconcile({1, 5, 6})
for i, tensor in enumerate(tensors):
    for b in (1, 5, 6):
        tensor[b * block : (b + 1) * block].fill_(i * 20 + b)
backend.synchronize()
pool.pause_preserved()
backups = [backend.backup(t.data_ptr() + backend.page) for t in tensors]
for tensor in tensors:
    backend.unmap(tensor.data_ptr() + backend.page)
pool.swapped[1] = backups
pool.mapped.remove(1)
payload = torch.empty(6 * block, dtype=torch.uint8, device="npu:0")
pool.copy_retained_blocks([6, 1, 5], payload.data_ptr())
actual = payload.cpu().reshape(6, block)
assert all(
    torch.all(row == value).item()
    for row, value in zip(actual, [6, 1, 5, 26, 21, 25], strict=True)
)
assert pool.paused and pool.swapped and pool.mapped == {0}
result = dict(
    bytes=payload.numel(),
    page_bytes=backend.page,
    mixed_hbm_dram_equal=True,
    source_remained_paused=True,
    remapped_pages=0,
)
print(json.dumps(result), flush=True)
open("/evidence/mixed-backing.json", "w").write(json.dumps(result, indent=2))
pool.discard_preserved()
