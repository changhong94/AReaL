"""CPU ownership, publication markers, and error propagation for HF saves."""

import threading
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from transformers import PretrainedConfig

from areal.engine.async_hf_save import (
    AsyncHfSave,
    dram_model_state,
    independent_cpu_state,
)


def test_writer_preserves_staged_values_after_source_changes(tmp_path, monkeypatch):
    import huggingface_hub

    entered, release = threading.Event(), threading.Event()
    original = huggingface_hub.save_torch_state_dict

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(huggingface_hub, "save_torch_state_dict", blocked)
    source = {"weight": torch.arange(8, dtype=torch.float32)}
    state = independent_cpu_state(source, torch.bfloat16)
    writer = AsyncHfSave()
    try:
        writer.submit(
            tmp_path, state, PretrainedConfig(), metadata={"version": 3}, verify=True
        )
        assert entered.wait(10)
        source["weight"].zero_()
        assert not (tmp_path / "areal_save_complete.json").exists()
        release.set()
        writer.wait()
        saved = load_file(str(tmp_path / "model.safetensors"))
        torch.testing.assert_close(
            saved["weight"], torch.arange(8, dtype=torch.bfloat16), rtol=0, atol=0
        )
        assert (tmp_path / "areal_save_complete.json").exists()
    finally:
        release.set()
        writer.close()


def test_writer_failure_is_joined_and_never_marks_complete(tmp_path, monkeypatch):
    import huggingface_hub

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(huggingface_hub, "save_torch_state_dict", fail)
    writer = AsyncHfSave()
    writer.submit(tmp_path, {"weight": torch.ones(2)}, PretrainedConfig())
    with pytest.raises(OSError, match="disk unavailable"):
        writer.wait()
    assert not (tmp_path / "areal_save_complete.json").exists()
    with pytest.raises(OSError, match="disk unavailable"):
        writer.close()


def test_dram_state_restores_tied_aliases_and_persistent_buffers():
    from areal.engine.idle_state import manifest

    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.arange(4.0)))
    model.register_parameter("alias", model.weight)
    model.register_buffer("counter", torch.tensor(7, dtype=torch.int64))
    model.register_buffer("temporary", torch.tensor(9), persistent=False)
    values = {
        "weight/weight": model.weight,
        "buffer/counter": model.counter,
        "buffer/temporary": model.temporary,
    }
    layout = manifest(values)
    data = torch.empty(layout["bytes"], dtype=torch.uint8)
    for name, value in values.items():
        entry = layout["entries"][name]
        data.narrow(0, entry["offset"], entry["size"]).copy_(
            value.detach().reshape(-1).view(torch.uint8)
        )
    engine = SimpleNamespace(
        model=model, _idle_dram=SimpleNamespace(data=data, layout=layout)
    )
    saved = independent_cpu_state(dram_model_state(engine), torch.bfloat16)
    assert saved.keys() == model.state_dict().keys()
    data.zero_()
    torch.testing.assert_close(saved["weight"], saved["alias"], rtol=0, atol=0)
    assert saved["counter"].item() == 7
    assert saved["counter"].dtype == torch.int64


def test_fsdp_generic_async_save_stages_cpu_model(tmp_path, monkeypatch):
    from contextlib import nullcontext

    import torch.distributed as dist

    from areal.api import SaveLoadMeta
    from areal.engine.fsdp_engine import FSDPEngine

    monkeypatch.setattr(dist, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(dist, "get_global_rank", lambda *a: 0)
    model = torch.nn.Linear(3, 2)
    expected = {
        k: v.detach().to(torch.bfloat16).clone() for k, v in model.state_dict().items()
    }
    engine = SimpleNamespace(
        rank=0,
        cpu_group=None,
        model=model,
        model_config=PretrainedConfig(),
        config=SimpleNamespace(use_lora=False),
        lr_scheduler=SimpleNamespace(last_epoch=2),
        wait_async_save=lambda: None,
        wait_shared_snapshot=lambda **kwargs: None,
        _offload_aware_context=nullcontext,
        _compute_dtype=lambda: torch.bfloat16,
        get_version=lambda: 1,
    )
    FSDPEngine.save_async(
        engine, SaveLoadMeta(path=str(tmp_path), weight_format="hf", with_optim=False)
    )
    with torch.no_grad():
        for value in model.parameters():
            value.zero_()
    FSDPEngine.wait_async_save(engine, finalize=True)
    restored = load_file(str(tmp_path / "model.safetensors"))
    for key, value in expected.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
    assert engine._async_hf_save is None


def test_async_destination_rejects_old_shards(tmp_path):
    (tmp_path / "model-old.safetensors").write_bytes(b"old")
    writer = AsyncHfSave()
    writer.submit(tmp_path, {"weight": torch.ones(2)}, PretrainedConfig())
    with pytest.raises(FileExistsError, match="must be empty"):
        writer.close()
    assert not (tmp_path / "areal_save_complete.json").exists()


def test_fsdp_auto_retains_sync_default(monkeypatch):
    from areal.utils.async_checkpoint import AsyncMode
    from areal.utils.saver import Saver

    saver = object.__new__(Saver)
    monkeypatch.setattr(
        Saver, "_supports_fsdp_async", staticmethod(lambda engine: True)
    )
    saver._async_mode = AsyncMode.AUTO
    assert not saver._should_use_async(SimpleNamespace())
    saver._async_mode = AsyncMode.ASYNC
    assert saver._should_use_async(SimpleNamespace())


def test_asset_snapshot_preserves_files_and_observes_mutations(tmp_path):
    import json

    from areal.engine.async_hf_save import capture_save_assets, write_save_assets

    class Metadata:
        value = "before"

        def save_pretrained(self, directory):
            from pathlib import Path

            (Path(directory) / "tokenizer_config.json").write_text(
                json.dumps({"value": self.value})
            )

    tokenizer = Metadata()
    old = capture_save_assets(tokenizer)
    tokenizer.value = "after"
    new = capture_save_assets(tokenizer)
    write_save_assets(tmp_path / "old", old)
    write_save_assets(tmp_path / "new", new)
    assert (
        json.loads((tmp_path / "old/tokenizer_config.json").read_text())["value"]
        == "before"
    )
    assert (
        json.loads((tmp_path / "new/tokenizer_config.json").read_text())["value"]
        == "after"
    )


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "model.safetensors",
        "config.json",
        "./config.json",
        ".",
    ],
)
def test_asset_snapshot_rejects_unsafe_paths(tmp_path, name):
    import base64

    from areal.engine.async_hf_save import write_save_assets

    with pytest.raises(ValueError, match="Unsafe"):
        write_save_assets(tmp_path, {name: base64.b64encode(b"bad").decode()})


def test_controller_uses_files_instead_of_tokenizer_rpc(tmp_path):
    from areal.api import SaveLoadMeta
    from areal.engine.async_hf_save import write_save_assets
    from areal.infra.controller.train_controller import TrainController

    calls = []

    class Tokenizer:
        def save_pretrained(self, directory):
            from pathlib import Path

            (Path(directory) / "tokenizer.json").write_text("exact metadata")

    tokenizer = Tokenizer()
    meta = SaveLoadMeta(
        path=str(tmp_path), weight_format="hf", with_optim=False, tokenizer=tokenizer
    )
    controller = SimpleNamespace(
        _custom_function_call=lambda *a, **k: calls.append((a, k))
    )
    TrainController.save_async(controller, meta)
    args, kwargs = calls[0]
    assert args[0] == "save_async"
    assert args[1].tokenizer is None
    assert meta.tokenizer is tokenizer
    write_save_assets(tmp_path, kwargs["assets"])
    assert (tmp_path / "tokenizer.json").read_text() == "exact metadata"


def test_capture_rejects_reserved_processor_file_before_staging():
    from areal.engine.async_hf_save import capture_save_assets

    class Processor:
        def save_pretrained(self, directory):
            from pathlib import Path

            (Path(directory) / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="Unsafe"):
        capture_save_assets(processor=Processor())
