"""Expose upstream module rename aliases; keep original module/function identity."""

from pathlib import Path

root = Path("/opt/areal-runtime/vllm/vllm/entrypoints")
aliases = {
    "launchers/api_server/entry.py": "vllm.entrypoints.openai.api_server",
    "launchers/cli_args.py": "vllm.entrypoints.openai.cli_args",
    "serve/engine/protocol.py": "vllm.entrypoints.openai.engine.protocol",
}
for relative, target in aliases.items():
    path = root / relative
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = path.parent
    while directory != root:
        init = directory / "__init__.py"
        if not init.exists():
            init.write_text("")
        directory = directory.parent
    path.write_text(
        f"import importlib\nimport sys\nsys.modules[__name__] = importlib.import_module({target!r})\n"
    )
