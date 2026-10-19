"""Translate trainer-private output paths for persistent inference workers."""

from pathlib import PurePosixPath


def snapshot_path(model_path, trace_dir):
    path = PurePosixPath(model_path)
    source = PurePosixPath("/run-output")
    if not path.is_relative_to(source):
        return model_path
    relative = path.relative_to(source)
    if ".." in relative.parts:
        raise ValueError("Snapshot path must stay inside the training output")
    trace = PurePosixPath(trace_dir)
    if not trace.is_relative_to("/session") or ".." in trace.parts:
        raise ValueError("Snapshot target must stay inside the development session")
    return str(trace.parent / relative)
