"""CPU scoring events, independent of the selected AReaL checkout."""

import json
import os
import threading
import time
from pathlib import Path

_lock = threading.Lock()


def emit(event, **fields):
    directory = os.environ.get("AREAL_REQUEST_TRACE_DIR")
    if not directory:
        return
    record = dict(event=event, time=time.time(), pid=os.getpid(), **fields)
    with _lock:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"events-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
