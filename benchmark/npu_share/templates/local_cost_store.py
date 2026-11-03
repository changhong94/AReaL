"""Host-owned calibration records. Never mount the store into the experiment."""

import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from pathlib import Path


class IdentityUnavailable(RuntimeError):
    """Do not share timings when deployment identity cannot be verified."""


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def host_identity():
    # No Docker hostname, device index, PCI address, or device serial in this key.
    machine = Path("/etc/machine-id").read_text().strip()
    if not machine or machine == "uninitialized":
        raise IdentityUnavailable("Cannot identify calibration host")
    uuid = Path("/sys/class/dmi/id/product_uuid")
    return digest([machine, uuid.read_text().strip() if uuid.is_file() else None])


def context(kind, host, model, config, settings, runtime, env):
    # The public interface deliberately accepts device count and topology, never card IDs.
    topology = {
        k: settings[k]
        for k in (
            "engine",
            "npu_count",
            "inference_tp",
            "allocation",
            "inference_training_ratio",
            "inference_replicas",
            "inference_mode",
        )
    }
    config = re.sub(
        r"(?m)^total_train_steps: [0-9]+$", "total_train_steps: <run length>", config
    )
    return dict(
        layouts=sorted(
            {str(settings["allocation"]["training"]), str(settings["npu_count"])}
        ),
        schema=1,
        kind=kind,
        host=host,
        model=model,
        topology=topology,
        config=digest(config),
        runtime=runtime,
        environment=digest(env),
    )


def host_context(kind, profile, settings, workspace, env):
    model = Path(profile["model"]).resolve()
    files = {
        str(p.relative_to(model)): file_hash(p)
        for p in sorted(model.rglob("*"))
        if p.is_file()
    }
    if not files:
        raise IdentityUnavailable("No model files for calibration identity")
    manifest = json.loads((workspace / "latest-source-manifest.json").read_text())[
        "files"
    ]
    adapter = json.loads((workspace / "adapter-manifest.json").read_text())
    driver_info = Path(profile["driver"]) / "version.info"
    if not driver_info.is_file():
        raise IdentityUnavailable("Missing driver version for calibration identity")
    runtime = dict(
        image=settings["arm_image_ids"]["latest"],
        driver=file_hash(driver_info),
        source=digest({k: v for k, v in manifest.items() if k.startswith("areal/")}),
        adapter=digest(adapter),
    )
    return context(
        kind,
        host_identity(),
        digest([str(model), files]),
        (workspace / "latest/config.yaml").read_text(),
        settings,
        runtime,
        env,
    )


def positive(x):
    return type(x) in (int, float) and math.isfinite(x) and x > 0


def valid(kind, data, layouts=()):
    if not isinstance(data, dict):
        return False
    if kind == "inference-return":
        rows = data.get("samples")
        return (
            isinstance(rows, list)
            and bool(rows)
            and all(
                isinstance(s, dict)
                and set(s) == {"publish", "restore"}
                and all(positive(v) for v in s.values())
                for s in rows
            )
        )
    if kind != "training-layout":
        return False
    if set(data) != {"probes", "costs", "actual"}:
        return False
    for k, fields in (
        ("probes", {"tokens", "seconds"}),
        ("costs", {"entry", "publication"}),
    ):
        if not isinstance(data[k], dict) or set(data[k]) != set(layouts):
            return False
        if not all(
            isinstance(row, dict)
            and set(row) == fields
            and all(positive(v) for v in row.values())
            for row in data[k].values()
        ):
            return False
    return (
        isinstance(data["actual"], dict)
        and set(data["actual"]) <= set(layouts)
        and all(
            isinstance(row, dict)
            and set(row) <= {"compute_per_position", "entry", "publication"}
            and all(positive(v) for v in row.values())
            for row in data["actual"].values()
        )
    )


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".cal-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, sort_keys=True, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Store:
    def __init__(self, ctx, root=None):
        self.ctx = ctx
        self.key = digest(ctx)
        self.path = (
            (
                Path(root)
                if root
                else Path.home() / ".cache/areal-deployment-calibration"
            )
            / ctx["host"]
            / ctx["kind"]
            / (self.key + ".json")
        )

    def snapshot(self):
        try:
            pointer = json.loads(self.path.read_text())
            row = self.revision(pointer["revision"])
            return row if row.get("data") is not None else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def revision(self, revision):
        if not re.fullmatch(r"[0-9a-f]{32}", revision):
            raise ValueError("Invalid revision")
        path = self.path.with_suffix(".history") / (revision + ".json")
        row = json.loads(path.read_text())
        check = row.pop("checksum")
        if (
            check != digest(row)
            or row["context"] != self.ctx
            or row["revision"] != revision
        ):
            raise ValueError("Invalid calibration revision")
        if row["data"] is not None and not valid(
            self.ctx["kind"], row["data"], self.ctx.get("layouts", ())
        ):
            raise ValueError("Invalid calibration measurements")
        row["checksum"] = check
        return row

    def read(self):
        row = self.snapshot()
        return row["data"] if row else None

    def put(self, data, *, source, details):
        if source not in {
            "deployment_probe",
            "runtime_observation",
            "mixed",
            "invalidate",
            "rollback",
            "research_adjustment",
        }:
            raise ValueError("Unknown calibration source")
        if not details or (
            source in {"rollback", "invalidate", "research_adjustment"}
            and not details.get("reason")
        ):
            raise ValueError("Missing provenance or revision reason")
        if (data is None) != (source == "invalidate") or (
            data is not None
            and not valid(self.ctx["kind"], data, self.ctx.get("layouts", ()))
        ):
            raise ValueError("Invalid calibration measurement")
        try:
            previous = json.loads(self.path.read_text())["revision"]
        except (OSError, KeyError, ValueError, TypeError):
            previous = None
        rev = uuid.uuid4().hex
        row = dict(
            schema=1,
            revision=rev,
            previous=previous,
            created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),  # noqa: UP017 -- host supports Python 3.9
            context=self.ctx,
            source=source,
            details=details,
            data=data,
        )
        row["checksum"] = digest(row)
        atomic(self.path.with_suffix(".history") / (rev + ".json"), row)
        atomic(self.path, dict(revision=rev))
        return rev


def adjust_layout_probe(store, layout, field, seconds, reason):
    """Adjust the fallback only; preserve every actual and the parent revision."""
    import copy

    if (
        store.ctx["kind"] != "training-layout"
        or layout not in store.ctx.get("layouts", ())
        or field not in {"compute_seconds", "entry", "publication"}
        or not positive(seconds)
    ):
        raise ValueError("Invalid layout probe adjustment")
    row = store.snapshot()
    if row is None or not reason:
        raise ValueError("Need usable prior revision and reason")
    data = copy.deepcopy(row["data"])
    section, key, metric = (
        ("probes", "seconds", "compute_per_position")
        if field == "compute_seconds"
        else ("costs", field, field)
    )
    previous = data[section][layout][key]
    data[section][layout][key] = seconds
    origins = copy.deepcopy(row["details"].get("origins", {}))
    detail = dict(
        source="research_adjustment",
        reason=reason,
        prior_revision=row["revision"],
        layout=layout,
        metric=metric,
        previous_seconds=previous,
        seconds=seconds,
    )
    if metric not in data["actual"].get(layout, {}):
        origins[f"{layout}:{metric}"] = detail
    return store.put(
        data,
        source="research_adjustment",
        details=dict(reason=reason, adjustment=detail, origins=origins),
    )
