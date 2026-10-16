"""Host-side, read-only NPU telemetry. Preserve raw output and measured query windows."""

import json
import re
import subprocess
import threading
import time
from pathlib import Path


def parse_info(raw):
    devices = {}
    processes = []
    current = None
    in_processes = False
    for line in raw.splitlines():
        p = [x.strip() for x in line.split("|")]
        if "Process id" in line:
            in_processes = True
        if len(p) > 4 and "910B2" in p[1]:
            current = p[1].split()[0]
            nums = p[3].split()
            devices[current] = {
                "name": "910B2",
                "health": p[2],
                "power_w": float(nums[0]),
                "temperature_c": float(nums[1]),
            }
        elif current is not None and len(p) > 4 and ":" in p[2]:
            nums = re.findall(r"\d+(?:\.\d+)?", p[3])
            assert len(nums) == 5, (p[3], nums)
            used, total = float(nums[3]), float(nums[4])
            devices[current].update(
                aicore=float(nums[0]),
                hbm_used_mb=used,
                hbm_total_mb=total,
                hbm_used_pct=100 * used / total,
                bus_id=p[2],
                memory_raw=p[3],
            )
            current = None
        elif in_processes and len(p) > 5 and p[2].isdigit():
            try:
                processes.append(
                    {
                        "device": p[1].split()[0],
                        "chip": p[1].split()[1],
                        "pid": int(p[2]),
                        "name": p[3],
                        "memory_mb": float(p[4]),
                    }
                )
            except (ValueError, IndexError):
                pass
    return devices, processes


def parse_usages(raw):
    values = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            values[k.strip()] = float(v.strip())
        except ValueError:
            values[k.strip()] = v.strip()
    return values


def query(args):
    start = time.time()
    mono = time.monotonic()
    p = subprocess.run(args, text=True, capture_output=True, timeout=15)
    return {
        "start": start,
        "end": time.time(),
        "query_seconds": time.monotonic() - mono,
        "returncode": p.returncode,
        "stdout": p.stdout,
        "stderr": p.stderr,
    }


def sampler(folder, done, devices=(1, 2)):
    folder = Path(folder)

    def info_loop():
        with (folder / "npu-samples.jsonl").open("w") as f:
            while not done.is_set():
                mono = time.monotonic()
                try:
                    row = query(["npu-smi", "info"])
                    row["devices"], row["processes"] = parse_info(row["stdout"])
                    row["parse_ok"] = (
                        row["returncode"] == 0 and len(row["devices"]) == 8
                    )
                except Exception as e:
                    row = {"start": time.time(), "error": repr(e), "parse_ok": False}
                f.write(json.dumps(row) + "\n")
                f.flush()
                done.wait(max(0, 2 - (time.monotonic() - mono)))

    def detailed_loop():
        with (folder / "npu-usages.jsonl").open("w") as f:
            while not done.is_set():
                mono = time.monotonic()
                for device in map(str, devices):
                    if done.is_set():
                        break
                    try:
                        row = query(
                            ["npu-smi", "info", "-t", "usages", "-i", device, "-c", "0"]
                        )
                        row["device"] = device
                        row["metrics"] = parse_usages(row["stdout"])
                    except Exception as e:
                        row = {"start": time.time(), "device": device, "error": repr(e)}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                done.wait(max(0, 5 - (time.monotonic() - mono)))

    threads = [
        threading.Thread(target=f, daemon=True) for f in [info_loop, detailed_loop]
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    row = query(["npu-smi", "info"])
    devices, processes = parse_info(row["stdout"])
    assert len(devices) == 8 and all(
        0 <= x["hbm_used_pct"] <= 100 and 0 <= x["aicore"] <= 100
        for x in devices.values()
    )
    print(
        json.dumps(
            {
                "query_seconds": row["query_seconds"],
                "devices": devices,
                "processes": processes,
            },
            indent=2,
        )
    )
    for dev in ["1", "2"]:
        x = query(["npu-smi", "info", "-t", "usages", "-i", dev, "-c", "0"])
        v = parse_usages(x["stdout"])
        assert "HBM Bandwidth Usage Rate(%)" in v and "Aicore Usage Rate(%)" in v
        print(dev, x["query_seconds"], v)
