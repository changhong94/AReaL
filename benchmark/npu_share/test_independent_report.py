import json

import independent_report


def test_missing_colocated_events_preserve_real_comparators(tmp_path):
    arms = ["fixed", "colocated", "latest"]
    settings = {
        "arms": arms,
        "source_provenance": {
            a: {"head": "community" if a != "latest" else "research"} for a in arms
        },
    }
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    (tmp_path / "arm-status.json").write_text(
        json.dumps({a: {"status": "complete"} for a in arms})
    )
    for arm, duration in [("fixed", 100), ("colocated", None), ("latest", 90)]:
        (tmp_path / arm).mkdir()
        events = [{"event": "train.begin", "time": 1000}]
        if duration:
            events.append({"event": "train.end", "time": 1000 + duration})
        (tmp_path / arm / "phases.jsonl").write_text("\n".join(map(json.dumps, events)))
    independent_report.render(tmp_path)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["loop_seconds"] == {"fixed": 100, "latest": 90}
    assert result["reduction_percent"] == 10
    assert result["colocated_reduction_percent"] is None
    assert result["arm_status"]["colocated"]["status"] == "analysis_failed"
