from unittest.mock import patch

import arm_sources
import pytest


def test_baseline_never_falls_back_to_shared(tmp_path):
    with pytest.raises(ValueError, match="explicit independent"):
        arm_sources.resolve({"repo": str(tmp_path / "shared")}, ["fixed"])


def test_resolve_separates_baseline_and_shared(tmp_path):
    research, upstream = tmp_path / "shared", tmp_path / "upstream"
    profile = {"repo": str(research), "baseline_repo": str(upstream)}

    def git(path, *args):
        return (
            ""
            if args[0] == "status"
            else ("upstream" if path == upstream else "research")
        )

    with patch.object(arm_sources, "git", side_effect=git):
        result = arm_sources.resolve(profile, ["fixed", "colocated", "latest"])
    assert result == {"fixed": upstream, "colocated": upstream, "latest": research}


def test_dirty_baseline_rejected(tmp_path):
    def git(path, *args):
        return " M areal/trainer.py" if args[0] == "status" else str(path)

    with (
        patch.object(arm_sources, "git", side_effect=git),
        pytest.raises(ValueError, match="clean"),
    ):
        arm_sources.resolve(
            {"repo": str(tmp_path / "shared"), "baseline_repo": str(tmp_path / "base")},
            ["colocated"],
        )


def test_same_commit_rejected_even_in_another_directory(tmp_path):
    with (
        patch.object(arm_sources, "git", return_value="same"),
        pytest.raises(ValueError, match="independent"),
    ):
        arm_sources.resolve(
            {"repo": str(tmp_path / "shared"), "baseline_repo": str(tmp_path / "base")},
            ["fixed"],
        )
