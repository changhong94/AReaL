"""Resolve independent baseline and research sources without fallback."""

import subprocess
from pathlib import Path


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def resolve(profile, arms):
    shared = Path(profile["repo"]).resolve()
    configured = profile.get("arm_repos", {})
    result = {}
    for arm in arms:
        path = Path(
            configured.get(arm, shared)
            if arm == "latest"
            else configured.get(arm, profile.get("baseline_repo", ""))
        )
        if arm != "latest" and not str(
            configured.get(arm, profile.get("baseline_repo", ""))
        ):
            raise ValueError(
                f"{arm} requires an explicit independent baseline_repo; shared-source fallback is forbidden"
            )
        path = path.resolve()
        if arm != "latest":
            if path == shared or git(path, "rev-parse", "HEAD") == git(
                shared, "rev-parse", "HEAD"
            ):
                raise ValueError(
                    "Baseline must be independent of the shared research source"
                )
            if git(path, "status", "--porcelain", "--untracked-files=all"):
                raise ValueError(f"Baseline checkout must be clean: {path}")
        result[arm] = path
    return result
