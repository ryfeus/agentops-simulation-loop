"""Phase 10 subset manifests; never package unseen executable tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rl.phase8c.execution_suite import load_execution_suite


def subset_suite(source: Path, destination: Path, task_ids: set[str]) -> dict[str, Any]:
    """Copy only permitted tasks and write a subset HarborSpec manifest."""
    import shutil

    full = load_execution_suite(source / "execution-suite.json")
    if not task_ids or not task_ids <= set(full["task_ids"]):
        raise ValueError("subset includes missing or unknown execution tasks")
    if destination.exists():
        raise FileExistsError(destination)
    (destination / "tasks").mkdir(parents=True)
    for task_id in sorted(task_ids):
        shutil.copytree(source / "tasks" / task_id, destination / "tasks" / task_id, symlinks=False)
    manifest = {
        **full,
        "task_count": len(task_ids),
        "task_ids": sorted(task_ids),
        "tasks": {key: full["tasks"][key] for key in sorted(task_ids)},
        "source_execution_suite_sha256": full["execution_suite_sha256"],
    }
    # This is a transport manifest, not a claim that the subset hashes to the full suite SHA.
    (destination / "execution-suite.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_subset(destination, task_ids)
    return manifest


def validate_subset(root: Path, expected: set[str]) -> None:
    suite = load_execution_suite(root / "execution-suite.json")
    tasks = root / "tasks"
    actual = {path.name for path in tasks.iterdir() if path.is_dir()}
    if actual != expected or set(suite["task_ids"]) != expected or set(suite["tasks"]) != expected:
        raise ValueError("payload executable task IDs differ from allowed set")
    if any(path.is_symlink() for path in tasks.rglob("*")):
        raise ValueError("payload must not contain task symlinks")
