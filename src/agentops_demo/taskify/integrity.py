"""Canonical integrity primitives for taskify evidence and rendered Harbor tasks."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agentops_demo.contracts.scenario import Scenario
from agentops_demo.harbor.provenance import is_sha256


def scenario_sha256(scenario: Scenario) -> str:
    """Return the Phase 5 canonical Scenario digest."""

    payload = json.dumps(
        scenario.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def harbor_task_sha256(task: Path) -> str:
    """Hash every rendered task file using a stable path-and-content framing."""

    if not task.is_dir():
        raise ValueError(f"Harbor task is not a directory: {task}")
    files = sorted(path for path in task.rglob("*") if path.is_file() or path.is_symlink())
    digest = hashlib.sha256()
    for path in files:
        if path.is_symlink():
            raise ValueError(f"Harbor task may not contain symlinks: {path}")
        relative = path.relative_to(task).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


_REQUIRED_HARBOR_TASK_FILES = (
    "task.toml",
    "instruction.md",
    "scenario.yaml",
    "environment/Dockerfile",
    "environment/requirements.txt",
    "environment/seed.sql",
    "solution/solve.sh",
    "tests/Dockerfile",
    "tests/test.sh",
    "tests/verify.py",
    "tests/scenario.json",
)


def validate_harbor_task(task: Path) -> None:
    """Require the credential-free structural contract of a rendered Harbor task."""

    if not task.is_dir():
        raise ValueError(f"Harbor task is not a directory: {task}")
    missing = [
        relative for relative in _REQUIRED_HARBOR_TASK_FILES if not (task / relative).is_file()
    ]
    if missing:
        raise ValueError(f"Harbor task is missing required files: {', '.join(missing)}")
    if not (task / "instruction.md").read_text(encoding="utf-8").strip():
        raise ValueError("Harbor task instruction.md must be non-empty")
    try:
        with (task / "task.toml").open("rb") as config:
            tomllib.load(config)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Harbor task task.toml is invalid: {exc}") from exc
    # The tree hash rejects symlinks and verifies every rendered file is readable.
    harbor_task_sha256(task)


def harbor_suite_sha256(task_identities: Iterable[tuple[str, str, str]]) -> str:
    """Hash sorted Scenario and Harbor-task identities for one rendered suite."""

    identities = sorted(task_identities)
    task_ids = [task_id for task_id, _, _ in identities]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Harbor suite contains duplicate task IDs")
    for _, scenario_digest, task_digest in identities:
        if not is_sha256(scenario_digest) or not is_sha256(task_digest):
            raise ValueError("Harbor suite task identities must be SHA-256 digests")
    payload = json.dumps(identities, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def manifest_scenario_sha256(path: Path, scenario: Scenario) -> str:
    """Validate the taskify artifact manifest and return its Scenario digest."""

    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read taskify manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1":
        raise ValueError("taskify manifest has an unsupported schema")
    scenario_data = payload.get("scenario")
    if (
        payload.get("scenario_id") != scenario.id
        or not isinstance(scenario_data, dict)
        or scenario_data.get("path") != "scenario.yaml"
        or not is_sha256(scenario_data.get("sha256"))
    ):
        raise ValueError("taskify manifest scenario identity is invalid")
    digest = scenario_sha256(scenario)
    if scenario_data["sha256"] != digest:
        raise ValueError("taskify manifest Scenario SHA-256 does not match scenario.yaml")
    return digest
