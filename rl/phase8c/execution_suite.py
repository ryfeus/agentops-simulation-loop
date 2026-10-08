"""Deterministically derive the Phase 8C shared-verifier billing suite."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rl.phase8b.execution_task import ExecutionTask

EXECUTION_PROFILE = "trl-harbor-shared-verifier"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TASKS = REPOSITORY_ROOT / "benchmarks" / "billing" / "tasks"


@dataclass(frozen=True)
class ExecutionSuite:
    """Execution-only copies and their immutable benchmark lineage."""

    root: Path
    manifest: dict[str, Any]
    tasks: dict[str, ExecutionTask]

    @property
    def execution_suite_sha256(self) -> str:
        return str(self.manifest["execution_suite_sha256"])


def _canonical_task_ids(root: Path) -> set[str]:
    if not root.is_dir():
        raise ValueError(f"canonical billing task snapshot is missing: {root}")
    task_ids = {path.name for path in root.iterdir() if path.is_dir() and not path.is_symlink()}
    if not task_ids:
        raise ValueError("canonical billing task snapshot is empty")
    return task_ids


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def derive_execution_suite(
    output: Path,
    *,
    catalog: Path | None = None,
    canonical_tasks: Path = CANONICAL_TASKS,
    task_ids: set[str] | None = None,
) -> ExecutionSuite:
    """Derive a complete, hash-addressed shared-verifier execution suite.

    ``output`` must not exist.  Keeping it write-once prevents an operator from
    accidentally mixing execution copies from two benchmark revisions.
    """

    from agentops_demo.benchmark.catalog import DEFAULT_CATALOG, load_catalog
    from agentops_demo.taskify.integrity import (
        harbor_suite_sha256,
        scenario_sha256,
        validate_harbor_task,
    )
    from rl.phase8b.execution_task import derive_execution_task

    entries = load_catalog(catalog or DEFAULT_CATALOG)
    catalog_ids = {entry.scenario.id for entry in entries}
    rendered_ids = _canonical_task_ids(canonical_tasks)
    if catalog_ids != rendered_ids:
        raise ValueError(
            "billing scenario IDs and checked-in rendered task IDs differ: "
            f"missing={sorted(catalog_ids - rendered_ids)}, "
            f"extra={sorted(rendered_ids - catalog_ids)}"
        )
    selected = catalog_ids if task_ids is None else set(task_ids)
    unknown = selected - catalog_ids
    if unknown:
        raise ValueError(f"unknown Phase 8C task IDs: {sorted(unknown)}")
    if not selected:
        raise ValueError("Phase 8C execution suite must contain at least one task")
    if output.exists():
        raise FileExistsError(f"Phase 8C execution suite destination exists: {output}")

    output.mkdir(parents=True)
    task_root = output / "tasks"
    task_root.mkdir()
    derived: dict[str, ExecutionTask] = {}
    metadata: dict[str, dict[str, str]] = {}
    try:
        for entry in sorted(entries, key=lambda item: item.scenario.id):
            task_id = entry.scenario.id
            if task_id not in selected:
                continue
            base = canonical_tasks / task_id
            validate_harbor_task(base)
            execution = derive_execution_task(base, task_root / task_id)
            derived[task_id] = execution
            metadata[task_id] = {
                "archetype": entry.scenario.benchmark.archetype,
                "difficulty": entry.scenario.benchmark.difficulty,
                "scenario_sha256": scenario_sha256(entry.scenario),
                "base_task_sha256": execution.base_task_sha256,
                "execution_task_sha256": execution.execution_task_sha256,
            }
        base_suite = harbor_suite_sha256(
            (task_id, details["scenario_sha256"], details["base_task_sha256"])
            for task_id, details in metadata.items()
        )
        execution_suite = harbor_suite_sha256(
            (task_id, details["scenario_sha256"], details["execution_task_sha256"])
            for task_id, details in metadata.items()
        )
        manifest: dict[str, Any] = {
            "schema_version": "1",
            "benchmark": "billing",
            "task_count": len(metadata),
            "task_ids": sorted(metadata),
            "base_suite_sha256": base_suite,
            "execution_suite_sha256": execution_suite,
            "execution_profile": EXECUTION_PROFILE,
            "tasks": {task_id: metadata[task_id] for task_id in sorted(metadata)},
        }
        _write_json(output / "execution-suite.json", manifest)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return ExecutionSuite(root=output, manifest=manifest, tasks=derived)


def load_execution_suite(path: Path) -> dict[str, Any]:
    """Load and minimally validate a previously derived suite manifest."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read execution suite manifest: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "1":
        raise ValueError("execution suite manifest has an unsupported schema")
    tasks = value.get("tasks")
    if not isinstance(tasks, dict) or value.get("task_count") != len(tasks):
        raise ValueError("execution suite manifest task count is invalid")
    if value.get("execution_profile") != EXECUTION_PROFILE:
        raise ValueError("execution suite profile is not the TRL shared-verifier profile")
    for required in ("base_suite_sha256", "execution_suite_sha256"):
        if not isinstance(value.get(required), str) or len(value[required]) != 64:
            raise ValueError(f"execution suite {required} is invalid")
    return value
