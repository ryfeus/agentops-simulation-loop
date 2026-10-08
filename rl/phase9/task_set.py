"""Explicit, hash-pinned Phase 9 task-set manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSet:
    """Validated operator selection over a full Phase 8C execution suite."""

    name: str
    execution_suite_sha256: str
    source_phase8c_runs: tuple[str, ...]
    training_tasks: tuple[dict[str, Any], ...]
    regression_tasks: tuple[dict[str, Any], ...]
    value: dict[str, Any]

    @property
    def training_ids(self) -> set[str]:
        return {str(item["task_id"]) for item in self.training_tasks}

    @property
    def regression_ids(self) -> set[str]:
        return {str(item["task_id"]) for item in self.regression_tasks}

    @property
    def evaluation_ids(self) -> set[str]:
        return self.training_ids | self.regression_ids


def _task_entries(value: dict[str, Any], key: str) -> tuple[dict[str, Any], ...]:
    entries = value.get(key)
    if not isinstance(entries, list):
        raise ValueError(f"task set {key} must be a list")
    if not all(isinstance(item, dict) for item in entries):
        raise ValueError(f"task set {key} entries must be objects")
    return tuple(entries)


def _validate_entry(entry: dict[str, Any], suite_tasks: dict[str, Any]) -> None:
    task_id = entry.get("task_id")
    if not isinstance(task_id, str) or task_id not in suite_tasks:
        raise ValueError(f"task set references an unknown task: {task_id!r}")
    metadata = suite_tasks[task_id]
    for key in ("archetype", "difficulty"):
        if entry.get(key) != metadata.get(key):
            raise ValueError(f"task set {task_id} {key} does not match execution suite")
    if not isinstance(entry.get("selection_reason"), str) or not entry["selection_reason"].strip():
        raise ValueError(f"task set {task_id} requires a selection_reason")


def load_task_set(path: Path, suite: dict[str, Any]) -> TaskSet:
    """Load an explicit task selection and verify immutable suite lineage."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Phase 9 task set: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "1":
        raise ValueError("Phase 9 task set has an unsupported schema")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Phase 9 task set requires a name")
    expected_sha = suite.get("execution_suite_sha256")
    if value.get("execution_suite_sha256") != expected_sha:
        raise ValueError("Phase 9 task set execution suite SHA does not match derived suite")
    sources = value.get("source_phase8c_runs")
    if (
        not isinstance(sources, list)
        or not sources
        or not all(isinstance(item, str) and item for item in sources)
    ):
        raise ValueError("Phase 9 task set requires non-empty source_phase8c_runs")
    training, regression = (
        _task_entries(value, "training_tasks"),
        _task_entries(value, "regression_tasks"),
    )
    if not 1 <= len(training) <= 10:
        raise ValueError("Phase 9 task set must contain 1-10 training tasks")
    if len(regression) > 5:
        raise ValueError("Phase 9 task set may contain at most five regression tasks")
    suite_tasks = suite.get("tasks")
    if not isinstance(suite_tasks, dict):
        raise ValueError("execution suite has no task metadata")
    for entry in (*training, *regression):
        _validate_entry(entry, suite_tasks)
    training_ids = [str(item["task_id"]) for item in training]
    regression_ids = [str(item["task_id"]) for item in regression]
    if len(set(training_ids)) != len(training_ids) or len(set(regression_ids)) != len(
        regression_ids
    ):
        raise ValueError("Phase 9 task set contains duplicate task IDs")
    if set(training_ids) & set(regression_ids):
        raise ValueError("Phase 9 training and regression task IDs must not overlap")
    for entry in training:
        attempts, passes, rate = (
            entry.get("baseline_attempts"),
            entry.get("baseline_passes"),
            entry.get("baseline_pass_rate"),
        )
        if (
            not isinstance(attempts, int)
            or attempts < 1
            or not isinstance(passes, int)
            or not 0 <= passes <= attempts
        ):
            raise ValueError("Phase 9 training baseline counts are invalid")
        if not isinstance(rate, (int, float)) or float(rate) != passes / attempts:
            raise ValueError("Phase 9 training baseline pass rate is invalid")
    return TaskSet(
        name=name,
        execution_suite_sha256=str(expected_sha),
        source_phase8c_runs=tuple(sources),
        training_tasks=training,
        regression_tasks=regression,
        value=value,
    )
