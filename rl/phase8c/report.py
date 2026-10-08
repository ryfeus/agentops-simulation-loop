"""Deterministic offline aggregation for Phase 8C rollout evidence."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wilson_interval(passes: int, attempts: int) -> list[float]:
    """95% Wilson interval, represented as stable JSON-friendly floats."""

    if attempts <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = passes / attempts
    denominator = 1 + z * z / attempts
    centre = (proportion + z * z / (2 * attempts)) / denominator
    variance = (proportion * (1 - proportion) + z * z / (4 * attempts)) / attempts
    delta = z * math.sqrt(variance) / denominator
    return [centre - delta, centre + delta]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _tool_sequence(row: dict[str, Any]) -> str:
    calls = row.get("tool_calls", [])
    if not isinstance(calls, list):
        return ""
    return " → ".join(str(call.get("name", "")) for call in calls if isinstance(call, dict))


def _task_summary(task_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    passes = sum(row["reward"] == 1.0 for row in rows)
    attempts = len(rows)
    components: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.get("verifier_components", {}).items():
            if isinstance(value, (int, float)):
                components[str(key)].append(float(value))
    tools = [int(row.get("tool_call_count", 0)) for row in rows]
    sequences = Counter(_tool_sequence(row) for row in rows)
    first = rows[0]
    if passes == 0:
        bucket = "observed_always_fail"
    elif passes == attempts:
        bucket = "observed_always_pass"
    else:
        bucket = "observed_mixed"
    return {
        "task_id": task_id,
        "archetype": first["archetype"],
        "difficulty": first["difficulty"],
        "attempts": attempts,
        "passes": passes,
        "pass_rate": passes / attempts if attempts else 0.0,
        "wilson_95": wilson_interval(passes, attempts),
        "observed_bucket": bucket,
        "mean_tool_calls": _mean(tools),
        "tool_call_rate": _mean(value > 0 for value in tools),
        "component_pass_rates": {key: _mean(values) for key, values in sorted(components.items())},
        "mean_reset_ms": _mean(float(row.get("environment_reset_ms", 0)) for row in rows),
        "mean_tool_execution_ms": _mean(float(row.get("tool_execution_ms", 0)) for row in rows),
        "mean_verifier_ms": _mean(float(row.get("verifier_ms", 0)) for row in rows),
        "tool_sequence_histogram": dict(sorted(sequences.items())),
        "infrastructure_retry_count": sum(int(row.get("retry_count", 0)) for row in rows),
    }


def _slices(tasks: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        groups[str(task[key])].append(task)
    result: dict[str, dict[str, Any]] = {}
    for name, entries in sorted(groups.items()):
        attempts = sum(int(item["attempts"]) for item in entries)
        passes = sum(int(item["passes"]) for item in entries)
        result[name] = {
            "tasks": len(entries),
            "rollouts": attempts,
            "pass_rate": passes / attempts if attempts else 0.0,
            "tool_call_rate": _mean(float(item["tool_call_rate"]) for item in entries),
            "mean_tool_calls": _mean(float(item["mean_tool_calls"]) for item in entries),
            "mean_verifier_ms": _mean(float(item["mean_verifier_ms"]) for item in entries),
            "infrastructure_retries": sum(
                int(item["infrastructure_retry_count"]) for item in entries
            ),
        }
    return result


def build_reports(
    rollouts: list[dict[str, Any]], *, attempts_per_task: int, destination: Path
) -> dict[str, Any]:
    """Validate completed logical rows and write every deterministic report."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, int]] = set()
    for row in rollouts:
        if row.get("reward") not in (0.0, 1.0):
            raise ValueError("every final Phase 8C reward must be exactly 0.0 or 1.0")
        identity = (str(row.get("task_id", "")), int(row.get("attempt", -1)))
        if not identity[0] or identity[1] < 0 or identity in identities:
            raise ValueError("Phase 8C final logical attempt identities are invalid")
        identities.add(identity)
        grouped[identity[0]].append(row)
    tasks = [
        _task_summary(task_id, sorted(rows, key=lambda row: row["attempt"]))
        for task_id, rows in sorted(grouped.items())
    ]
    expected = len(tasks) * attempts_per_task
    if len(rollouts) != expected or any(task["attempts"] != attempts_per_task for task in tasks):
        raise ValueError("Phase 8C rollout coverage is incomplete")
    archetypes, difficulties = _slices(tasks, "archetype"), _slices(tasks, "difficulty")
    mixed = [task for task in tasks if task["observed_bucket"] == "observed_mixed"]
    candidates = {
        "schema_version": "1",
        "mixed_reward_candidates": mixed,
        "always_fail_but_valid_tool_use": [
            task
            for task in tasks
            if task["observed_bucket"] == "observed_always_fail" and task["tool_call_rate"] > 0
        ],
        "always_pass_regression_controls": [
            task for task in tasks if task["observed_bucket"] == "observed_always_pass"
        ],
        "mixed_task_count": len(mixed),
        "mixed_archetype_count": len({task["archetype"] for task in mixed}),
        "phase9_candidate_threshold_met": len(mixed) >= 5
        and len({task["archetype"] for task in mixed}) >= 2,
        "phase9_candidate_selection_requires_human_review": True,
    }
    destination.mkdir(parents=True, exist_ok=True)
    _write(destination / "task-summary.json", tasks)
    _write(destination / "archetype-summary.json", archetypes)
    _write(destination / "difficulty-summary.json", difficulties)
    _write(destination / "phase9-candidates.json", candidates)
    return {
        "tasks": tasks,
        "archetypes": archetypes,
        "difficulties": difficulties,
        "candidates": candidates,
    }
