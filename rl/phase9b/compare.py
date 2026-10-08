"""Deterministic Phase 9b paired-policy comparison reports."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from rl.phase8c.report import wilson_interval


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(rows: list[dict[str, Any]], task_ids: set[str]) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for task_id in sorted(task_ids):
        items = [item for item in rows if item["task_id"] == task_id]
        passes = sum(float(item["reward"]) for item in items)
        sequences = Counter(
            " → ".join(
                str(call.get("name", ""))
                for call in item.get("tool_calls", [])
                if isinstance(call, dict)
            )
            for item in items
        )
        components = {
            key: sum(float(item.get("verifier_components", {}).get(key, 0.0)) for item in items)
            / len(items)
            for key in sorted(
                {key for item in items for key in item.get("verifier_components", {})}
            )
            if items
        }
        values.append(
            {
                "task_id": task_id,
                "attempts": len(items),
                "passes": int(passes),
                "pass_rate": passes / len(items),
                "wilson_95": wilson_interval(int(passes), len(items)),
                "tool_call_rate": sum(int(item.get("tool_call_count", 0)) > 0 for item in items)
                / len(items),
                "tool_sequence_histogram": dict(sorted(sequences.items())),
                "verifier_component_rates": components,
            }
        )
    rewards = [float(row["reward"]) for row in rows]
    return {
        "tasks": values,
        "micro_pass_rate": sum(rewards) / len(rewards),
        "macro_pass_rate": sum(item["pass_rate"] for item in values) / len(values),
        "micro_wilson_95": wilson_interval(int(sum(rewards)), len(rewards)),
    }


def build_comparison(
    *,
    baseline: list[dict[str, Any]],
    trained: list[dict[str, Any]],
    training_ids: set[str],
    control_id: str,
    destination: Path,
) -> dict[str, Any]:
    task_ids = training_ids | {control_id}
    before, after = _summary(baseline, task_ids), _summary(trained, task_ids)
    before_by, after_by = (
        {item["task_id"]: item for item in source["tasks"]} for source in (before, after)
    )
    tasks = []
    for task_id in sorted(task_ids):
        tasks.append(
            {
                "task_id": task_id,
                "is_training_task": task_id in training_ids,
                "is_selected_control": task_id == control_id,
                "baseline": before_by[task_id],
                "trained": after_by[task_id],
                "absolute_change": after_by[task_id]["pass_rate"] - before_by[task_id]["pass_rate"],
            }
        )
    train = [row for row in tasks if row["is_training_task"]]
    control = next(row for row in tasks if row["is_selected_control"])
    aggregate_before = sum(row["baseline"]["passes"] for row in train) / sum(
        row["baseline"]["attempts"] for row in train
    )
    aggregate_after = sum(row["trained"]["passes"] for row in train) / sum(
        row["trained"]["attempts"] for row in train
    )
    catastrophic = (
        control["baseline"]["pass_rate"] >= 0.75 and control["trained"]["pass_rate"] <= 0.25
    )
    supported = (
        aggregate_after > aggregate_before
        and sum(row["absolute_change"] > 0 for row in train) >= 2
        and not any(row["absolute_change"] <= -0.25 for row in train)
        and not catastrophic
    )
    value = {
        "schema_version": "1",
        "baseline": before,
        "trained": after,
        "tasks": tasks,
        "training_aggregate_baseline": aggregate_before,
        "training_aggregate_trained": aggregate_after,
        "training_aggregate_delta": aggregate_after - aggregate_before,
        "selected_control_delta": control["absolute_change"],
        "catastrophic_regression_detected": catastrophic,
        "behavioral_improvement_supported": supported,
        "human_review_required": True,
    }
    _write(destination / "comparison.json", value)
    return value
