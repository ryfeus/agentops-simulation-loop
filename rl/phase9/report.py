"""Deterministic Phase 9 before/after and technical-validity reports."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _task_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    return grouped


def _summary(rows: list[dict[str, Any]], task_ids: set[str]) -> tuple[dict[str, Any], float, float]:
    grouped = _task_rows(rows)
    values: list[dict[str, Any]] = []
    rates: list[float] = []
    all_rewards: list[float] = []
    for task_id in sorted(task_ids):
        task_rows = grouped.get(task_id, [])
        rewards = [float(row["reward"]) for row in task_rows]
        rate = sum(rewards) / len(rewards) if rewards else 0.0
        rates.append(rate)
        all_rewards.extend(rewards)
        sequences = Counter(
            " → ".join(
                str(call.get("name", ""))
                for call in row.get("tool_calls", [])
                if isinstance(call, dict)
            )
            for row in task_rows
        )
        values.append(
            {
                "task_id": task_id,
                "archetype": task_rows[0].get("archetype") if task_rows else None,
                "difficulty": task_rows[0].get("difficulty") if task_rows else None,
                "attempts": len(rewards),
                "passes": int(sum(rewards)),
                "pass_rate": rate,
                "tool_sequence_histogram": dict(sorted(sequences.items())),
                "tool_call_rate": sum(int(row.get("tool_call_count", 0)) > 0 for row in task_rows)
                / len(task_rows)
                if task_rows
                else 0.0,
                "verifier_component_rates": {
                    key: sum(
                        float(row.get("verifier_components", {}).get(key, 0.0)) for row in task_rows
                    )
                    / len(task_rows)
                    for key in sorted(
                        {
                            key
                            for row in task_rows
                            for key in row.get("verifier_components", {})
                            if isinstance(row.get("verifier_components", {}).get(key), (int, float))
                        }
                    )
                },
            }
        )
    return (
        {item["task_id"]: item for item in values},
        sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        sum(rates) / len(rates) if rates else 0.0,
    )


def build_reports(
    *, output: Path, training_ids: set[str], regression_ids: set[str]
) -> dict[str, Any]:
    """Write deterministic reports from captured native-TRL rollout evidence."""

    before, after, train = (
        _read_jsonl(output / f"rollouts-{stage}.jsonl") for stage in ("before", "after", "train")
    )
    before_tasks, _, _ = _summary(before, training_ids | regression_ids)
    after_tasks, _, _ = _summary(after, training_ids | regression_ids)
    _write(output / "before-task-summary.json", list(before_tasks.values()))
    _write(output / "after-task-summary.json", list(after_tasks.values()))
    entries: list[dict[str, Any]] = []
    improved = unchanged = worsened = 0
    for task_id in sorted(training_ids | regression_ids):
        first, final = before_tasks[task_id], after_tasks[task_id]
        delta = float(final["pass_rate"]) - float(first["pass_rate"])
        if task_id in training_ids:
            improved += delta > 0
            unchanged += delta == 0
            worsened += delta < 0
        entries.append(
            {
                "task_id": task_id,
                "archetype": first["archetype"],
                "is_training_task": task_id in training_ids,
                "is_regression_control": task_id in regression_ids,
                "before_attempts": first["attempts"],
                "before_passes": first["passes"],
                "before_pass_rate": first["pass_rate"],
                "after_attempts": final["attempts"],
                "after_passes": final["passes"],
                "after_pass_rate": final["pass_rate"],
                "absolute_change": delta,
            }
        )
    groups = _read_jsonl(output / "reward-groups.jsonl")
    exposure: dict[str, dict[str, int]] = defaultdict(
        lambda: {"groups": 0, "rollouts": 0, "mixed_groups": 0}
    )
    for group in groups:
        target = exposure[str(group["task_id"])]
        target["groups"] += 1
        target["rollouts"] += len(group["rewards"])
        target["mixed_groups"] += int(group["has_reward_variance"])
    regression = [entry for entry in entries if entry["is_regression_control"]]
    catastrophic = any(
        float(entry["before_pass_rate"]) >= 0.75 and float(entry["after_pass_rate"]) <= 0.25
        for entry in regression
    )
    training_before = (
        sum(float(row["reward"]) for row in before if row["task_id"] in training_ids)
        / sum(row["task_id"] in training_ids for row in before)
        if any(row["task_id"] in training_ids for row in before)
        else 0.0
    )
    training_after = (
        sum(float(row["reward"]) for row in after if row["task_id"] in training_ids)
        / sum(row["task_id"] in training_ids for row in after)
        if any(row["task_id"] in training_ids for row in after)
        else 0.0
    )
    _write(
        output / "before-after-summary.json",
        {
            "training_micro_before": training_before,
            "training_micro_after": training_after,
            "training_macro_before": sum(
                float(before_tasks[task_id]["pass_rate"]) for task_id in training_ids
            )
            / len(training_ids),
            "training_macro_after": sum(
                float(after_tasks[task_id]["pass_rate"]) for task_id in training_ids
            )
            / len(training_ids),
            "tasks": entries,
            "training_tasks_improved": improved,
            "training_tasks_unchanged": unchanged,
            "training_tasks_worsened": worsened,
        },
    )
    _write(output / "regression-summary.json", regression)
    _write(output / "task-training-exposure.json", dict(sorted(exposure.items())))
    return {
        "before": before,
        "after": after,
        "train": train,
        "groups": groups,
        "entries": entries,
        "training_tasks_improved": improved,
        "training_tasks_unchanged": unchanged,
        "training_tasks_worsened": worsened,
        "training_micro_before": training_before,
        "training_micro_after": training_after,
        "catastrophic_regression_detected": catastrophic,
    }
