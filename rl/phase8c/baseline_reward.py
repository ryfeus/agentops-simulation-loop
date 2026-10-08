"""Unshaped reward/evidence function passed to TRL's native evaluation path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _values(value: Any, count: int) -> list[Any]:
    return value if isinstance(value, list) and len(value) == count else [value] * count


def _append_records(records: list[dict[str, Any]]) -> None:
    destination = os.environ.get("PHASE8C_ROLLOUT_CAPTURE", "")
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def baseline_reward(
    *,
    environments: list[Any] | None = None,
    completions: list[Any] | None = None,
    log_extra: Any | None = None,
    **_: Any,
) -> list[float]:
    """Return the Harbor scalar unchanged and retain rollout-level semantic evidence."""

    environments = environments or []
    completions = completions or []
    task_ids = _values(_.get("task_id"), len(environments))
    archetypes = _values(_.get("archetype"), len(environments))
    difficulties = _values(_.get("difficulty"), len(environments))
    scenario_hashes = _values(_.get("scenario_sha256"), len(environments))
    base_hashes = _values(_.get("base_task_sha256"), len(environments))
    execution_hashes = _values(_.get("execution_task_sha256"), len(environments))
    rewards: list[float] = []
    records: list[dict[str, Any]] = []
    for index, environment in enumerate(environments):
        reward = float(environment.reward)
        evidence = environment._baseline_evidence()
        completion = completions[index] if index < len(completions) else ""
        records.append(
            {
                **evidence,
                "task_id": str(task_ids[index]),
                "archetype": str(archetypes[index]),
                "difficulty": str(difficulties[index]),
                "scenario_sha256": str(scenario_hashes[index]),
                "base_task_sha256": str(base_hashes[index]),
                "execution_task_sha256": str(execution_hashes[index]),
                "reward": reward,
                "completion": _as_text(completion),
            }
        )
        rewards.append(reward)
    _append_records(records)
    if callable(log_extra):
        log_extra("phase8c_rollout", records)
    return rewards
