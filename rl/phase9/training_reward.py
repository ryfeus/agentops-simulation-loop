"""Stage-scoped exact Harbor reward and rollout/group evidence capture."""

from __future__ import annotations

import contextvars
import json
from collections import Counter
from pathlib import Path
from typing import Any

_stage: contextvars.ContextVar[str] = contextvars.ContextVar("phase9_stage", default="")
_output: contextvars.ContextVar[Path | None] = contextvars.ContextVar("phase9_output", default=None)
_group_index: contextvars.ContextVar[int] = contextvars.ContextVar("phase9_group_index", default=0)
_stage_sequence: contextvars.ContextVar[int] = contextvars.ContextVar(
    "phase9_stage_sequence", default=0
)


def set_stage(
    stage: str, output: Path
) -> tuple[contextvars.Token[str], contextvars.Token[Path | None], contextvars.Token[int]]:
    """Enter a lifecycle stage without process-global filename coupling."""

    if stage not in {"before", "train", "after"}:
        raise ValueError(f"unsupported Phase 9 stage: {stage}")
    return _stage.set(stage), _output.set(output), _stage_sequence.set(0)


def reset_stage(
    tokens: tuple[contextvars.Token[str], contextvars.Token[Path | None], contextvars.Token[int]],
) -> None:
    """Leave a lifecycle stage."""

    _stage.reset(tokens[0])
    _output.reset(tokens[1])
    _stage_sequence.reset(tokens[2])


def reset_group_counter() -> None:
    """Start deterministic reward-group numbering for a fresh run."""

    _group_index.set(0)


def _values(value: Any, count: int) -> list[Any]:
    return value if isinstance(value, list) and len(value) == count else [value] * count


def _as_text(value: Any) -> str:
    return (
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    )


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def training_reward(
    *,
    environments: list[Any] | None = None,
    completions: list[Any] | None = None,
    log_extra: Any | None = None,
    **kwargs: Any,
) -> list[float]:
    """Return exact verifier rewards while retaining semantic rollout evidence."""

    stage, output = _stage.get(), _output.get()
    if not stage or output is None:
        raise RuntimeError("Phase 9 reward callback ran outside an explicit lifecycle stage")
    environments, completions = environments or [], completions or []
    count = len(environments)
    task_ids = _values(kwargs.get("task_id"), count)
    archetypes = _values(kwargs.get("archetype"), count)
    difficulties = _values(kwargs.get("difficulty"), count)
    records: list[dict[str, Any]] = []
    rewards: list[float] = []
    sequence = _stage_sequence.get()
    for index, environment in enumerate(environments):
        reward = float(environment.reward)
        evidence = environment._baseline_evidence()
        task_id = str(task_ids[index])
        records.append(
            {
                **evidence,
                "stage": stage,
                "task_id": task_id,
                "rollout_id": f"{stage}:{task_id}:{sequence:06d}",
                "archetype": str(archetypes[index]),
                "difficulty": str(difficulties[index]),
                "reward": reward,
                "completion": _as_text(completions[index] if index < len(completions) else ""),
            }
        )
        rewards.append(reward)
        sequence += 1
    _stage_sequence.set(sequence)
    _append_jsonl(output / f"rollouts-{stage}.jsonl", records)
    if stage == "train":
        by_task = Counter(str(record["task_id"]) for record in records)
        if any(size != 4 for size in by_task.values()):
            raise RuntimeError(
                "Phase 9 training reward callback did not contain complete four-rollout groups"
            )
        group_index = _group_index.get()
        groups: list[dict[str, Any]] = []
        for task_id in dict.fromkeys(str(record["task_id"]) for record in records):
            group = [record for record in records if record["task_id"] == task_id]
            group_rewards = [float(record["reward"]) for record in group]
            groups.append(
                {
                    "group_index": group_index,
                    "stage": stage,
                    "task_id": task_id,
                    "rewards": group_rewards,
                    "mean_reward": sum(group_rewards) / len(group_rewards),
                    "reward_std": (
                        sum(
                            (value - sum(group_rewards) / len(group_rewards)) ** 2
                            for value in group_rewards
                        )
                        / len(group_rewards)
                    )
                    ** 0.5,
                    "has_reward_variance": len(set(group_rewards)) > 1,
                    "rollout_ids": [str(record["rollout_id"]) for record in group],
                }
            )
            group_index += 1
        _group_index.set(group_index)
        _append_jsonl(output / "reward-groups.jsonl", groups)
    if callable(log_extra):
        log_extra("phase9_rollout", records)
    return rewards
