"""Train canonical Phase 11 GRPO on the 160 train tasks only."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rl.phase8c.execution_suite import load_execution_suite
from rl.phase9.train_overfit import _adapter_delta, _adapter_snapshot, _MetricCallback
from rl.phase9.training_reward import reset_group_counter, reset_stage, set_stage, training_reward
from rl.phase11.selection import adapter_files
from rl.phase11.training_config import lora_config, training_config


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def validate_exposure(
    groups: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    train_ids: set[str],
    dev_ids: set[str],
    holdout_ids: set[str],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if len(groups) != 320 or len(rollouts) != 1280:
        raise ValueError("Phase 11 requires 320 complete groups and 1280 rollouts")
    if len(train_ids) != 160 or train_ids & (dev_ids | holdout_ids):
        raise ValueError("Phase 11 training split invalid")
    by_task = defaultdict(lambda: {"groups": 0, "rollouts": 0})
    rollout_ids = {row["rollout_id"] for row in rollouts}
    if len(rollout_ids) != len(rollouts):
        raise ValueError("duplicate training rollout ID")
    for index, group in enumerate(groups):
        task_id = group.get("task_id")
        ids = group.get("rollout_ids", [])
        rewards = group.get("rewards", [])
        if (
            task_id not in train_ids
            or group.get("group_index") != index
            or len(ids) != 4
            or len(set(ids)) != 4
            or any(rollout_id not in rollout_ids for rollout_id in ids)
            or len(rewards) != 4
            or any(reward not in (0.0, 1.0) for reward in rewards)
        ):
            raise ValueError("invalid Phase 11 training reward group")
        by_task[task_id]["groups"] += 1
        by_task[task_id]["rollouts"] += 4
        group.update(
            {
                key: metadata[task_id][key]
                for key in ("family_id", "prototype_id", "archetype", "difficulty")
            }
        )
        group["global_step"] = index + 1
        group["group_class"] = (
            "mixed"
            if group["has_reward_variance"]
            else ("all-one" if all(reward == 1.0 for reward in rewards) else "all-zero")
        )
    if set(by_task) != train_ids or any(
        value != {"groups": 2, "rollouts": 8} for value in by_task.values()
    ):
        raise ValueError("Phase 11 exact two-exposure invariant failed")
    groups_by_rollout = {
        rollout_id: group for group in groups for rollout_id in group["rollout_ids"]
    }
    for row in rollouts:
        task_id = row.get("task_id")
        group = groups_by_rollout.get(row.get("rollout_id"))
        if task_id not in train_ids or group is None or task_id != group["task_id"]:
            raise ValueError("unknown or cross-task Phase 11 training rollout")
        row.update(
            {
                key: metadata[task_id][key]
                for key in ("family_id", "prototype_id", "archetype", "difficulty")
            }
        )
        row["group_id"] = group["group_index"]
        row["global_step"] = group["global_step"]
    return {
        "schema_version": "1",
        "tasks": dict(sorted(by_task.items())),
        "group_count": len(groups),
        "rollout_count": len(rollouts),
        "dev_groups": 0,
        "holdout_groups": 0,
    }


def snapshot_checkpoint(checkpoint: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        source = checkpoint / name
        if not source.is_file():
            raise ValueError(f"checkpoint lacks {name}: {checkpoint}")
        shutil.copy2(source, destination / name)
    return adapter_files(destination)


def run(dataset_root: Path, metadata_path: Path, split_path: Path, output: Path) -> dict[str, Any]:
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOTrainer
    from trl.experimental.harbor import HarborSpec

    from rl.phase9.training_env import TrainingBillingHarborEnv

    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids, dev_ids, holdout_ids = (set(split[key]) for key in ("train", "dev", "holdout"))
    if set(metadata) != train_ids | dev_ids or len(train_ids) != 160 or len(dev_ids) != 20:
        raise ValueError("training payload metadata must contain exactly train and dev")
    suite = load_execution_suite(dataset_root / "execution-suite.json")
    if set(suite["task_ids"]) != set(metadata):
        raise ValueError("training executable payload contains unexpected tasks")
    spec = HarborSpec(str(dataset_root), agent=TrainingBillingHarborEnv, environment_type="docker")
    rows = []
    for row in spec.train_dataset:
        task_id = Path(str(row["task_dir"])).name
        if task_id in train_ids:
            rows.append({**row, "task_id": task_id, **suite["tasks"][task_id], **metadata[task_id]})
    if {row["task_id"] for row in rows} != train_ids or len(rows) != 160:
        raise ValueError("GRPO train dataset differs from 160 frozen train IDs")
    config = training_config(output)
    write_json(
        output / "training-config.json",
        {
            "training_task_ids": sorted(train_ids),
            "model_id": "Qwen/Qwen3-0.6B",
            "max_steps": 320,
            "num_generations": 4,
            "seed": 20260925,
            "learning_rate": 1e-5,
            "execution_suite_sha256": suite["source_execution_suite_sha256"],
        },
    )
    write_json(output / "effective-trl-config.json", config.to_dict())
    callback_type = type("Phase11MetricCallback", (_MetricCallback, TrainerCallback), {})
    trainer = GRPOTrainer(
        model="Qwen/Qwen3-0.6B",
        args=config,
        train_dataset=Dataset.from_list(rows),
        environment_factory=spec.environment_factory,
        reward_funcs=training_reward,
        peft_config=lora_config(),
        callbacks=[callback_type(output)],
    )
    before = _adapter_snapshot(trainer.model)
    reset_group_counter()
    tokens = set_stage("train", output)
    try:
        result = trainer.train()
    finally:
        reset_stage(tokens)
    step = int(trainer.state.global_step)
    after = _adapter_snapshot(trainer.model)
    delta = _adapter_delta(before, after)
    write_json(output / "adapter-delta.json", delta)
    groups = read_jsonl(output / "reward-groups.jsonl")
    rollouts = read_jsonl(output / "rollouts-train.jsonl")
    exposure = validate_exposure(groups, rollouts, train_ids, dev_ids, holdout_ids, metadata)
    write_jsonl(output / "reward-groups.jsonl", groups)
    write_jsonl(output / "rollouts-train.jsonl", rollouts)
    write_json(output / "task-training-exposure.json", exposure)
    manifests = {
        str(step): snapshot_checkpoint(
            output / "checkpoints" / f"checkpoint-{step}", output / "adapters" / f"step-{step}"
        )
        for step in (160, 320)
    }
    write_json(output / "adapter-manifest.json", manifests)
    metrics = read_jsonl(output / "training-metrics.jsonl")
    if (
        step != 320
        or not math.isfinite(delta["delta_l2_norm"])
        or delta["delta_l2_norm"] <= 0
        or delta["changed_tensor_count"] <= 0
        or not metrics
    ):
        raise ValueError("Phase 11 optimizer or adapter validity failed")
    counts = Counter(group["group_class"] for group in groups)
    by_dimension: dict[str, dict[str, dict[str, int]]] = {}
    seen_tasks: Counter[str] = Counter()
    for group in groups:
        seen_tasks[group["task_id"]] += 1
        group["exposure_index"] = seen_tasks[group["task_id"]]
    for dimension in ("family_id", "prototype_id", "archetype", "difficulty", "exposure_index"):
        buckets: dict[str, Counter[str]] = defaultdict(Counter)
        for group in groups:
            buckets[str(group[dimension])][str(group["group_class"])] += 1
        by_dimension[dimension] = {key: dict(value) for key, value in sorted(buckets.items())}
    write_jsonl(output / "reward-groups.jsonl", groups)
    write_json(
        output / "reward-group-summary.json",
        {
            "classes": dict(counts),
            "mixed_fraction": counts["mixed"] / len(groups),
            "by_dimension": by_dimension,
        },
    )
    summary = {
        "status": "PASS",
        "global_step": step,
        "reward_groups": len(groups),
        "training_rollouts": len(rollouts),
        "reward_group_classes": dict(counts),
        "adapter_delta_l2_norm": delta["delta_l2_norm"],
        "train_metrics": result.metrics,
        "training_execution_valid": True,
    }
    write_json(output / "training-result.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.metadata, args.split, args.output), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
