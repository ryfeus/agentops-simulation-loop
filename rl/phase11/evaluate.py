"""Frozen native TRL evaluation with one process per vLLM pass."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from rl.phase8c.baseline_env import BaselineBillingHarborEnv
from rl.phase8c.baseline_reward import baseline_reward
from rl.phase8c.evaluate_baseline import (
    EXPECTED_TOOLS,
    _declared_tools,
    _finite_metrics,
    _load_captured_rollouts,
    build_evaluation_config,
)
from rl.phase8c.execution_suite import load_execution_suite
from rl.phase11.compare import summarize
from rl.phase11.selection import adapter_files
from rl.phase11.train import write_json, write_jsonl
from rl.phase11.training_config import lora_config


def worker(
    *,
    dataset_root: Path,
    metadata_path: Path,
    task_ids_path: Path,
    adapter: Path | None,
    output: Path,
    run_id: str,
    seed: int,
    pass_index: int,
) -> None:
    from datasets import Dataset
    from trl import GRPOTrainer
    from trl.experimental.harbor import HarborSpec

    if _declared_tools() != EXPECTED_TOOLS:
        raise ValueError("Phase 11 model-visible tool surface changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = set(json.loads(task_ids_path.read_text(encoding="utf-8")))
    suite = load_execution_suite(dataset_root / "execution-suite.json")
    if not selected or not selected <= set(suite["task_ids"]) or not selected <= set(metadata):
        raise ValueError("Phase 11 evaluation task selection invalid")
    spec = HarborSpec(str(dataset_root), agent=BaselineBillingHarborEnv, environment_type="docker")
    rows = []
    for row in spec.train_dataset:
        task_id = Path(str(row["task_dir"])).name
        if task_id in selected:
            rows.append({**row, "task_id": task_id, **suite["tasks"][task_id], **metadata[task_id]})
    if len(rows) != len(selected) or {row["task_id"] for row in rows} != selected:
        raise ValueError("Phase 11 evaluation dataset coverage mismatch")
    before = adapter_files(adapter) if adapter is not None else None
    model: Any = "Qwen/Qwen3-0.6B"
    extra: dict[str, Any] = {"peft_config": lora_config()}
    if adapter is not None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="bfloat16"),
            adapter,
            is_trainable=False,
        )
        extra = {}
    config = build_evaluation_config(output=output, attempts_per_pass=4, seed=seed + pass_index)
    capture = output / f"rollouts.pass-{pass_index}.raw.jsonl"
    os.environ["PHASE8C_ROLLOUT_CAPTURE"] = str(capture)
    trainer = GRPOTrainer(
        model=model,
        args=config,
        eval_dataset=Dataset.from_list(rows),
        environment_factory=spec.environment_factory,
        reward_funcs=baseline_reward,
        **extra,
    )
    if trainer.state.global_step != 0 or trainer.optimizer is not None:
        raise ValueError("Phase 11 evaluation initialized training state")
    metrics = trainer.evaluate()
    if trainer.state.global_step != 0 or trainer.optimizer is not None:
        raise ValueError("Phase 11 evaluation mutated training state")
    if before is not None and adapter_files(adapter) != before:
        raise ValueError("Phase 11 evaluation mutated adapter")
    if list(output.glob("checkpoint-*")):
        raise ValueError("Phase 11 evaluation emitted checkpoint")
    result = _load_captured_rollouts(
        capture, run_id=run_id, evaluation_pass=pass_index, attempts_per_pass=4
    )
    if {row["task_id"] for row in result} != selected:
        raise ValueError("Phase 11 evaluation reward coverage incomplete")
    for row in result:
        row.update(
            {
                key: metadata[row["task_id"]][key]
                for key in ("family_id", "prototype_id", "archetype", "difficulty")
            }
        )
        row["policy"] = "baseline" if adapter is None else "trained"
        row["rollout_id"] = f"{row['policy']}:{row['task_id']}:{row['attempt']:02d}"
    write_jsonl(output / f"rollouts.pass-{pass_index}.jsonl", result)
    write_json(
        output / f"pass-{pass_index}-result.json",
        {
            "evaluation_pass": pass_index,
            "seed": seed + pass_index,
            "global_step": 0,
            "optimizer_created": False,
            "training_performed": False,
            "checkpoint_emitted": False,
            "valid_rollouts": len(result),
            "eval_metrics": _finite_metrics(metrics),
        },
    )


def evaluate_policy(
    *,
    dataset_root: Path,
    metadata_path: Path,
    task_ids_path: Path,
    adapter: Path | None,
    passes: int,
    seed: int,
    output: Path,
    run_id: str,
) -> dict[str, Any]:
    if passes not in (1, 2):
        raise ValueError("Phase 11 supports one or two four-generation passes")
    selected = set(json.loads(task_ids_path.read_text(encoding="utf-8")))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    adapter_before = adapter_files(adapter) if adapter is not None else None
    all_rows = []
    for pass_index in range(passes):
        command = [
            sys.executable,
            "-m",
            "rl.phase11.evaluate",
            "--dataset-root",
            str(dataset_root),
            "--metadata",
            str(metadata_path),
            "--task-ids",
            str(task_ids_path),
            "--output",
            str(output),
            "--run-id",
            run_id,
            "--seed",
            str(seed),
            "--passes",
            str(passes),
            "--worker-pass",
            str(pass_index),
        ]
        if adapter is not None:
            command.extend(("--adapter", str(adapter)))
        subprocess.run(command, check=True)
        rows = [
            json.loads(line)
            for line in (output / f"rollouts.pass-{pass_index}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        all_rows.extend(rows)
    if adapter_before is not None and adapter_files(adapter) != adapter_before:
        raise ValueError("Phase 11 adapter changed across passes")
    write_jsonl(output / "rollouts.jsonl", all_rows)
    summary = summarize(all_rows, {key: metadata[key] for key in selected}, passes * 4)
    write_json(output / "task-summary.json", summary)
    proof = {
        "status": "PASS",
        "training_performed": False,
        "global_step": 0,
        "optimizer_created": False,
        "checkpoint_emitted": False,
        "attempts_per_task": passes * 4,
        "valid_rollouts": len(all_rows),
        "seed_schedule": [seed + i for i in range(passes)],
        "adapter_files_before": adapter_before,
        "adapter_files_after": adapter_files(adapter) if adapter is not None else None,
    }
    if len(all_rows) != len(selected) * passes * 4 or list(output.glob("checkpoint-*")):
        raise ValueError("Phase 11 policy evaluation incomplete or emitted checkpoint")
    write_json(output / "evaluation-result.json", proof)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--task-ids", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--passes", type=int, required=True)
    parser.add_argument("--worker-pass", type=int)
    args = parser.parse_args()
    if args.worker_pass is not None:
        worker(
            dataset_root=args.dataset_root,
            metadata_path=args.metadata,
            task_ids_path=args.task_ids,
            adapter=args.adapter,
            output=args.output,
            run_id=args.run_id,
            seed=args.seed,
            pass_index=args.worker_pass,
        )
    else:
        print(
            json.dumps(
                evaluate_policy(
                    dataset_root=args.dataset_root,
                    metadata_path=args.metadata,
                    task_ids_path=args.task_ids,
                    adapter=args.adapter,
                    passes=args.passes,
                    seed=args.seed,
                    output=args.output,
                    run_id=args.run_id,
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
