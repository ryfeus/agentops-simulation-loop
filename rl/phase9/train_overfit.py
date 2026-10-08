"""Run the tiny intentional Phase 9 Harbor GRPO overfit experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

from .report import build_reports
from .task_set import load_task_set
from .training_config import MODEL_ID, build_training_config, config_evidence
from .training_reward import reset_group_counter, reset_stage, set_stage, training_reward

# This is the fixed Phase 8B/8C policy-visible billing surface.  Keep this
# contract local to Phase 9: importing the executable Phase 8C evaluator would
# also pull its reward implementation into the sealed remote payload solely to
# access this immutable three-item set.
EXPECTED_TOOLS = {"get_invoice", "refund_invoice", "escalate_dispute"}


class TrainingFailure(RuntimeError):
    """Stable runtime phase retained in compact remote evidence."""

    def __init__(self, phase: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.phase, self.cause = phase, cause


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite(value: Any) -> dict[str, float]:
    return {
        str(key): float(number)
        for key, number in value.items()
        if isinstance(number, (int, float)) and math.isfinite(float(number))
    }


def _adapter_snapshot(model: Any) -> dict[str, Any]:

    tensors = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not tensors:
        raise RuntimeError("fresh LoRA model has no trainable adapter parameters")
    return tensors


def _adapter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    import torch

    if before.keys() != after.keys():
        raise RuntimeError("trainable LoRA parameter names changed during training")
    initial_sq = final_sq = delta_sq = 0.0
    changed_tensors = 0
    scalar_parameters = 0
    for name in sorted(before):
        initial, final = before[name], after[name]
        initial_sq += float(torch.sum(initial * initial))
        final_sq += float(torch.sum(final * final))
        difference = final - initial
        delta_sq += float(torch.sum(difference * difference))
        scalar_parameters += int(initial.numel())
        changed_tensors += int(not torch.equal(initial, final))
    return {
        "schema_version": "2",
        "trainable_tensor_count": len(before),
        "trainable_parameter_count": scalar_parameters,
        "changed_tensor_count": changed_tensors,
        "initial_l2_norm": initial_sq**0.5,
        "final_l2_norm": final_sq**0.5,
        "delta_l2_norm": delta_sq**0.5,
    }


class _MetricCallback:
    """A light callback that persists all finite native TRL train logs."""

    def __init__(self, output: Path) -> None:
        self.output = output

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any
    ) -> Any:
        from .training_reward import _stage

        if _stage.get() != "train" or not logs:
            return control
        finite = _finite(logs)
        if finite:
            finite["global_step"] = float(getattr(state, "global_step", 0))
            with (self.output / "training-metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(finite, sort_keys=True) + "\n")
        return control


def _run_stage(trainer: Any, stage: str, output: Path, method: str) -> Any:
    tokens = set_stage(stage, output)
    try:
        return getattr(trainer, method)()
    finally:
        reset_stage(tokens)


def run(
    *,
    dataset_root: Path,
    output: Path,
    task_set_path: Path,
    max_steps: int = 20,
    learning_rate: float = 1e-5,
    seed: int = 20260922,
) -> dict[str, Any]:
    """Evaluate, train, save, and re-evaluate fresh zero-init LoRA exactly once."""

    phase = "HARBOR_IMPORT"
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import TrainerCallback
        from trl import GRPOTrainer
        from trl.experimental.harbor import HarborSpec

        from rl.phase8c.execution_suite import load_execution_suite

        from .training_env import TrainingBillingHarborEnv

        phase = "TASK_SET_INVALID"
        suite = load_execution_suite(dataset_root / "execution-suite.json")
        task_set = load_task_set(task_set_path, suite)
        import inspect

        tools = {
            name
            for name, member in TrainingBillingHarborEnv.__mro__[2].__dict__.items()
            if not name.startswith("_") and inspect.isfunction(member)
        }
        if tools != EXPECTED_TOOLS:
            raise RuntimeError("Phase 9 model-visible tool surface differs from Phase 8B")
        phase = "HARBOR_TASK_INVALID"
        spec = HarborSpec(
            str(dataset_root), agent=TrainingBillingHarborEnv, environment_type="docker"
        )
        rows = [spec.train_dataset[index] for index in range(len(spec.train_dataset))]
        all_rows = [
            {
                **row,
                "task_id": Path(str(row["task_dir"])).name,
                **suite["tasks"][Path(str(row["task_dir"])).name],
            }
            for row in rows
        ]
        train_rows = [row for row in all_rows if row["task_id"] in task_set.training_ids]
        eval_rows = [row for row in all_rows if row["task_id"] in task_set.evaluation_ids]
        if {row["task_id"] for row in train_rows} != task_set.training_ids:
            raise RuntimeError("Phase 9 train dataset does not match selected training task IDs")
        if {row["task_id"] for row in eval_rows} != task_set.evaluation_ids:
            raise RuntimeError("Phase 9 eval dataset does not match selected task IDs")
        if {row["task_id"] for row in train_rows} & task_set.regression_ids:
            raise RuntimeError("Phase 9 regression task leaked into train dataset")
        train_dataset, eval_dataset = Dataset.from_list(train_rows), Dataset.from_list(eval_rows)
        config = build_training_config(
            output=output, max_steps=max_steps, learning_rate=learning_rate, seed=seed
        )
        training_config = config_evidence(
            max_steps=max_steps, learning_rate=learning_rate, seed=seed
        )
        task_set_bytes = task_set_path.read_bytes()
        task_set_sha256 = hashlib.sha256(task_set_bytes).hexdigest()
        training_config.update(
            {
                "execution_suite_sha256": suite["execution_suite_sha256"],
                "task_set_name": task_set.name,
                "training_task_ids": sorted(task_set.training_ids),
                "regression_task_ids": sorted(task_set.regression_ids),
                "training_task_set_sha256": task_set_sha256,
            }
        )
        _write(output / "training-config.json", training_config)
        # Preserve the exact bytes selected by the operator for Phase 9b,
        # rather than serializing an equivalent JSON object with new whitespace.
        (output / "training-task-set.json").write_bytes(task_set_bytes)
        _write(output / "effective-trl-config.json", _finite(vars(config)))
        phase = "TRL_HARBOR_INIT"
        callback_type = type("Phase9MetricCallback", (_MetricCallback, TrainerCallback), {})
        trainer = GRPOTrainer(
            model=MODEL_ID,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            environment_factory=spec.environment_factory,
            reward_funcs=training_reward,
            peft_config=LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules="all-linear",
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
            callbacks=[callback_type(output)],
        )
        before_adapter = _adapter_snapshot(trainer.model)
        reset_group_counter()
        phase = "PRE_EVALUATION"
        before_metrics = _finite(_run_stage(trainer, "before", output, "evaluate"))
        phase = "TRL_HARBOR_ROLLOUT"
        train_result = _run_stage(trainer, "train", output, "train")
        global_step = int(getattr(trainer.state, "global_step", 0))
        if global_step != max_steps:
            raise RuntimeError(f"GRPO reached {global_step} steps; expected {max_steps}")
        phase = "GRPO_BACKWARD"
        after_adapter = _adapter_snapshot(trainer.model)
        delta = _adapter_delta(before_adapter, after_adapter)
        if delta["delta_l2_norm"] == 0.0:
            raise RuntimeError("GRPO updated optimizer steps but did not change LoRA weights")
        phase = "CHECKPOINT_SAVE"
        adapter = output / "adapter" / "final"
        trainer.save_model(str(adapter))
        if not adapter.is_dir() or not any(adapter.iterdir()):
            raise RuntimeError("final LoRA adapter was not saved")
        _write(output / "adapter-delta.json", delta)
        phase = "POST_EVALUATION"
        after_metrics = _finite(_run_stage(trainer, "after", output, "evaluate"))
        phase = "REPORTING"
        reports = build_reports(
            output=output,
            training_ids=task_set.training_ids,
            regression_ids=task_set.regression_ids,
        )
        groups = reports["groups"]
        metrics_file = output / "training-metrics.jsonl"
        if (
            not groups
            or not metrics_file.is_file()
            or not metrics_file.read_text(encoding="utf-8").strip()
        ):
            raise RuntimeError("Phase 9 did not capture reward-group or train-metric evidence")
    except Exception as exc:
        raise TrainingFailure(phase, exc) from exc
    training_before, training_after = (
        reports["training_micro_before"],
        reports["training_micro_after"],
    )
    improved = training_after > training_before and reports["training_tasks_improved"] >= 2
    group_variance = sum(bool(group["has_reward_variance"]) for group in groups)
    summary = {
        "schema_version": "1",
        "phase": "9",
        "status": "PASS",
        "model_id": MODEL_ID,
        "execution_suite_sha256": suite["execution_suite_sha256"],
        "task_set_name": task_set.name,
        "training_task_set_sha256": task_set_sha256,
        "training_task_count": len(task_set.training_ids),
        "regression_task_count": len(task_set.regression_ids),
        "configured_max_steps": max_steps,
        "global_step": global_step,
        "training_reward_groups": len(groups),
        "groups_with_reward_variance": group_variance,
        "groups_without_reward_variance": len(groups) - group_variance,
        "fraction_groups_with_reward_variance": group_variance / len(groups),
        "adapter_delta_l2_norm": delta["delta_l2_norm"],
        "train_pass_rate_before": training_before,
        "train_pass_rate_after": training_after,
        "train_pass_rate_delta": training_after - training_before,
        "training_tasks_improved": reports["training_tasks_improved"],
        "training_tasks_unchanged": reports["training_tasks_unchanged"],
        "training_tasks_worsened": reports["training_tasks_worsened"],
        "training_execution_valid": True,
        "behavioral_improvement_observed": improved and group_variance > 0,
        "strong_overfit_signal": improved
        and group_variance > 0
        and training_after - training_before >= 0.20,
        # This is intentionally advisory: this tiny selected-task experiment is
        # not a generalization or significance claim.
        "phase9_success_candidate": improved and group_variance > 0,
        "catastrophic_regression_detected": reports["catastrophic_regression_detected"],
        "human_review_required": True,
        "no_grpo_signal": group_variance == 0,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "train_metrics": _finite(getattr(train_result, "metrics", {})),
        "wall_time_seconds": time.perf_counter() - started,
    }
    _write(output / "training-result.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-set", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args(argv)
    try:
        result = run(
            dataset_root=args.dataset_root,
            output=args.output,
            task_set_path=args.task_set,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    except TrainingFailure as exc:
        _write(
            args.output / "training-result.json",
            {
                "schema_version": "1",
                "phase": "9",
                "status": "FAIL",
                "failure_phase": exc.phase,
                "error_type": type(exc.cause).__name__,
                "error": str(exc.cause),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
