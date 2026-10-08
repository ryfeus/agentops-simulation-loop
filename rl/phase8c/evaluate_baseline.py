"""Run the canonical Phase 8C frozen-policy evaluation through TRL."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

from .baseline_reward import baseline_reward
from .execution_suite import load_execution_suite
from .report import build_reports

MODEL_ID = "Qwen/Qwen3-0.6B"
EVALUATION_MAX_STEPS = 25
ATTEMPTS_PER_PASS = 4
EXPECTED_TOOLS = {"get_invoice", "refund_invoice", "escalate_dispute"}
FROZEN_EVALUATION_SETTINGS = {
    "model_init_kwargs": {"dtype": "bfloat16"},
    "per_device_eval_batch_size": 4,
    "max_completion_length": 256,
    "max_tool_calling_iterations": 4,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": None,
    "repetition_penalty": 1.0,
    "use_vllm": True,
    "vllm_mode": "colocate",
    "vllm_tensor_parallel_size": 1,
}
RETRYABLE_FAILURES = frozenset(
    {
        "DOCKER_UNAVAILABLE",
        "HARBOR_IMPORT",
        "HARBOR_SANDBOX_START",
        "HARNESS_SETUP",
        "BILLING_BRIDGE",
        "HARBOR_VERIFIER",
        "TRL_HARBOR_INIT",
        "TRL_HARBOR_ROLLOUT",
        "VLLM_OOM",
    }
)


class BaselineFailure(RuntimeError):
    """Stable phase attribution for remote evidence and controller classification."""

    def __init__(self, phase: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.phase, self.cause = phase, cause


def classify_infrastructure_failure(error: BaseException | str) -> str | None:
    text = str(error).lower()
    checks = (
        ("DOCKER_UNAVAILABLE", ("docker daemon", "docker_unavailable", "cannot connect to docker")),
        ("HARBOR_IMPORT", ("no module named 'harbor'", "harbor import")),
        ("HARBOR_SANDBOX_START", ("sandbox", "environment start", "docker build")),
        ("HARNESS_SETUP", ("phase8b_wheel_path", "sandbox billing wheel")),
        ("BILLING_BRIDGE", ("sandbox_billing_bridge", "bridge process")),
        ("HARBOR_VERIFIER", ("rewardfilenotfounderror", "reward file", "verifier")),
        ("VLLM_OOM", ("out of memory", "cuda oom")),
        ("TRL_HARBOR_ROLLOUT", ("grpotrainer", "tool calling")),
    )
    for phase, markers in checks:
        if any(marker in text for marker in markers):
            return phase
    return None


def build_evaluation_config(
    *, output: Path, attempts_per_pass: int, seed: int, vllm_gpu_memory_utilization: float = 0.30
) -> Any:
    """Return the pinned native-TRL frozen evaluation configuration."""

    from trl import GRPOConfig

    if attempts_per_pass != ATTEMPTS_PER_PASS:
        raise ValueError(f"Phase 8C requires attempts_per_pass={ATTEMPTS_PER_PASS}")
    return GRPOConfig(
        output_dir=str(output),
        # TRL requires a positive horizon when the Harbor environment supplies
        # prompts through reset(), including evaluate()-only runs.  This is an
        # evaluation bound, not permission to call train().
        max_steps=EVALUATION_MAX_STEPS,
        model_init_kwargs=FROZEN_EVALUATION_SETTINGS["model_init_kwargs"],
        per_device_eval_batch_size=FROZEN_EVALUATION_SETTINGS["per_device_eval_batch_size"],
        do_eval=True,
        report_to="none",
        push_to_hub=False,
        bf16=True,
        save_strategy="no",
        use_vllm=True,
        vllm_mode=FROZEN_EVALUATION_SETTINGS["vllm_mode"],
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        num_generations=attempts_per_pass,
        num_generations_eval=attempts_per_pass,
        max_completion_length=FROZEN_EVALUATION_SETTINGS["max_completion_length"],
        max_tool_calling_iterations=FROZEN_EVALUATION_SETTINGS["max_tool_calling_iterations"],
        temperature=FROZEN_EVALUATION_SETTINGS["temperature"],
        top_p=FROZEN_EVALUATION_SETTINGS["top_p"],
        top_k=FROZEN_EVALUATION_SETTINGS["top_k"],
        min_p=FROZEN_EVALUATION_SETTINGS["min_p"],
        repetition_penalty=FROZEN_EVALUATION_SETTINGS["repetition_penalty"],
        seed=seed,
        log_completions=True,
    )


def _declared_tools() -> set[str]:
    import inspect

    from .baseline_env import BaselineBillingHarborEnv

    return {
        name
        for name, member in BaselineBillingHarborEnv.__mro__[1].__dict__.items()
        if not name.startswith("_") and inspect.isfunction(member)
    }


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def _load_captured_rollouts(
    path: Path, *, run_id: str, evaluation_pass: int, attempts_per_pass: int
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError("TRL evaluation did not emit Phase 8C rollout evidence")
    counters: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        task_id = str(record.get("task_id", ""))
        pass_local_attempt = counters.get(task_id, 0)
        counters[task_id] = pass_local_attempt + 1
        record.update(
            {
                "run_id": run_id,
                "attempt": evaluation_pass * attempts_per_pass + pass_local_attempt,
                "evaluation_pass": evaluation_pass,
                "pass_local_attempt": pass_local_attempt,
            }
        )
        record.setdefault("retry_count", 0)
        record.setdefault("prior_infrastructure_failures", [])
        records.append(record)
    if any(value != attempts_per_pass for value in counters.values()):
        raise ValueError("TRL evaluation did not yield the required attempts per task")
    return records


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def run(
    *,
    dataset_root: Path,
    output: Path,
    run_id: str,
    attempts_per_pass: int = ATTEMPTS_PER_PASS,
    passes: int = 1,
    seed: int = 20260920,
    infra_retries: int = 2,
    task_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen policy once using TRL; this function never calls train."""

    if attempts_per_pass != ATTEMPTS_PER_PASS:
        raise ValueError(f"PHASE8C_ATTEMPTS_PER_PASS must equal {ATTEMPTS_PER_PASS}")
    if passes < 1:
        raise ValueError("PHASE8C_PASSES must be positive")
    if infra_retries < 0:
        raise ValueError("PHASE8C_INFRA_RETRIES cannot be negative")
    suite = load_execution_suite(dataset_root / "execution-suite.json")
    if _declared_tools() != EXPECTED_TOOLS:
        raise ValueError("Phase 8C model-visible tool surface differs from Phase 8B")
    phase = "HARBOR_IMPORT"
    started = time.perf_counter()
    capture = output / "rollouts.raw.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    os.environ["PHASE8C_ROLLOUT_CAPTURE"] = str(capture)
    os.environ["PHASE8C_INFRA_RETRIES"] = str(infra_retries)
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOTrainer
        from trl.experimental.harbor import HarborSpec

        from .baseline_env import BaselineBillingHarborEnv

        phase = "HARBOR_TASK_INVALID"
        spec = HarborSpec(
            str(dataset_root), agent=BaselineBillingHarborEnv, environment_type="docker"
        )
        if len(spec.train_dataset) != int(suite["task_count"]):
            raise RuntimeError("HarborSpec task count does not match execution suite")
        tasks = suite["tasks"]
        rows = [spec.train_dataset[index] for index in range(len(spec.train_dataset))]
        all_rows = [
            {
                **row,
                "task_id": Path(str(row["task_dir"])).name,
                **tasks[Path(str(row["task_dir"])).name],
            }
            for row in rows
        ]
        available_task_ids = {str(row["task_id"]) for row in all_rows}
        selected_task_ids = available_task_ids if task_ids is None else task_ids
        unknown = selected_task_ids - available_task_ids
        if unknown:
            raise ValueError(f"unknown Phase 8C task IDs: {sorted(unknown)}")
        eval_dataset = Dataset.from_list(
            [row for row in all_rows if row["task_id"] in selected_task_ids]
        )
        run_type = (
            "canonical" if len(selected_task_ids) == 25 and passes == 1 else "targeted_followup"
        )
        _write(
            output / "baseline-config.json",
            {
                "schema_version": "1",
                "run_type": run_type,
                "model_id": MODEL_ID,
                "lora": {
                    "r": 8,
                    "lora_alpha": 16,
                    "target_modules": "all-linear",
                    "lora_dropout": 0.0,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                },
                "sampling": {
                    key: FROZEN_EVALUATION_SETTINGS[key]
                    for key in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")
                }
                | {"seed": seed},
                "evaluation": {
                    "attempts_per_pass": attempts_per_pass,
                    "passes": passes,
                    "per_device_eval_batch_size": FROZEN_EVALUATION_SETTINGS[
                        "per_device_eval_batch_size"
                    ],
                    "max_completion_length": FROZEN_EVALUATION_SETTINGS["max_completion_length"],
                    "max_tool_calling_iterations": FROZEN_EVALUATION_SETTINGS[
                        "max_tool_calling_iterations"
                    ],
                },
                "vllm": {
                    "mode": FROZEN_EVALUATION_SETTINGS["vllm_mode"],
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.30,
                },
                "infra_retries": infra_retries,
                "suite_task_count": 25,
                "evaluated_task_count": len(selected_task_ids),
                "evaluated_task_ids": sorted(selected_task_ids),
            },
        )
        phase = "TRL_HARBOR_INIT"
        rollouts, metrics = [], {}
        for evaluation_pass in range(passes):
            pass_capture = output / f"rollouts.pass-{evaluation_pass}.raw.jsonl"
            os.environ["PHASE8C_ROLLOUT_CAPTURE"] = str(pass_capture)
            config = build_evaluation_config(
                output=output, attempts_per_pass=attempts_per_pass, seed=seed + evaluation_pass
            )
            trainer = GRPOTrainer(
                model=MODEL_ID,
                args=config,
                eval_dataset=eval_dataset,
                environment_factory=spec.environment_factory,
                reward_funcs=baseline_reward,
                peft_config=LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules="all-linear",
                    lora_dropout=0.0,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
            if (
                int(getattr(trainer.state, "global_step", -1)) != 0
                or getattr(trainer, "optimizer", None) is not None
            ):
                raise RuntimeError("frozen evaluation constructed training state")
            phase = "TRL_HARBOR_ROLLOUT"
            metrics = trainer.evaluate()
            if (
                int(getattr(trainer.state, "global_step", -1)) != 0
                or getattr(trainer, "optimizer", None) is not None
            ):
                raise RuntimeError("frozen evaluation performed training or created an optimizer")
            rollouts.extend(
                _load_captured_rollouts(
                    pass_capture,
                    run_id=run_id,
                    evaluation_pass=evaluation_pass,
                    attempts_per_pass=attempts_per_pass,
                )
            )
        if list(output.glob("checkpoint-*")):
            raise RuntimeError("frozen evaluation emitted a checkpoint")
        _write_jsonl(output / "rollouts.jsonl", rollouts)
        retry_rows = [row for row in rollouts if int(row.get("retry_count", 0)) > 0]
        _write_jsonl(output / "rollout-retries.jsonl", retry_rows)
        reports = build_reports(
            rollouts, attempts_per_task=attempts_per_pass * passes, destination=output
        )
    except Exception as exc:
        raise BaselineFailure(phase, exc) from exc
    finally:
        os.environ.pop("PHASE8C_ROLLOUT_CAPTURE", None)
    wall_time = time.perf_counter() - started
    task_summaries = reports["tasks"]
    result = {
        "schema_version": "1",
        "run_type": run_type,
        "status": "PASS",
        "model_id": MODEL_ID,
        "training_performed": False,
        "global_step": 0,
        "optimizer_created": False,
        "checkpoint_emitted": False,
        "suite_task_count": suite["task_count"],
        "evaluated_task_count": len(selected_task_ids),
        "evaluated_task_ids": sorted(selected_task_ids),
        "task_count": len(selected_task_ids),
        "attempts_per_pass": attempts_per_pass,
        "passes": passes,
        "attempts_per_task": attempts_per_pass * passes,
        "expected_rollouts": len(selected_task_ids) * attempts_per_pass * passes,
        "valid_rollouts": len(rollouts),
        "infrastructure_invalid_rollouts": 0,
        "infrastructure_retry_count": sum(int(row.get("retry_count", 0)) for row in rollouts),
        "base_suite_sha256": suite["base_suite_sha256"],
        "execution_suite_sha256": suite["execution_suite_sha256"],
        "overall_pass_rate": sum(float(row["reward"]) for row in rollouts) / len(rollouts),
        "observed_always_fail_tasks": sum(
            task["observed_bucket"] == "observed_always_fail" for task in task_summaries
        ),
        "observed_mixed_tasks": sum(
            task["observed_bucket"] == "observed_mixed" for task in task_summaries
        ),
        "observed_always_pass_tasks": sum(
            task["observed_bucket"] == "observed_always_pass" for task in task_summaries
        ),
        "eval_metrics": _finite_metrics(metrics),
        "wall_time_seconds": wall_time,
        "rollouts_per_second": len(rollouts) / wall_time if wall_time else 0.0,
        "phase9_candidate_threshold_met": reports["candidates"]["phase9_candidate_threshold_met"],
        "phase9_candidate_selection_requires_human_review": True,
    }
    _write(output / "eval-metrics.json", result["eval_metrics"])
    _write(output / "evaluation-result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempts-per-pass", type=int, default=ATTEMPTS_PER_PASS)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--infra-retries", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        result = run(
            dataset_root=args.dataset_root,
            output=args.output,
            run_id=args.run_id,
            attempts_per_pass=args.attempts_per_pass,
            passes=args.passes,
            seed=args.seed,
            infra_retries=args.infra_retries,
            task_ids={item for item in args.task_ids.split(",") if item} or None,
        )
    except BaselineFailure as exc:
        _write(
            args.output / "evaluation-result.json",
            {
                "schema_version": "1",
                "status": "FAIL",
                "failure_phase": exc.phase,
                "failure_class": classify_infrastructure_failure(exc.cause),
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
