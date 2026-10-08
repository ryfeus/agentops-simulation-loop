"""Run one Harbor-backed Qwen LoRA GRPO step for Phase 8B."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

from billing_harbor_env import BillingHarborEnv

MODEL_ID = "Qwen/Qwen3-0.6B"


class TrainingFailure(RuntimeError):
    """Preserve the stable runtime phase for compact SSM evidence."""

    def __init__(self, phase: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.phase = phase
        self.cause = cause


def finite_metrics(state: Any) -> dict[str, float]:
    """Extract only finite numerical metrics from the trainer state."""

    result: dict[str, float] = {}
    for item in getattr(state, "log_history", []):
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    result[key] = float(value)
    return result


def build_training_config(*, output: Path, vllm_gpu_memory_utilization: float) -> Any:
    """Build the pinned TRL Harbor configuration without loading model weights."""

    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=str(output),
        model_init_kwargs={"dtype": "bfloat16"},
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        max_steps=1,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1,
        report_to="none",
        push_to_hub=False,
        bf16=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        num_generations=2,
        max_completion_length=256,
        max_tool_calling_iterations=4,
    )


def run(*, dataset_root: Path, output: Path, vllm_gpu_memory_utilization: float) -> dict[str, Any]:
    """Train exactly one optimizer step through Harbor's environment contract."""

    phase = "HARBOR_IMPORT"
    try:
        from peft import LoraConfig
        from trl import GRPOTrainer
        from trl.experimental.harbor import HarborSpec

        phase = "HARBOR_TASK_INVALID"
        spec = HarborSpec(str(dataset_root), agent=BillingHarborEnv, environment_type="docker")
        if len(spec.train_dataset) != 1:
            raise RuntimeError("Phase 8B requires exactly one Harbor task")
        phase = "TRL_HARBOR_INIT"
        config = build_training_config(
            output=output, vllm_gpu_memory_utilization=vllm_gpu_memory_utilization
        )
        trainer = GRPOTrainer(
            model=MODEL_ID,
            args=config,
            train_dataset=spec.train_dataset,
            environment_factory=spec.environment_factory,
            reward_funcs=spec.reward_funcs,
            peft_config=LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules="all-linear",
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        phase = "TRL_HARBOR_ROLLOUT"
        train_result = trainer.train()
        phase = "GRPO_BACKWARD"
        metrics = finite_metrics(trainer.state)
        global_step = int(getattr(trainer.state, "global_step", 0))
        if global_step < 1:
            raise RuntimeError("GRPO did not reach optimizer step 1")
        tool_calls = metrics.get("tools/call_frequency", 0.0)
        if tool_calls <= 0:
            raise RuntimeError("MODEL_NO_TOOL_CALL: Harbor tool call frequency was zero")
        output.mkdir(parents=True, exist_ok=True)
        phase = "CHECKPOINT_SAVE"
        trainer.save_model(str(output / "adapter"))
    except Exception as exc:
        raise TrainingFailure(phase, exc) from exc
    return {
        "schema_version": "1",
        "status": "PASS",
        "model_id": MODEL_ID,
        "task_id": "paid-refund-direct",
        "max_steps": 1,
        "num_generations": 2,
        "max_completion_length": 256,
        "max_tool_calling_iterations": 4,
        "global_step": global_step,
        "tool_call_frequency": tool_calls,
        "tool_failure_frequency": metrics.get("tools/failure_frequency"),
        "reward": metrics.get("reward"),
        "reward_std": metrics.get("reward_std"),
        "finite_metrics": metrics,
        "train_metrics": {
            key: value
            for key, value in train_result.metrics.items()
            if isinstance(value, (int, float))
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.30)
    args = parser.parse_args(argv)
    try:
        result = run(
            dataset_root=args.dataset_root,
            output=args.output,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
    except TrainingFailure as exc:
        result = {
            "schema_version": "1",
            "status": "FAIL",
            "failure_phase": exc.phase,
            "error_type": type(exc.cause).__name__,
            "error": str(exc.cause),
            "traceback": traceback.format_exc(),
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
