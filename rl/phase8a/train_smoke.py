"""Run a tiny deterministic LoRA GRPO smoke test with optional colocated vLLM."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPTS = (
    "Give a short answer: what is 6 * 7?",
    "Compute 7 * 6 and answer briefly.",
    "What integer equals six times seven?",
)


def answer_42_reward(completions: list[str] | list[list[dict[str, str]]], **_: Any) -> list[float]:
    """Reward only completions which contain the requested deterministic answer."""

    rewards: list[float] = []
    for completion in completions:
        if isinstance(completion, str):
            text = completion
        else:
            text = " ".join(str(message.get("content", "")) for message in completion)
        normalized = " ".join(text.lower().split())
        rewards.append(1.0 if "42" in normalized else 0.0)
    return rewards


def _finite_metrics(state: Any) -> dict[str, float]:
    history = getattr(state, "log_history", [])
    result: dict[str, float] = {}
    for item in history:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                result[key] = float(value)
    return result


class TrainingFailure(RuntimeError):
    """Preserve the phase that failed for compact remote evidence."""

    def __init__(self, phase: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.phase = phase
        self.cause = cause


def build_training_config(*, mode: str, output: Path, vllm_gpu_memory_utilization: float) -> Any:
    """Build the pinned TRL 1.13 configuration without loading model weights."""

    from trl import GRPOConfig

    use_vllm = mode == "vllm"
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
        use_vllm=use_vllm,
        num_generations=2,
        max_completion_length=32,
        vllm_mode="colocate" if use_vllm else "server",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
    )


def run(*, mode: str, output: Path, vllm_gpu_memory_utilization: float) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    use_vllm = mode == "vllm"
    phase = "CONFIG"
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOTrainer

        dataset = Dataset.from_dict({"prompt": list(PROMPTS) * 2})
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        config = build_training_config(
            mode=mode,
            output=output,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
        phase = "TRAINER_INIT"
        trainer = GRPOTrainer(
            model=MODEL_ID,
            reward_funcs=answer_42_reward,
            args=config,
            train_dataset=dataset,
            peft_config=peft_config,
        )
        phase = "TRAIN"
        train_result = trainer.train()
        phase = "CHECKPOINT_SAVE"
        adapter_dir = output / "adapter"
        trainer.save_model(str(adapter_dir))
        metrics = _finite_metrics(trainer.state)
        if not metrics:
            raise RuntimeError("trainer emitted no finite numeric metrics")
    except Exception as exc:
        raise TrainingFailure(phase, exc) from exc
    return {
        "schema_version": "1",
        "mode": mode,
        "status": "PASS",
        "model_id": MODEL_ID,
        "max_steps": 1,
        "num_generations": 2,
        "max_completion_length": 32,
        "per_device_train_batch_size": 2,
        "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization if use_vllm else None,
        "train_metrics": {
            key: value
            for key, value in train_result.metrics.items()
            if isinstance(value, (int, float))
        },
        "finite_metrics": metrics,
        "adapter_path": str(adapter_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "vllm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.30)
    args = parser.parse_args(argv)
    try:
        result = run(
            mode=args.mode,
            output=args.output,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
    except TrainingFailure as exc:  # JSON evidence must survive a failed remote command.
        result = {
            "schema_version": "1",
            "mode": args.mode,
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
