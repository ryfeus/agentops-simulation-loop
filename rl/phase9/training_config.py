"""Pinned native TRL configuration for one tiny Phase 9 overfit experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-0.6B"
NUM_GENERATIONS = 4
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
MAX_COMPLETION_LENGTH = 256
MAX_TOOL_CALLING_ITERATIONS = 4
VLLM_GPU_MEMORY_UTILIZATION = 0.30


def build_training_config(*, output: Path, max_steps: int, learning_rate: float, seed: int) -> Any:
    """Build the locked TRL 1.13 GRPO configuration without loading model weights."""

    from trl import GRPOConfig

    if max_steps < 1:
        raise ValueError("PHASE9_MAX_STEPS must be positive")
    if learning_rate <= 0:
        raise ValueError("PHASE9_LEARNING_RATE must be positive")
    return GRPOConfig(
        output_dir=str(output / "checkpoints"),
        model_init_kwargs={"dtype": "bfloat16"},
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=1,
        do_eval=True,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=5,
        save_total_limit=5,
        report_to="none",
        push_to_hub=False,
        bf16=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        num_generations=NUM_GENERATIONS,
        num_generations_eval=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_tool_calling_iterations=MAX_TOOL_CALLING_ITERATIONS,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=None,
        repetition_penalty=1.0,
        seed=seed,
        log_completions=True,
    )


def config_evidence(*, max_steps: int, learning_rate: float, seed: int) -> dict[str, Any]:
    """Stable public provenance record, independent of optional TRL defaults."""

    return {
        "schema_version": "1",
        "model_id": MODEL_ID,
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": "all-linear",
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "training": {
            "num_generations": NUM_GENERATIONS,
            "per_device_train_batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": 1,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "max_tool_calling_iterations": MAX_TOOL_CALLING_ITERATIONS,
        },
        "evaluation": {
            "num_generations_eval": NUM_GENERATIONS,
            "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        },
        "sampling": {
            "seed": seed,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": None,
            "repetition_penalty": 1.0,
        },
        "vllm": {
            "mode": "colocate",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        },
    }
