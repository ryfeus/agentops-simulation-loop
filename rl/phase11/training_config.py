"""Canonical Phase 11 GRPO configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl.phase11.experiment import load_experiment


def lora_config() -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def training_config(output: Path, *, cpu_check: bool = False) -> Any:
    from trl import GRPOConfig

    exp = load_experiment()
    return GRPOConfig(
        output_dir=str(output / "checkpoints"),
        model_init_kwargs={"dtype": "bfloat16"},
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=exp["training"]["num_generations"],
        num_generations_eval=4,
        learning_rate=exp["training"]["learning_rate"],
        max_steps=exp["training"]["steps"],
        logging_steps=1,
        save_strategy="steps",
        save_steps=160,
        save_total_limit=3,
        do_eval=False,
        eval_strategy="no",
        report_to="none",
        push_to_hub=False,
        bf16=not cpu_check,
        use_cpu=cpu_check,
        max_completion_length=256,
        max_tool_calling_iterations=4,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=None,
        repetition_penalty=1.0,
        use_vllm=not cpu_check,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.30,
        seed=exp["training"]["seed"],
        log_completions=True,
    )
