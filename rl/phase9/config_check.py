"""Locked, CPU-only Phase 9 contract gate; it never loads Qwen or Docker."""

from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path

from rl.phase8c.execution_suite import derive_execution_suite, load_execution_suite

from .task_set import load_task_set
from .train_overfit import EXPECTED_TOOLS
from .training_config import (
    EVAL_BATCH_SIZE,
    MAX_COMPLETION_LENGTH,
    MAX_TOOL_CALLING_ITERATIONS,
    MODEL_ID,
    NUM_GENERATIONS,
    TRAIN_BATCH_SIZE,
    build_training_config,
)
from .training_env import TrainingBillingHarborEnv


def _task_set_path() -> Path:
    configured = os.environ.get("PHASE9_TASK_SET", "rl/phase9/task_sets/tiny-overfit-v1.json")
    return Path(configured)


def run() -> None:
    with tempfile.TemporaryDirectory(prefix="phase9-config-") as temporary:
        root = Path(temporary)
        derived = derive_execution_suite(root / "suite")
        suite = load_execution_suite(derived.root / "execution-suite.json")
        selected = load_task_set(_task_set_path(), suite)
        config = build_training_config(
            output=root / "output", max_steps=20, learning_rate=1e-5, seed=20260922
        )
    tools = {
        name
        for name, member in TrainingBillingHarborEnv.__mro__[2].__dict__.items()
        if not name.startswith("_") and inspect.isfunction(member)
    }
    assert tools == EXPECTED_TOOLS
    assert selected.training_ids.isdisjoint(selected.regression_ids)
    assert MODEL_ID == "Qwen/Qwen3-0.6B"
    assert NUM_GENERATIONS == config.num_generations == 4
    assert TRAIN_BATCH_SIZE == config.per_device_train_batch_size == 4
    assert EVAL_BATCH_SIZE == config.per_device_eval_batch_size == 4
    assert config.num_generations_eval == 4
    assert MAX_COMPLETION_LENGTH == config.max_completion_length == 256
    assert MAX_TOOL_CALLING_ITERATIONS == config.max_tool_calling_iterations == 4
    assert config.max_steps == 20 and config.learning_rate == 1e-5
    assert (
        config.use_vllm and config.vllm_mode == "colocate" and config.vllm_tensor_parallel_size == 1
    )


if __name__ == "__main__":
    run()
