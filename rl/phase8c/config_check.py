"""Cheap locked-runtime contract check for the Phase 8C baseline."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from unittest.mock import patch

from .baseline_env import BaselineBillingHarborEnv
from .evaluate_baseline import EXPECTED_TOOLS, build_evaluation_config
from .execution_suite import derive_execution_suite, load_execution_suite


def _declared_tools() -> set[str]:
    return {
        name
        for name, member in BaselineBillingHarborEnv.__mro__[1].__dict__.items()
        if not name.startswith("_") and inspect.isfunction(member)
    }


def main() -> int:
    from trl.experimental.harbor import HarborSpec

    if _declared_tools() != EXPECTED_TOOLS:
        raise RuntimeError(f"unexpected Phase 8C tool surface: {sorted(_declared_tools())}")
    with tempfile.TemporaryDirectory(prefix="phase8c-config-") as temporary:
        root = Path(temporary)
        suite = derive_execution_suite(root / "suite")
        loaded = load_execution_suite(root / "suite" / "execution-suite.json")
        if suite.manifest != loaded or loaded["task_count"] != 25:
            raise RuntimeError("Phase 8C suite derivation is not deterministic or complete")
        spec = HarborSpec(
            str(root / "suite"), agent=BaselineBillingHarborEnv, environment_type="docker"
        )
        if len(spec.train_dataset) != 25 or not callable(spec.environment_factory):
            raise RuntimeError("Phase 8C HarborSpec did not load the full suite")
        # Check the exact GPU settings on CPU-only hosts without initializing CUDA.
        with patch("transformers.training_args.is_torch_bf16_gpu_available", return_value=True):
            config = build_evaluation_config(
                output=root / "output", attempts_per_pass=4, seed=20260920
            )
        if not (
            config.use_vllm
            and config.vllm_mode == "colocate"
            and config.vllm_tensor_parallel_size == 1
            and config.num_generations_eval == 4
            and config.per_device_eval_batch_size == 4
            and config.max_completion_length == 256
            and config.max_tool_calling_iterations == 4
            and config.model_init_kwargs == {"dtype": "bfloat16"}
        ):
            raise RuntimeError("Phase 8C frozen evaluation configuration is invalid")
    print("Phase 8C locked frozen baseline configuration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
