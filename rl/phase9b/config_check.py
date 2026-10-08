"""Locked, no-Docker/no-CUDA Phase 9b configuration and provenance gate."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from rl.phase8c.evaluate_baseline import FROZEN_EVALUATION_SETTINGS, build_evaluation_config
from rl.phase8c.execution_suite import derive_execution_suite
from rl.phase9.task_set import load_task_set

from .adapter import validate_persisted_adapter
from .evaluate import ATTEMPTS_PER_PASS, PASSES


def run() -> dict[str, object]:
    # Runtime suite derivation remains exercised by the controller before upload.
    # Validate the pinned GPU arguments on CPU-only developer hosts without
    # changing the actual evaluation configuration or touching CUDA.
    with patch("transformers.training_args.is_torch_bf16_gpu_available", return_value=True):
        config = build_evaluation_config(
            output=Path(".tmp-phase9b"), attempts_per_pass=4, seed=20260923
        )
    assert ATTEMPTS_PER_PASS == 4 and PASSES == 4
    assert config.num_generations_eval == 4 and config.per_device_eval_batch_size == 4
    assert FROZEN_EVALUATION_SETTINGS["max_completion_length"] == 256
    value: dict[str, object] = {
        "status": "PASS",
        "passes": PASSES,
        "attempts_per_pass": ATTEMPTS_PER_PASS,
    }
    source_run = os.environ.get("PHASE9B_SOURCE_RUN", "").strip()
    if source_run:
        root = Path(__file__).resolve().parents[2]
        source_dir = root / ".rl-smoke" / "phase9" / "runs" / source_run
        with tempfile.TemporaryDirectory(prefix="phase9b-config-") as temporary:
            suite = derive_execution_suite(Path(temporary) / "suite").manifest
        adapter = validate_persisted_adapter(
            source_dir,
            source_run=source_run,
            execution_suite_sha256=str(suite["execution_suite_sha256"]),
        )
        task_set = load_task_set(adapter.task_set_path, suite)
        assert len(task_set.training_ids) == 4 and len(task_set.regression_ids) == 1
        value.update(
            {
                "source_run": source_run,
                "source_revision": adapter.source_revision,
                "training_task_set_sha256": adapter.task_set_sha256,
                "execution_suite_sha256": adapter.execution_suite_sha256,
            }
        )
    return value


if __name__ == "__main__":
    import json

    print(json.dumps(run(), sort_keys=True))
