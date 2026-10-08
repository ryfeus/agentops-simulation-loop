"""Deterministic dev selection, eligibility, and immutable adapter locks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter_files(adapter: Path) -> dict[str, dict[str, Any]]:
    if not adapter.is_dir():
        raise ValueError("adapter directory missing")
    files = {}
    for path in sorted(adapter.rglob("*")):
        if path.is_symlink():
            raise ValueError("adapter contains symlink")
        if path.is_file():
            files[path.relative_to(adapter).as_posix()] = {
                "sha256": digest(path),
                "size": path.stat().st_size,
            }
    if "adapter_config.json" not in files or "adapter_model.safetensors" not in files:
        raise ValueError("adapter lacks config or safetensors weights")
    return files


def choose_checkpoint(candidates: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if set(candidates) != {160, 320}:
        raise ValueError("Phase 11 requires both checkpoint candidates")
    ordered = sorted(
        candidates,
        key=lambda step: (
            -float(candidates[step]["prototype_macro_pass_rate"]),
            -float(candidates[step]["micro_pass_rate"]),
            step,
        ),
    )
    return {
        "schema_version": "1",
        "selected_step": ordered[0],
        "rule": "prototype_macro_then_micro_then_earlier",
        "candidates": candidates,
    }


def dev_gate(baseline: dict[str, Any], selected: dict[str, Any]) -> bool:
    return float(selected["prototype_macro_pass_rate"]) > float(
        baseline["prototype_macro_pass_rate"]
    ) and float(selected["micro_pass_rate"]) > float(baseline["micro_pass_rate"])


def verify_lock(
    lock: dict[str, Any],
    run: Path,
    experiment_sha256: str,
    split_sha256: str,
    source_revision: str | None = None,
) -> None:
    if (
        lock.get("holdout_opened") is not False
        or lock.get("experiment_sha256") != experiment_sha256
    ):
        raise ValueError("Phase 11 selection lock experiment or holdout state invalid")
    if lock.get("split_manifest_sha256") != split_sha256:
        raise ValueError("Phase 11 selection lock split changed")
    if source_revision is not None and lock.get("source_git_revision") != source_revision:
        raise ValueError("Phase 11 selection lock source revision changed")
    if lock.get("worktree_clean") is not True:
        raise ValueError("Phase 11 selection lock source was dirty")
    selected = run / "selected-adapter"
    if adapter_files(selected) != lock.get("adapter_files"):
        raise ValueError("Phase 11 selected adapter changed")
    for key, relative in (
        ("training_config_sha256", "training/training-config.json"),
        ("adapter_manifest_sha256", "training/adapter-manifest.json"),
        ("selection_decision_sha256", "selection-decision.json"),
        ("dev_baseline_sha256", "dev/baseline/rollouts.jsonl"),
        ("dev_step_160_sha256", "dev/step-160/rollouts.jsonl"),
        ("dev_step_320_sha256", "dev/step-320/rollouts.jsonl"),
    ):
        path = run / relative
        if not path.is_file() or lock.get(key) != digest(path):
            raise ValueError(f"Phase 11 selection lock {key} changed")
    decision = json.loads((run / "selection-decision.json").read_text())
    if decision.get("selected_step") != lock.get("selected_step"):
        raise ValueError("Phase 11 selection step changed")
