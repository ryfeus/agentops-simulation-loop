"""Strict Phase 9 persisted-adapter provenance and integrity checks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from rl.phase8c.evaluate_baseline import MODEL_ID


@dataclass(frozen=True)
class AdapterIdentity:
    source_run: str
    source_dir: Path
    adapter_dir: Path
    task_set_path: Path
    files: dict[str, dict[str, Any]]
    source_revision: str
    execution_suite_sha256: str
    task_set_sha256: str
    manifest_sha256: str
    config_sha256: str
    weights_sha256: str

    def value(self) -> dict[str, Any]:
        return {
            "source_run": self.source_run,
            "source_revision": self.source_revision,
            "execution_suite_sha256": self.execution_suite_sha256,
            "training_task_set_sha256": self.task_set_sha256,
            "adapter_path": "adapter/final",
            "artifact_manifest_sha256": self.manifest_sha256,
            "adapter_config_sha256": self.config_sha256,
            "adapter_weights_sha256": self.weights_sha256,
            "files": self.files,
        }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _adapter_manifest_files(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate the adapter inventory without importing the local AWS controller."""

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Phase 9 artifact manifest has no artifacts list")
    root = PurePosixPath("adapter/final")
    files: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ValueError("Phase 9 artifact manifest entry is invalid")
        raw_path, size, digest = item.get("path"), item.get("size"), item.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(size, int) or size < 0:
            raise ValueError("Phase 9 artifact manifest path or size is invalid")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Phase 9 artifact manifest SHA-256 is invalid")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("Phase 9 artifact manifest contains an unsafe path")
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if not relative.parts or relative.as_posix() in files:
            raise ValueError("Phase 9 artifact manifest has an empty or duplicate adapter path")
        files[relative.as_posix()] = {"size": size, "sha256": digest}
    if not files:
        raise ValueError("Phase 9 artifact manifest has no final adapter files")
    return files


def validate_persisted_adapter(
    source_dir: Path, *, source_run: str, execution_suite_sha256: str
) -> AdapterIdentity:
    """Fail closed unless a retained Phase 9 final adapter matches its manifest."""

    source_dir = source_dir.resolve()
    summary = _load(source_dir / "summary.json", "Phase 9 source summary")
    if (
        summary.get("status") != "PASS"
        or summary.get("adapter_persisted_locally") is not True
        or summary.get("adapter_checksums_verified") is not True
    ):
        raise ValueError("Phase 9 source does not have a locally verified passing adapter")
    if summary.get("execution_suite_sha256") != execution_suite_sha256:
        raise ValueError("Phase 9 source execution suite does not match Phase 9b")
    source_revision = summary.get("source_revision")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("Phase 9 source revision is missing")
    task_set_path = source_dir / "training-task-set.json"
    if not task_set_path.is_file():
        raise ValueError("Phase 9 source task set is missing")
    task_set_sha256 = _sha(task_set_path)
    if summary.get("training_task_set_sha256") != task_set_sha256:
        raise ValueError("Phase 9 source task-set hash does not match its summary")
    manifest_path = source_dir / "artifacts-manifest.json"
    manifest = _load(manifest_path, "Phase 9 artifact manifest")
    files = _adapter_manifest_files(manifest)
    adapter_dir = source_dir / "adapter" / "final"
    actual: dict[str, dict[str, Any]] = {}
    for path in sorted(adapter_dir.rglob("*")) if adapter_dir.is_dir() else []:
        if path.is_symlink():
            raise ValueError("retained adapter contains a symlink")
        if path.is_file():
            actual[path.relative_to(adapter_dir).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": _sha(path),
            }
    if actual != files:
        raise ValueError("retained adapter files differ from the Phase 9 artifact manifest")
    config = _load(adapter_dir / "adapter_config.json", "adapter config")
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError("adapter base model is not the pinned Qwen3-0.6B model")
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("adapter is not a causal-LM LoRA adapter")
    if config.get("r") != 8 or config.get("lora_alpha") != 16 or config.get("bias") != "none":
        raise ValueError("adapter LoRA shape does not match the Phase 9 contract")
    targets = config.get("target_modules")
    if not targets:
        raise ValueError("adapter has no target modules")
    # PEFT may save trainer arguments alongside the adapter as training_args.bin.
    # Only the named adapter_model file is a serialized adapter weight file.
    weight_names = [
        name for name in files if name in {"adapter_model.safetensors", "adapter_model.bin"}
    ]
    if len(weight_names) != 1:
        raise ValueError("adapter must have exactly one serialized weight file")
    return AdapterIdentity(
        source_run=source_run,
        source_dir=source_dir,
        adapter_dir=adapter_dir,
        task_set_path=task_set_path,
        files=files,
        source_revision=source_revision,
        execution_suite_sha256=execution_suite_sha256,
        task_set_sha256=task_set_sha256,
        manifest_sha256=_sha(manifest_path),
        config_sha256=_sha(adapter_dir / "adapter_config.json"),
        weights_sha256=str(files[weight_names[0]]["sha256"]),
    )
