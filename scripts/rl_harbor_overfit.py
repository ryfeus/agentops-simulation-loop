"""SSM-only controller for the Phase 9 tiny Harbor GRPO overfit experiment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentops_demo.harbor.provenance import build_clean_wheel
from rl.phase8c.execution_suite import derive_execution_suite
from rl.phase8c.suite_check import load_oracle_cache
from rl.phase9.task_set import load_task_set
from scripts import rl_smoke as phase8a

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY_ROOT / ".rl-smoke" / "phase9"
OUTPUTS = REPOSITORY_ROOT / ".rl-smoke" / "terraform-outputs.json"
PHASE8B = REPOSITORY_ROOT / "rl" / "phase8b"
PHASE8C = REPOSITORY_ROOT / "rl" / "phase8c"
PHASE9 = REPOSITORY_ROOT / "rl" / "phase9"
EVIDENCE_FILES = (
    "run-status.json",
    "versions.json",
    "training-config.json",
    "training-task-set.json",
    "effective-trl-config.json",
    "training-result.json",
    "adapter-delta.json",
    "before-task-summary.json",
    "after-task-summary.json",
    "before-after-summary.json",
    "regression-summary.json",
    "task-training-exposure.json",
    "rollouts-before.jsonl",
    "rollouts-train.jsonl",
    "rollouts-after.jsonl",
    "reward-groups.jsonl",
    "training-metrics.jsonl",
    "training.log",
    "artifacts-manifest.json",
    "gpu.csv",
    "docker.txt",
    "nvidia-smi.txt",
)


class OverfitControllerError(phase8a.SmokeError):
    """A Phase 9 failure that must preserve compact evidence."""


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    phase8a._atomic_json(path, value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_set_sha256(destination: Path) -> str:
    path = destination / "training-task-set.json"
    if not path.is_file():
        raise OverfitControllerError("training task-set evidence is missing")
    return _sha256(path)


def _backfill_task_set_lineage(destination: Path) -> str:
    """Bind retained legacy evidence to the exact fetched task-set bytes."""

    digest = _task_set_sha256(destination)
    identity_path = destination / "identity.json"
    identity = phase8a.load_json_object(identity_path, "Phase 9 identity")
    recorded = identity.get("training_task_set_sha256")
    if recorded not in (None, digest):
        raise OverfitControllerError("Phase 9 identity task-set SHA does not match evidence")
    if recorded is None:
        identity["training_task_set_sha256"] = digest
        _write_atomic(identity_path, identity)
    return digest


def _positive_env(name: str, default: int | float) -> int | float:
    value = os.environ.get(name, str(default)).strip()
    try:
        parsed: int | float = int(value) if isinstance(default, int) else float(value)
    except ValueError as exc:
        raise OverfitControllerError(f"{name} must be numeric") from exc
    if parsed <= 0:
        raise OverfitControllerError(f"{name} must be positive")
    return parsed


def _task_set_path() -> Path:
    candidate = Path(os.environ.get("PHASE9_TASK_SET", "rl/phase9/task_sets/tiny-overfit-v1.json"))
    path = candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate
    if not path.is_file():
        raise OverfitControllerError(f"PHASE9_TASK_SET does not identify a file: {candidate}")
    return path.resolve()


def _run_id(requested: str | None) -> str:
    if requested:
        return phase8a.validate_run_id(requested)
    return phase8a.validate_run_id("phase9-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S"))


def _payload_files(wheel: Path) -> tuple[Path, ...]:
    files = (
        REPOSITORY_ROOT / "rl" / "pyproject.toml",
        REPOSITORY_ROOT / "rl" / "uv.lock",
        PHASE8B / "billing_harbor_env.py",
        PHASE8B / "harbor_compat.py",
        PHASE8B / "sandbox_billing_bridge.py",
        PHASE8B / "execution_task.py",
        PHASE8C / "baseline_env.py",
        PHASE8C / "execution_suite.py",
        *sorted(PHASE9.glob("*.py")),
        PHASE9 / "remote_run.sh",
        wheel,
    )
    missing = [str(path.relative_to(REPOSITORY_ROOT)) for path in files if not path.is_file()]
    if missing:
        raise OverfitControllerError(f"Phase 9 runtime payload is incomplete: {', '.join(missing)}")
    return files


def package_payload(
    wheel: Path, suite_root: Path, oracle: Path, task_set: Path
) -> tuple[bytes, str]:
    """Package only locked runtime code, derived suite, Oracle proof, selection, and wheel."""

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in _payload_files(wheel):
            archive.add(
                path,
                arcname=f"wheel/{path.name}"
                if path == wheel
                else str(path.relative_to(REPOSITORY_ROOT)),
            )
        archive.add(suite_root / "tasks", arcname="dataset/tasks")
        archive.add(suite_root / "execution-suite.json", arcname="dataset/execution-suite.json")
        archive.add(oracle, arcname="oracle-suite-check.json")
        archive.add(task_set, arcname="task-set.json")
    payload = stream.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def _retain_exact_task_set(destination: Path, raw: bytes) -> None:
    """Keep operator selection byte-exact while rejecting unsafe or changed evidence."""

    identity = phase8a.load_json_object(destination / "identity.json", "Phase 9 identity")
    expected = identity.get("training_task_set_sha256")
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise OverfitControllerError("remote Phase 9 task-set SHA does not match identity")
    try:
        text = raw.decode("utf-8")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OverfitControllerError("remote Phase 9 task-set evidence is invalid JSON") from exc
    if not isinstance(value, dict):
        raise OverfitControllerError("remote Phase 9 task-set evidence is not an object")
    if phase8a.redact_text(text) != text or phase8a._redact_value(value) != value:
        raise OverfitControllerError("remote Phase 9 task-set evidence contains a secret")
    target = destination / "training-task-set.json"
    if target.is_file() and target.read_bytes() == raw:
        return
    if target.exists():
        prior_digest = _sha256(target)[:12]
        target.rename(destination / f"training-task-set.invalid-{prior_digest}.json")
    temporary = destination / ".training-task-set.json.tmp"
    temporary.write_bytes(raw)
    os.replace(temporary, target)


def collect_evidence(ssm_client: Any, instance_id: str, run_id: str, destination: Path) -> None:
    """Fetch compact evidence through the existing checksummed SSM transport."""

    phase8a.collect_evidence(ssm_client, instance_id, run_id, destination)
    remote = f"{phase8a.REMOTE_ROOT}/{run_id}/run"
    for name in EVIDENCE_FILES:
        target = destination / name
        if name == "training-task-set.json":
            identity = phase8a.load_json_object(destination / "identity.json", "Phase 9 identity")
            expected = identity.get("training_task_set_sha256")
            if target.is_file() and (expected is None or _sha256(target) == expected):
                continue
            raw = phase8a._read_remote_file(ssm_client, instance_id, f"{remote}/{name}")
            _retain_exact_task_set(destination, raw)
            continue
        if target.is_file():
            continue
        try:
            target.write_bytes(
                phase8a._read_remote_file(ssm_client, instance_id, f"{remote}/{name}")
            )
            phase8a.redacted_copy(target, target)
        except phase8a.SmokeError:
            continue
    artifacts = phase8a.load_json_object(
        destination / "artifacts-manifest.json", "artifact manifest"
    )
    try:
        adapter_files = phase8a.artifact_manifest_files(artifacts, "adapter/final")
        try:
            verify_local_adapter(destination)
            adapter_verified = True
        except (OverfitControllerError, ValueError):
            adapter_verified = False
        adapter_destination = destination / "adapter" / "final"
        if adapter_verified:
            transfer = {"verified_local_noop": True, "file_count": len(adapter_files)}
        else:
            if adapter_destination.exists():
                preserved = adapter_destination.with_name(
                    f"final-incomplete-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
                )
                shutil.move(str(adapter_destination), str(preserved))
            transfer = phase8a.download_remote_directory(
                ssm_client,
                instance_id,
                f"{remote}/adapter/final",
                adapter_destination,
                adapter_files,
                journal_path=destination / "adapter-transfer.journal.json",
            )
        _write_atomic(
            destination / "adapter-transfer.json",
            {"schema_version": "1", "files": adapter_files, **transfer},
        )
        if os.environ.get("PHASE9_DOWNLOAD_CHECKPOINTS", "") == "1":
            try:
                checkpoints = phase8a.artifact_manifest_files(artifacts, "checkpoints")
            except phase8a.SmokeError:
                checkpoints = {}
            if checkpoints:
                checkpoint_transfer = phase8a.download_remote_directory(
                    ssm_client,
                    instance_id,
                    f"{remote}/checkpoints",
                    destination / "checkpoints",
                    checkpoints,
                )
                _write_atomic(
                    destination / "checkpoint-transfer.json",
                    {"schema_version": "1", "files": checkpoints, **checkpoint_transfer},
                )
    except (ValueError, phase8a.SmokeError) as exc:
        _write_atomic(
            destination / "adapter-transfer.json", {"schema_version": "1", "error": str(exc)}
        )
        raise OverfitControllerError(f"could not retain the final Phase 9 adapter: {exc}") from exc


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise OverfitControllerError(f"cannot read {label}: {exc}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise OverfitControllerError(f"{label} must contain JSON object rows")
    return rows


def _evidence_task_ids(destination: Path) -> tuple[set[str], set[str]]:
    task_set = phase8a.load_json_object(destination / "training-task-set.json", "training task set")

    def ids(key: str) -> set[str]:
        value = task_set.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise OverfitControllerError(f"training task set {key} is invalid")
        return {str(item.get("task_id", "")) for item in value}

    return ids("training_tasks"), ids("regression_tasks")


def validate_rollout_evidence(destination: Path) -> dict[str, Any]:
    """Validate semantic remote evidence before accepting an overfit result."""

    training_ids, regression_ids = _evidence_task_ids(destination)
    evaluation_ids = training_ids | regression_ids
    ids_by_stage = {"before": evaluation_ids, "train": training_ids, "after": evaluation_ids}
    rows_by_stage: dict[str, list[dict[str, Any]]] = {}
    rollout_ids: set[str] = set()
    for stage, allowed in ids_by_stage.items():
        rows = _read_jsonl(destination / f"rollouts-{stage}.jsonl", f"{stage} rollouts")
        for row in rows:
            if row.get("stage") != stage or str(row.get("task_id", "")) not in allowed:
                raise OverfitControllerError(f"invalid {stage} rollout task or stage")
            reward = row.get("reward")
            if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
                raise OverfitControllerError(f"invalid {stage} rollout reward")
            rollout_id = row.get("rollout_id")
            if not isinstance(rollout_id, str) or not rollout_id or rollout_id in rollout_ids:
                raise OverfitControllerError(f"invalid or duplicate {stage} rollout_id")
            rollout_ids.add(rollout_id)
        rows_by_stage[stage] = rows
    groups = _read_jsonl(destination / "reward-groups.jsonl", "reward groups")
    for group in groups:
        references = group.get("rollout_ids")
        if not isinstance(references, list) or len(references) != 4:
            raise OverfitControllerError("reward group must reference four rollout IDs")
        if any(not isinstance(value, str) or value not in rollout_ids for value in references):
            raise OverfitControllerError("reward group has unknown rollout ID")
    metrics = _read_jsonl(destination / "training-metrics.jsonl", "training metrics")
    if not all(
        any(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in row.values()
        )
        for row in metrics
    ):
        raise OverfitControllerError("training metrics contain no finite numeric values")
    return {
        "schema_version": "1",
        "stage_rows": {stage: len(rows) for stage, rows in rows_by_stage.items()},
        "reward_group_count": len(groups),
        "training_metric_count": len(metrics),
        "valid": True,
    }


def verify_local_adapter(destination: Path) -> dict[str, Any]:
    artifacts = phase8a.load_json_object(
        destination / "artifacts-manifest.json", "artifact manifest"
    )
    expected = phase8a.artifact_manifest_files(artifacts, "adapter/final")
    root = destination / "adapter" / "final"
    if not root.is_dir():
        raise OverfitControllerError("final adapter was not retained locally")
    actual: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            if path.is_symlink():
                raise OverfitControllerError("retained adapter must not contain symlinks")
            continue
        relative = path.relative_to(root).as_posix()
        actual[relative] = {
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    if actual != expected:
        raise OverfitControllerError("retained adapter does not match artifact manifest")
    return {"schema_version": "1", "files": expected, "verified": True}


def write_summary(
    destination: Path, outputs: dict[str, Any], remote: phase8a.SsmResult | None
) -> dict[str, Any]:
    def load(name: str) -> dict[str, Any]:
        try:
            return phase8a.load_json_object(destination / name, name)
        except ValueError:
            return {}

    status, versions = load("run-status.json"), load("versions.json")
    result, config, delta = (
        load("training-result.json"),
        load("training-config.json"),
        load("adapter-delta.json"),
    )
    identity, suite, oracle = (
        load("identity.json"),
        load("execution-suite.json"),
        load("oracle-suite-check.json"),
    )
    artifacts = load("artifacts-manifest.json")
    # Artifact collection can resume after the source worktree has changed.
    # Canonicality belongs to the clean revision that produced the wheel, not
    # the controller's unrelated state at the time of evidence retrieval.
    current_revision, current_clean = phase8a.git_provenance()
    source_revision = identity.get("source_revision")
    source_clean = identity.get("worktree_clean")
    revision = source_revision if isinstance(source_revision, str) else current_revision
    clean = source_clean if isinstance(source_clean, bool) else current_clean
    expected_steps = config.get("training", {}).get("max_steps")
    try:
        rollout_evidence = validate_rollout_evidence(destination)
    except OverfitControllerError as exc:
        rollout_evidence = {"valid": False, "error": str(exc)}
    try:
        local_adapter = verify_local_adapter(destination)
    except (OverfitControllerError, ValueError) as exc:
        local_adapter = {"verified": False, "error": str(exc)}
    try:
        task_set_sha256 = _backfill_task_set_lineage(destination)
    except OverfitControllerError:
        task_set_sha256 = ""
    passed = (
        (remote is None or remote.succeeded)
        and status.get("training") == "PASS"
        and result.get("status") == "PASS"
        and result.get("global_step") == expected_steps
        and result.get("training_execution_valid") is True
        and float(delta.get("delta_l2_norm", 0.0)) > 0
        and bool(local_adapter.get("verified"))
        and bool(rollout_evidence.get("valid"))
        and oracle.get("status") == "PASS"
        and oracle.get("execution_suite_sha256") == suite.get("execution_suite_sha256")
    )
    value: dict[str, Any] = {
        "schema_version": "1",
        "phase": "9",
        "status": "PASS" if passed else "FAIL",
        "canonical_acceptance": passed and clean,
        "git_revision": revision,
        "worktree_clean": clean,
        "instance_type": phase8a.terraform_output_value(outputs, "instance_type"),
        "ami_id": phase8a.terraform_output_value(outputs, "ami_id"),
        "availability_zone": phase8a.terraform_output_value(outputs, "availability_zone"),
        "gpu": versions.get("gpu"),
        "model_id": result.get("model_id"),
        "trl_version": versions.get("trl"),
        "vllm_version": versions.get("vllm"),
        "harbor_version": versions.get("harbor"),
        "task_set_name": result.get("task_set_name"),
        "execution_suite_sha256": suite.get("execution_suite_sha256"),
        "base_suite_sha256": suite.get("base_suite_sha256"),
        "source_phase8c_runs": identity.get("source_phase8c_runs", []),
        "configured_max_steps": expected_steps,
        "global_step": result.get("global_step", 0),
        "training_execution_valid": result.get("training_execution_valid", False),
        "training_reward_groups": result.get("training_reward_groups", 0),
        "groups_with_reward_variance": result.get("groups_with_reward_variance", 0),
        "adapter_delta_l2_norm": delta.get("delta_l2_norm", 0.0),
        "adapter_artifacts": len(artifacts.get("artifacts", [])),
        "adapter_persisted_locally": bool(local_adapter.get("verified")),
        "adapter_file_count": len(local_adapter.get("files", {})),
        "adapter_checksums_verified": bool(local_adapter.get("verified")),
        "rollout_evidence": rollout_evidence,
        "train_pass_rate_before": result.get("train_pass_rate_before"),
        "train_pass_rate_after": result.get("train_pass_rate_after"),
        "train_pass_rate_delta": result.get("train_pass_rate_delta"),
        "training_tasks_improved": result.get("training_tasks_improved", 0),
        "behavioral_improvement_observed": result.get("behavioral_improvement_observed", False),
        "strong_overfit_signal": result.get("strong_overfit_signal", False),
        "phase9_success_candidate": result.get("phase9_success_candidate", False),
        "catastrophic_regression_detected": result.get("catastrophic_regression_detected", False),
        "human_review_required": result.get("human_review_required", True),
        "no_grpo_signal": result.get("no_grpo_signal", False),
        "wheel_sha256": identity.get("wheel_sha256"),
        "payload_sha256": identity.get("payload_sha256"),
        "source_revision": identity.get("source_revision"),
        "training_task_set_sha256": task_set_sha256,
    }
    if not passed:
        text = "\n".join((remote.stdout, remote.stderr)) if remote else ""
        value["failure_phase"] = result.get("failure_phase", status.get("stage", ""))
        value["failure_class"] = phase8a.classify_failure(text, str(value["failure_phase"]))
    # The summary is acceptance evidence: never expose a partially rewritten
    # PASS while an interrupted adapter transfer is being resumed.
    _write_atomic(destination / "summary.json", value)
    return value


def source_check(source_run: str) -> dict[str, Any]:
    """Fail closed unless a retained Phase 9 run is eligible for Phase 9b."""

    run_id = phase8a.validate_run_id(source_run)
    destination = ROOT / "runs" / run_id
    summary = phase8a.load_json_object(destination / "summary.json", "Phase 9 source summary")
    if summary.get("status") != "PASS":
        raise OverfitControllerError("Phase 9 source summary is not PASS")
    if (
        summary.get("adapter_persisted_locally") is not True
        or summary.get("adapter_checksums_verified") is not True
    ):
        raise OverfitControllerError("Phase 9 source adapter is not locally verified")
    if not isinstance(summary.get("source_revision"), str) or not summary["source_revision"]:
        raise OverfitControllerError("Phase 9 source revision is missing")
    task_set_sha256 = _backfill_task_set_lineage(destination)
    if summary.get("training_task_set_sha256") != task_set_sha256:
        raise OverfitControllerError("Phase 9 source task-set SHA does not match its summary")
    suite = phase8a.load_json_object(destination / "execution-suite.json", "execution suite")
    if summary.get("execution_suite_sha256") != suite.get("execution_suite_sha256"):
        raise OverfitControllerError(
            "Phase 9 source execution-suite SHA does not match its summary"
        )
    task_set = load_task_set(destination / "training-task-set.json", suite)
    if len(task_set.training_ids) != 4 or len(task_set.regression_ids) != 1:
        raise OverfitControllerError(
            "Phase 9b requires exactly four training tasks and one control"
        )
    adapter = verify_local_adapter(destination)
    evidence = validate_rollout_evidence(destination)
    return {
        "schema_version": "1",
        "status": "PASS",
        "source_run": run_id,
        "source_revision": summary["source_revision"],
        "execution_suite_sha256": suite["execution_suite_sha256"],
        "training_task_set_sha256": task_set_sha256,
        "training_task_count": len(task_set.training_ids),
        "regression_control_count": len(task_set.regression_ids),
        "adapter_file_count": len(adapter["files"]),
        "rollout_evidence": evidence,
    }


def fetch_artifacts(outputs_path: Path = OUTPUTS, requested_run_id: str | None = None) -> Path:
    """Resume only Phase 9 evidence retention; this never invokes a rollout."""

    run_id = (
        phase8a.validate_run_id(requested_run_id)
        if requested_run_id
        else str(phase8a.load_json_object(ROOT / "latest-run.json", "latest Phase 9 run")["run_id"])
    )
    destination = ROOT / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    outputs = phase8a.load_json_object(outputs_path, "RL smoke Terraform outputs")
    # A completed source can be inspected after G6 teardown without touching AWS.
    try:
        source_check(run_id)
        write_summary(destination, outputs, None)
        return destination
    except (OverfitControllerError, ValueError):
        pass

    instance_id = phase8a.terraform_output_value(outputs, "instance_id")
    sts, ec2, ssm = phase8a._aws_clients()
    phase8a.require_account(sts)
    phase8a.wait_for_instance(ec2, instance_id)
    phase8a.wait_for_ssm(ssm, instance_id)
    collect_evidence(ssm, instance_id, run_id, destination)
    summary = write_summary(destination, outputs, None)
    if summary["status"] != "PASS":
        raise OverfitControllerError(f"Phase 9 retained evidence is incomplete: {destination}")
    source_check(run_id)
    return destination


def run_overfit(outputs_path: Path = OUTPUTS, requested_run_id: str | None = None) -> Path:
    outputs = phase8a.load_json_object(outputs_path, "RL smoke Terraform outputs")
    instance_id = phase8a.terraform_output_value(outputs, "instance_id")
    max_steps = int(_positive_env("PHASE9_MAX_STEPS", 20))
    learning_rate = float(_positive_env("PHASE9_LEARNING_RATE", 1e-5))
    seed = int(_positive_env("PHASE9_SEED", 20260922))
    if os.environ.get("PHASE9_NUM_GENERATIONS", "4") != "4":
        raise OverfitControllerError("PHASE9_NUM_GENERATIONS must equal 4")
    run_id = _run_id(requested_run_id)
    destination = ROOT / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    _write(ROOT / "latest-run.json", {"run_id": run_id})
    suite = derive_execution_suite(destination / "execution-suite")
    _write(destination / "execution-suite.json", suite.manifest)
    oracle = ROOT.parent / "phase8c" / "oracle-cache" / f"{suite.execution_suite_sha256}.json"
    if not load_oracle_cache(
        oracle, suite.execution_suite_sha256, int(suite.manifest["task_count"])
    ):
        raise OverfitControllerError("matching Phase 8C Oracle suite evidence is missing")
    _write(destination / "oracle-suite-check.json", json.loads(oracle.read_text(encoding="utf-8")))
    task_set_path = _task_set_path()
    task_set = load_task_set(task_set_path, suite.manifest)
    task_set_sha256 = _sha256(task_set_path)
    wheel, package = build_clean_wheel(ROOT / "package")
    payload, digest = package_payload(wheel, suite.root, oracle, task_set_path)
    _write(
        destination / "identity.json",
        {
            **package.model_dump(),
            "wheel_sha256": package.sha256,
            "payload_sha256": digest,
            "training_task_set_sha256": task_set_sha256,
            "source_phase8c_runs": list(task_set.source_phase8c_runs),
            **suite.manifest,
        },
    )
    sts, ec2, ssm = phase8a._aws_clients()
    phase8a.require_account(sts)
    phase8a.wait_for_instance(ec2, instance_id)
    phase8a.wait_for_ssm(ssm, instance_id)
    remote_dir = phase8a.upload_payload(ssm, instance_id, run_id, payload, digest)
    command = (
        f"PHASE9_MAX_STEPS={max_steps} PHASE9_LEARNING_RATE={learning_rate} PHASE9_SEED={seed} "
        f"bash {remote_dir}/rl/phase9/remote_run.sh --run-root {remote_dir}/run"
    )
    result = phase8a.run_shell(ssm, instance_id, [command], timeout=7200)
    try:
        collect_evidence(ssm, instance_id, run_id, destination)
    finally:
        summary = write_summary(destination, outputs, result)
    if not result.succeeded or summary["status"] != "PASS":
        raise OverfitControllerError(f"Phase 9 overfit failed; evidence: {destination}")
    return destination


def evidence(requested_run_id: str | None = None) -> Path:
    run_id = (
        phase8a.validate_run_id(requested_run_id)
        if requested_run_id
        else str(phase8a.load_json_object(ROOT / "latest-run.json", "latest Phase 9 run")["run_id"])
    )
    path = ROOT / "runs" / run_id / "summary.json"
    if not path.is_file():
        raise OverfitControllerError(
            "Phase 9 evidence is missing; run aws-rl-harbor-overfit-run first"
        )
    print(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    run_parser.add_argument("--run-id")
    evidence_parser = commands.add_parser("evidence")
    evidence_parser.add_argument("--run-id")
    fetch_parser = commands.add_parser("fetch-artifacts")
    fetch_parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    fetch_parser.add_argument("--run-id")
    source_parser = commands.add_parser("source-check")
    source_parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        print(run_overfit(args.outputs, args.run_id))
    elif args.command == "evidence":
        evidence(args.run_id)
    elif args.command == "fetch-artifacts":
        print(fetch_artifacts(args.outputs, args.run_id))
    else:
        print(json.dumps(source_check(args.run_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
