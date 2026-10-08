"""SSM-only controller for the disposable Phase 8B TRL/Harbor smoke."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentops_demo.harbor.provenance import PackageProvenance, build_clean_wheel
from agentops_demo.taskify.integrity import validate_harbor_task
from rl.phase8b.execution_task import ExecutionTask, derive_execution_task
from scripts import rl_smoke as phase8a

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY_ROOT / ".rl-smoke" / "phase8b"
OUTPUTS = REPOSITORY_ROOT / ".rl-smoke" / "terraform-outputs.json"
TASK_ID = "paid-refund-direct"
TASK = REPOSITORY_ROOT / "benchmarks" / "billing" / "tasks" / TASK_ID
PHASE_DIR = REPOSITORY_ROOT / "rl" / "phase8b"
EVIDENCE_FILES = (
    "run-status.json",
    "versions.json",
    "harbor-env-preflight.json",
    "harbor-verifier-debug.json",
    "training-harbor.json",
    "training-harbor.log",
    "gpu.csv",
    "docker.txt",
    "nvidia-smi.txt",
    "nvidia-smi-before-training.txt",
    "nvidia-smi-after-training.txt",
)


class HarborSmokeError(phase8a.SmokeError):
    """A Phase 8B result that must not be represented as a passing smoke."""


def _files_for_payload(wheel: Path, execution_task: Path) -> tuple[Path, ...]:
    runtime = (
        REPOSITORY_ROOT / "rl" / "pyproject.toml",
        REPOSITORY_ROOT / "rl" / "uv.lock",
        PHASE_DIR / "billing_harbor_env.py",
        PHASE_DIR / "harbor_compat.py",
        PHASE_DIR / "execution_task.py",
        PHASE_DIR / "sandbox_billing_bridge.py",
        PHASE_DIR / "env_smoke.py",
        PHASE_DIR / "train_harbor_smoke.py",
        PHASE_DIR / "remote_run.sh",
    )
    missing = [str(path.relative_to(REPOSITORY_ROOT)) for path in runtime if not path.is_file()]
    if missing:
        raise HarborSmokeError(f"Phase 8B runtime payload is incomplete: {', '.join(missing)}")
    if not wheel.is_file():
        raise HarborSmokeError("clean project wheel is missing")
    validate_harbor_task(TASK)
    validate_harbor_task(execution_task)
    return (*runtime, wheel)


def package_payload(wheel: Path, execution_task: Path) -> tuple[bytes, str]:
    """Create a minimal locked runtime: Phase 8B, one task, and one wheel."""

    files = _files_for_payload(wheel, execution_task)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in files:
            if path == wheel:
                archive.add(path, arcname=f"wheel/{path.name}")
            else:
                archive.add(path, arcname=str(path.relative_to(REPOSITORY_ROOT)))
        archive.add(execution_task, arcname=f"dataset/tasks/{TASK_ID}")
    payload = stream.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def _run_id(requested: str | None) -> str:
    pointer = ROOT / "latest-run.json"
    if requested:
        return phase8a.validate_run_id(requested)
    if pointer.is_file():
        saved = phase8a.load_json_object(pointer, "latest run").get("run_id")
        return phase8a.validate_run_id(str(saved))
    from datetime import UTC, datetime

    return "phase8b-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S")


def _write_identity(
    destination: Path,
    *,
    package: PackageProvenance,
    payload_sha256: str,
    execution_task: ExecutionTask,
) -> dict[str, str]:
    value = {
        "schema_version": "1",
        "task_id": TASK_ID,
        # Preserve the legacy field as the canonical task hash for older evidence readers.
        "harbor_task_sha256": execution_task.base_task_sha256,
        "base_task_sha256": execution_task.base_task_sha256,
        "execution_task_sha256": execution_task.execution_task_sha256,
        "execution_profile": execution_task.execution_profile,
        "source_revision": package.source_revision,
        "wheel_sha256": package.sha256,
        "payload_sha256": payload_sha256,
    }
    (destination / "task-identity.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    return value


def classify_failure(text: str, stage: str = "") -> str:
    normalized = f"{stage}\n{text}".lower()
    checks = (
        ("HARBOR_VERIFIER", ("rewardfilenotfounderror", "reward file not found")),
        ("DOCKER_UNAVAILABLE", ("docker_unavailable", "docker", "docker daemon")),
        ("HARBOR_IMPORT", ("harbor_import", "harbor import", "no module named 'harbor'")),
        (
            "HARBOR_TASK_INVALID",
            ("harbor_task_invalid", "single harbor task", "harborspec", "task is missing"),
        ),
        (
            "HARBOR_SANDBOX_START",
            ("harbor_sandbox_start", "sandbox", "environment start", "docker build"),
        ),
        ("HARNESS_SETUP", ("harness_setup", "sandbox billing wheel", "phase8b_wheel_path")),
        ("BILLING_BRIDGE", ("billing_bridge", "sandbox_billing_bridge", "billing operation")),
        (
            "HARBOR_VERIFIER",
            ("harbor_verifier", "unexpected harbor rewards", "correct_reward", "noop_reward"),
        ),
        ("MODEL_NO_TOOL_CALL", ("model_no_tool_call", "tool call frequency was zero")),
        ("TRL_HARBOR_INIT", ("trl_harbor_init",)),
        ("TRL_HARBOR_ROLLOUT", ("trl_harbor_rollout",)),
        ("GRPO_BACKWARD", ("grpo_backward", "optimizer step")),
        ("VLLM_OOM", ("out of memory", "cuda oom")),
    )
    for name, markers in checks:
        if any(marker in normalized for marker in markers):
            return name
    return phase8a.classify_failure(text, stage)


def collect_evidence(ssm_client: Any, instance_id: str, run_id: str, destination: Path) -> None:
    """Retrieve the Phase 8B evidence bundle, preserving partial result records."""

    phase8a.collect_evidence(ssm_client, instance_id, run_id, destination)
    remote_dir = f"{phase8a.REMOTE_ROOT}/{run_id}/run"
    for name in EVIDENCE_FILES:
        target = destination / name
        if target.is_file():
            continue
        try:
            target.write_bytes(
                phase8a._read_remote_file(ssm_client, instance_id, f"{remote_dir}/{name}")
            )
        except phase8a.SmokeError:
            continue
        phase8a.redacted_copy(target, target)


def write_summary(
    destination: Path, outputs: Mapping[str, Any], remote_result: phase8a.SsmResult | None
) -> dict[str, Any]:
    def load(name: str, label: str) -> dict[str, Any]:
        try:
            return phase8a.load_json_object(destination / name, label)
        except ValueError:
            return {}

    status = load("run-status.json", "Phase 8B run status")
    versions = load("versions.json", "Phase 8B versions")
    identity = load("task-identity.json", "Phase 8B identity")
    preflight = load("harbor-env-preflight.json", "Harbor preflight")
    training = load("training-harbor.json", "Harbor training")
    revision, clean = phase8a.git_provenance()
    noop = preflight.get("noop_reward")
    correct = preflight.get("correct_reward")
    global_step = training.get("global_step", 0)
    tool_frequency = training.get("tool_call_frequency", 0)
    passed = (
        (remote_result is None or remote_result.succeeded)
        and status.get("harbor_env_preflight") == "PASS"
        and status.get("training") == "PASS"
        and preflight.get("status") == "PASS"
        and noop == 0.0
        and correct == 1.0
        and preflight.get("noop_reward_json_exists") is True
        and preflight.get("correct_reward_json_exists") is True
        and training.get("status") == "PASS"
        and isinstance(global_step, int)
        and global_step >= 1
        and isinstance(tool_frequency, (int, float))
        and tool_frequency > 0
    )
    value: dict[str, Any] = {
        "schema_version": "1",
        "phase": "8B",
        "status": "PASS" if passed else "FAIL",
        "canonical_acceptance": passed and clean,
        "git_revision": revision,
        "worktree_clean": clean,
        "instance_type": phase8a.terraform_output_value(outputs, "instance_type"),
        "ami_id": phase8a.terraform_output_value(outputs, "ami_id"),
        "availability_zone": phase8a.terraform_output_value(outputs, "availability_zone"),
        "gpu": versions.get("gpu"),
        "model_id": "Qwen/Qwen3-0.6B",
        "trl_version": versions.get("trl"),
        "vllm_version": versions.get("vllm"),
        "harbor_version": versions.get("harbor"),
        "task_id": identity.get("task_id", TASK_ID),
        "harbor_task_sha256": identity.get("harbor_task_sha256"),
        "base_task_sha256": identity.get("base_task_sha256", identity.get("harbor_task_sha256")),
        "execution_task_sha256": identity.get("execution_task_sha256"),
        "execution_profile": identity.get("execution_profile"),
        "wheel_sha256": identity.get("wheel_sha256"),
        "payload_sha256": identity.get("payload_sha256"),
        "noop_reward": noop,
        "correct_reward": correct,
        "global_step": global_step,
        "tool_call_frequency": tool_frequency,
        "tool_failure_frequency": training.get("tool_failure_frequency"),
        "reward": training.get("reward"),
        "reward_std": training.get("reward_std"),
    }
    if not passed:
        failed_record = training if training.get("status") == "FAIL" else preflight
        error = str(failed_record.get("error", ""))
        value["failure_phase"] = str(failed_record.get("failure_phase", status.get("stage", "")))
        value["failure_class"] = classify_failure(error, value["failure_phase"])
    (destination / "summary.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def run_smoke(outputs_path: Path = OUTPUTS, requested_run_id: str | None = None) -> Path:
    """Package, execute, retrieve, and fail closed on one Phase 8B run."""

    outputs = phase8a.load_json_object(outputs_path, "RL smoke Terraform outputs")
    instance_id = phase8a.terraform_output_value(outputs, "instance_id")
    run_id = _run_id(requested_run_id)
    destination = ROOT / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    (ROOT / "latest-run.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    wheel, package = build_clean_wheel(ROOT / "package")
    execution_task = derive_execution_task(TASK, destination / "execution-task")
    payload, digest = package_payload(wheel, execution_task.path)
    _write_identity(
        destination,
        package=package,
        payload_sha256=digest,
        execution_task=execution_task,
    )
    sts, ec2, ssm = phase8a._aws_clients()
    phase8a.require_account(sts)
    phase8a.wait_for_instance(ec2, instance_id)
    phase8a.wait_for_ssm(ssm, instance_id)
    remote_dir = phase8a.upload_payload(ssm, instance_id, run_id, payload, digest)
    result = phase8a.run_shell(
        ssm,
        instance_id,
        [f"bash {remote_dir}/rl/phase8b/remote_run.sh --run-root {remote_dir}/run"],
        timeout=7200,
    )
    try:
        collect_evidence(ssm, instance_id, run_id, destination)
    finally:
        summary = write_summary(destination, outputs, result)
    if not result.succeeded or summary["status"] != "PASS":
        raise HarborSmokeError(f"Phase 8B run failed; evidence: {destination}")
    return destination


def evidence(requested_run_id: str | None = None) -> Path:
    path = ROOT / "runs" / _run_id(requested_run_id) / "summary.json"
    if not path.is_file():
        raise HarborSmokeError("Phase 8B evidence is missing; run aws-rl-harbor-smoke-run first")
    print(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--outputs", type=Path, default=OUTPUTS)
    run.add_argument("--run-id")
    show = commands.add_parser("evidence")
    show.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.command == "run":
        print(run_smoke(args.outputs, args.run_id))
    else:
        evidence(args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
