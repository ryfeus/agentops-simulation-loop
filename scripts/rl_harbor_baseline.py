"""SSM-only controller for the Phase 8C frozen Harbor baseline."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentops_demo.harbor.provenance import build_clean_wheel
from rl.phase8c.execution_suite import derive_execution_suite
from rl.phase8c.suite_check import load_oracle_cache
from scripts import rl_smoke as phase8a

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY_ROOT / ".rl-smoke" / "phase8c"
OUTPUTS = REPOSITORY_ROOT / ".rl-smoke" / "terraform-outputs.json"
PHASE_DIR = REPOSITORY_ROOT / "rl" / "phase8c"
PHASE8B_DIR = REPOSITORY_ROOT / "rl" / "phase8b"
EVIDENCE_FILES = (
    "run-status.json",
    "versions.json",
    "baseline-config.json",
    "evaluation-result.json",
    "eval-metrics.json",
    "rollouts.jsonl",
    "rollout-retries.jsonl",
    "task-summary.json",
    "archetype-summary.json",
    "difficulty-summary.json",
    "phase9-candidates.json",
    "evaluation.log",
    "gpu.csv",
    "docker.txt",
    "nvidia-smi.txt",
)


class BaselineControllerError(phase8a.SmokeError):
    """A failed frozen baseline that must never be represented as PASS."""


def _positive_env(name: str, default: int) -> int:
    value = os.environ.get(name, str(default)).strip()
    try:
        parsed = int(value)
    except ValueError as exc:
        raise BaselineControllerError(f"{name} must be an integer") from exc
    if parsed < 1:
        raise BaselineControllerError(f"{name} must be positive")
    return parsed


def _nonnegative_env(name: str, default: int) -> int:
    value = os.environ.get(name, str(default)).strip()
    try:
        parsed = int(value)
    except ValueError as exc:
        raise BaselineControllerError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise BaselineControllerError(f"{name} cannot be negative")
    return parsed


def _task_selection() -> set[str] | None:
    value = os.environ.get("PHASE8C_TASKS", "").strip()
    if not value:
        return None
    selected = {part.strip() for part in value.split(",") if part.strip()}
    if not selected:
        raise BaselineControllerError("PHASE8C_TASKS must contain comma-separated task IDs")
    return selected


def _run_id(requested: str | None, followup: bool = False) -> str:
    if requested:
        return phase8a.validate_run_id(requested)
    prefix = "phase8c-followup-" if followup else "phase8c-"
    return phase8a.validate_run_id(prefix + datetime.now(UTC).strftime("%Y%m%d%H%M%S"))


def _payload_files(wheel: Path) -> tuple[Path, ...]:
    files = (
        REPOSITORY_ROOT / "rl" / "pyproject.toml",
        REPOSITORY_ROOT / "rl" / "uv.lock",
        PHASE8B_DIR / "billing_harbor_env.py",
        PHASE8B_DIR / "harbor_compat.py",
        PHASE8B_DIR / "sandbox_billing_bridge.py",
        PHASE8B_DIR / "execution_task.py",
        *sorted(PHASE_DIR.glob("*.py")),
        PHASE_DIR / "remote_run.sh",
        wheel,
    )
    missing = [str(path.relative_to(REPOSITORY_ROOT)) for path in files if not path.is_file()]
    if missing:
        raise BaselineControllerError(
            f"Phase 8C runtime payload is incomplete: {', '.join(missing)}"
        )
    return files


def package_payload(wheel: Path, suite_root: Path, oracle: Path) -> tuple[bytes, str]:
    """Package only locked runtime code, derived suite, Oracle proof, and clean wheel."""

    files = _payload_files(wheel)
    if not oracle.is_file():
        raise BaselineControllerError("Phase 8C requires a matching local Oracle suite cache")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in files:
            if path == wheel:
                archive.add(path, arcname=f"wheel/{path.name}")
            else:
                archive.add(path, arcname=str(path.relative_to(REPOSITORY_ROOT)))
        archive.add(suite_root / "tasks", arcname="dataset/tasks")
        archive.add(suite_root / "execution-suite.json", arcname="dataset/execution-suite.json")
        archive.add(oracle, arcname="oracle-suite-check.json")
    payload = stream.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_evidence(ssm_client: Any, instance_id: str, run_id: str, destination: Path) -> None:
    phase8a.collect_evidence(ssm_client, instance_id, run_id, destination)
    remote_dir = f"{phase8a.REMOTE_ROOT}/{run_id}/run"
    for name in EVIDENCE_FILES:
        target = destination / name
        if target.is_file():
            continue
        try:
            payload = phase8a._read_remote_file(ssm_client, instance_id, f"{remote_dir}/{name}")
            target.write_bytes(payload)
            phase8a.redacted_copy(target, target)
        except phase8a.SmokeError:
            continue


def write_summary(
    destination: Path, outputs: dict[str, Any], remote: phase8a.SsmResult | None
) -> dict[str, Any]:
    def load(name: str) -> dict[str, Any]:
        path = destination / name
        if not path.is_file():
            return {}
        try:
            return phase8a.load_json_object(path, name)
        except ValueError:
            return {}

    status, versions = load("run-status.json"), load("versions.json")
    evaluation, suite = load("evaluation-result.json"), load("execution-suite.json")
    oracle, candidates = load("oracle-suite-check.json"), load("phase9-candidates.json")
    revision, clean = phase8a.git_provenance()
    task_count, attempts = evaluation.get("task_count"), evaluation.get("attempts_per_task")
    expected = evaluation.get("expected_rollouts")
    passed = (
        (remote is None or remote.succeeded)
        and status.get("evaluation") == "PASS"
        and evaluation.get("status") == "PASS"
        and evaluation.get("training_performed") is False
        and evaluation.get("global_step") == 0
        and evaluation.get("optimizer_created") is False
        and evaluation.get("checkpoint_emitted") is False
        and evaluation.get("valid_rollouts") == expected
        and evaluation.get("infrastructure_invalid_rollouts") == 0
        and oracle.get("status") == "PASS"
        and oracle.get("execution_suite_sha256") == suite.get("execution_suite_sha256")
    )
    value: dict[str, Any] = {
        "schema_version": "1",
        "phase": "8C",
        "status": "PASS" if passed else "FAIL",
        "run_type": evaluation.get("run_type", "canonical"),
        "canonical_acceptance": passed and clean and evaluation.get("run_type") == "canonical",
        "git_revision": revision,
        "worktree_clean": clean,
        "instance_type": phase8a.terraform_output_value(outputs, "instance_type"),
        "ami_id": phase8a.terraform_output_value(outputs, "ami_id"),
        "availability_zone": phase8a.terraform_output_value(outputs, "availability_zone"),
        "gpu": versions.get("gpu"),
        "model_id": evaluation.get("model_id", "Qwen/Qwen3-0.6B"),
        "trl_version": versions.get("trl"),
        "vllm_version": versions.get("vllm"),
        "harbor_version": versions.get("harbor"),
        "training_performed": evaluation.get("training_performed", False),
        "global_step": evaluation.get("global_step", 0),
        "optimizer_created": evaluation.get("optimizer_created", False),
        "checkpoint_emitted": evaluation.get("checkpoint_emitted", False),
        "task_count": task_count,
        "suite_task_count": evaluation.get("suite_task_count"),
        "evaluated_task_count": evaluation.get("evaluated_task_count"),
        "evaluated_task_ids": evaluation.get("evaluated_task_ids", []),
        "attempts_per_task": attempts,
        "expected_rollouts": expected,
        "valid_rollouts": evaluation.get("valid_rollouts", 0),
        "infrastructure_invalid_rollouts": evaluation.get("infrastructure_invalid_rollouts", 0),
        "infrastructure_retry_count": evaluation.get("infrastructure_retry_count", 0),
        "base_suite_sha256": suite.get("base_suite_sha256"),
        "execution_suite_sha256": suite.get("execution_suite_sha256"),
        "overall_pass_rate": evaluation.get("overall_pass_rate"),
        "observed_always_fail_tasks": evaluation.get("observed_always_fail_tasks"),
        "observed_mixed_tasks": evaluation.get("observed_mixed_tasks"),
        "observed_always_pass_tasks": evaluation.get("observed_always_pass_tasks"),
        "phase9_candidate_threshold_met": candidates.get("phase9_candidate_threshold_met", False),
        "phase9_candidate_selection_requires_human_review": True,
        "wall_time_seconds": evaluation.get("wall_time_seconds"),
        "rollouts_per_second": evaluation.get("rollouts_per_second"),
    }
    if not passed:
        text = "\n".join((remote.stdout, remote.stderr)) if remote else ""
        value["failure_phase"] = evaluation.get("failure_phase", status.get("stage", ""))
        value["failure_class"] = evaluation.get(
            "failure_class", phase8a.classify_failure(text, str(value["failure_phase"]))
        )
    _write(destination / "summary.json", value)
    return value


def run_baseline(outputs_path: Path = OUTPUTS, requested_run_id: str | None = None) -> Path:
    outputs = phase8a.load_json_object(outputs_path, "RL smoke Terraform outputs")
    instance_id = phase8a.terraform_output_value(outputs, "instance_id")
    selection = _task_selection()
    attempts_per_pass = _positive_env("PHASE8C_ATTEMPTS_PER_PASS", 4)
    if attempts_per_pass != 4:
        raise BaselineControllerError("PHASE8C_ATTEMPTS_PER_PASS must equal 4")
    passes = _positive_env("PHASE8C_PASSES", 1)
    run_id = _run_id(requested_run_id, followup=selection is not None or passes != 1)
    destination = ROOT / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    _write(ROOT / "latest-run.json", {"run_id": run_id})
    retries = _nonnegative_env("PHASE8C_INFRA_RETRIES", 2)
    seed = _positive_env("PHASE8C_SEED", 20260920)
    suite = derive_execution_suite(destination / "execution-suite")
    _write(destination / "execution-suite.json", suite.manifest)
    cache = ROOT / "oracle-cache" / f"{suite.execution_suite_sha256}.json"
    if not load_oracle_cache(
        cache, suite.execution_suite_sha256, int(suite.manifest["task_count"])
    ):
        raise BaselineControllerError(
            "matching Phase 8C Oracle suite evidence is missing; "
            "run make rl-harbor-baseline-suite-check"
        )
    _write(destination / "oracle-suite-check.json", json.loads(cache.read_text()))
    wheel, package = build_clean_wheel(ROOT / "package")
    payload, digest = package_payload(wheel, suite.root, cache)
    _write(
        destination / "identity.json",
        {
            **package.model_dump(),
            "wheel_sha256": package.sha256,
            "payload_sha256": digest,
            **suite.manifest,
        },
    )
    sts, ec2, ssm = phase8a._aws_clients()
    phase8a.require_account(sts)
    phase8a.wait_for_instance(ec2, instance_id)
    phase8a.wait_for_ssm(ssm, instance_id)
    remote_dir = phase8a.upload_payload(ssm, instance_id, run_id, payload, digest)
    task_argument = ",".join(sorted(selection or set()))
    command = (
        f"PHASE8C_RUN_ID={run_id} PHASE8C_ATTEMPTS_PER_PASS={attempts_per_pass} "
        f"PHASE8C_PASSES={passes} PHASE8C_TASKS={task_argument} PHASE8C_SEED={seed} "
        f"PHASE8C_INFRA_RETRIES={retries} bash {remote_dir}/rl/phase8c/remote_run.sh "
        f"--run-root {remote_dir}/run"
    )
    result = phase8a.run_shell(ssm, instance_id, [command], timeout=7200)
    try:
        collect_evidence(ssm, instance_id, run_id, destination)
    finally:
        summary = write_summary(destination, outputs, result)
    if not result.succeeded or summary["status"] != "PASS":
        raise BaselineControllerError(f"Phase 8C baseline failed; evidence: {destination}")
    return destination


def evidence(requested_run_id: str | None = None) -> Path:
    if requested_run_id:
        run_id = phase8a.validate_run_id(requested_run_id)
    else:
        latest = phase8a.load_json_object(ROOT / "latest-run.json", "latest Phase 8C run")
        run_id = str(latest["run_id"])
    path = ROOT / "runs" / run_id / "summary.json"
    if not path.is_file():
        raise BaselineControllerError(
            "Phase 8C evidence is missing; run aws-rl-harbor-baseline-run first"
        )
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
        print(run_baseline(args.outputs, args.run_id))
    else:
        evidence(args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
