"""SSM-only controller for the immutable-adapter Phase 9b comparison."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentops_demo.harbor.provenance import build_clean_wheel
from rl.phase8c.execution_suite import derive_execution_suite
from rl.phase8c.suite_check import load_oracle_cache
from rl.phase9.task_set import load_task_set
from rl.phase9b.adapter import validate_persisted_adapter
from scripts import rl_smoke as smoke

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY_ROOT / ".rl-smoke" / "phase9b"
OUTPUTS = REPOSITORY_ROOT / ".rl-smoke" / "terraform-outputs.json"
EVIDENCE = (
    "run-status.json",
    "versions.json",
    "evaluation-config.json",
    "evaluation-result.json",
    "comparison.json",
    "source-adapter.json",
    "gpu.csv",
    "docker.txt",
    "nvidia-smi.txt",
    "evaluation.log",
)


class Phase9bControllerError(smoke.SmokeError):
    pass


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_run() -> str:
    value = os.environ.get("PHASE9B_SOURCE_RUN", "").strip()
    if not value:
        raise Phase9bControllerError("PHASE9B_SOURCE_RUN must name a newly retained Phase 9 run")
    return smoke.validate_run_id(value)


def _run_id(requested: str | None) -> str:
    return (
        smoke.validate_run_id(requested)
        if requested
        else smoke.validate_run_id("phase9b-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S"))
    )


def _payload_files(wheel: Path) -> tuple[Path, ...]:
    required = (
        REPOSITORY_ROOT / "rl" / "pyproject.toml",
        REPOSITORY_ROOT / "rl" / "uv.lock",
        wheel,
        *sorted((REPOSITORY_ROOT / "rl" / "phase8b").glob("*.py")),
        REPOSITORY_ROOT / "rl" / "phase8c" / "__init__.py",
        REPOSITORY_ROOT / "rl" / "phase8c" / "baseline_env.py",
        REPOSITORY_ROOT / "rl" / "phase8c" / "baseline_reward.py",
        REPOSITORY_ROOT / "rl" / "phase8c" / "evaluate_baseline.py",
        REPOSITORY_ROOT / "rl" / "phase8c" / "execution_suite.py",
        REPOSITORY_ROOT / "rl" / "phase8c" / "report.py",
        REPOSITORY_ROOT / "rl" / "phase9" / "task_set.py",
        *sorted((REPOSITORY_ROOT / "rl" / "phase9b").glob("*.py")),
        REPOSITORY_ROOT / "rl" / "phase9b" / "remote_run.sh",
    )
    missing = [str(path.relative_to(REPOSITORY_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise Phase9bControllerError("Phase 9b payload is incomplete: " + ", ".join(missing))
    return required


def package_payload(
    *, wheel: Path, suite_root: Path, oracle: Path, task_set: Path, adapter: Any
) -> tuple[bytes, str]:
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
        archive.add(
            adapter.source_dir / "artifacts-manifest.json",
            arcname="source-adapter/artifacts-manifest.json",
        )
        archive.add(adapter.source_dir / "summary.json", arcname="source-adapter/summary.json")
        archive.add(adapter.task_set_path, arcname="source-adapter/training-task-set.json")
        for relative in sorted(adapter.files):
            archive.add(
                adapter.adapter_dir / relative,
                arcname=f"source-adapter/adapter/final/{relative}",
            )
        identity = adapter.value()
        identity_bytes = json.dumps(identity, sort_keys=True).encode()
        info = tarfile.TarInfo("source-adapter/identity.json")
        info.size = len(identity_bytes)
        archive.addfile(info, io.BytesIO(identity_bytes))
    payload = stream.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def collect_evidence(ssm: Any, instance_id: str, run_id: str, destination: Path) -> None:
    smoke.collect_evidence(ssm, instance_id, run_id, destination)
    remote = f"{smoke.REMOTE_ROOT}/{run_id}/run"
    for name in EVIDENCE:
        target = destination / name
        if target.is_file():
            continue
        try:
            target.write_bytes(smoke._read_remote_file(ssm, instance_id, f"{remote}/{name}"))
            smoke.redacted_copy(target, target)
        except smoke.SmokeError:
            continue
    for policy in ("baseline", "trained"):
        for name in ("rollouts.jsonl", "evaluation-result.json", "policy-config.json"):
            target = destination / policy / name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.write_bytes(
                    smoke._read_remote_file(ssm, instance_id, f"{remote}/{policy}/{name}")
                )
                smoke.redacted_copy(target, target)
            except smoke.SmokeError:
                continue


def _valid_policy(destination: Path, policy: str, expected_task_ids: set[str]) -> bool:
    try:
        value = smoke.load_json_object(
            destination / policy / "evaluation-result.json", f"{policy} result"
        )
        rows = [
            json.loads(line)
            for line in (destination / policy / "rollouts.jsonl").read_text().splitlines()
            if line
        ]
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected_attempts = set(range(16))
    attempts_by_task: dict[str, set[int]] = {task_id: set() for task_id in expected_task_ids}
    valid_rows = True
    for row in rows:
        task_id = row.get("task_id")
        reward = row.get("reward")
        attempt = row.get("attempt")
        evaluation_pass = row.get("evaluation_pass")
        pass_local_attempt = row.get("pass_local_attempt")
        if (
            row.get("policy") != policy
            or task_id not in expected_task_ids
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
            or not isinstance(attempt, int)
            or attempt not in expected_attempts
            or evaluation_pass != attempt // 4
            or pass_local_attempt != attempt % 4
            or attempt in attempts_by_task[str(task_id)]
        ):
            valid_rows = False
            break
        attempts_by_task[str(task_id)].add(attempt)
    return (
        value.get("status") == "PASS"
        and value.get("training_performed") is False
        and value.get("optimizer_created") is False
        and value.get("global_step") == 0
        and value.get("checkpoint_emitted") is False
        and len(rows) == len(expected_task_ids) * 16
        and valid_rows
        and set(attempts_by_task) == expected_task_ids
        and all(attempts == expected_attempts for attempts in attempts_by_task.values())
    )


def write_summary(
    destination: Path, outputs: dict[str, Any], remote: smoke.SsmResult | None
) -> dict[str, Any]:
    def load(name: str) -> dict[str, Any]:
        try:
            return smoke.load_json_object(destination / name, name)
        except ValueError:
            return {}

    status, result, comparison, config = (
        load("run-status.json"),
        load("evaluation-result.json"),
        load("comparison.json"),
        load("evaluation-config.json"),
    )
    revision, clean = smoke.git_provenance()
    expected_task_ids = {
        *[str(task_id) for task_id in config.get("training_task_ids", [])],
        str(config.get("control_task_id", "")),
    }
    passed = (
        (remote is None or remote.succeeded)
        and status.get("evaluation") == "PASS"
        and result.get("status") == "PASS"
        and len(expected_task_ids) == 5
        and _valid_policy(destination, "baseline", expected_task_ids)
        and _valid_policy(destination, "trained", expected_task_ids)
        and bool(comparison)
    )
    value = {
        "schema_version": "1",
        "phase": "9b",
        "status": "PASS" if passed else "FAIL",
        "canonical_acceptance": False,
        "git_revision": revision,
        "worktree_clean": clean,
        "source_run": config.get("source_run"),
        "execution_suite_sha256": config.get("execution_suite_sha256"),
        "instance_type": smoke.terraform_output_value(outputs, "instance_type"),
        "ami_id": smoke.terraform_output_value(outputs, "ami_id"),
        "availability_zone": smoke.terraform_output_value(outputs, "availability_zone"),
        "baseline_rollouts": 80 if _valid_policy(destination, "baseline", expected_task_ids) else 0,
        "trained_rollouts": 80 if _valid_policy(destination, "trained", expected_task_ids) else 0,
        "attempts_per_task_per_policy": 16,
        "training_task_count": len(config.get("training_task_ids", [])),
        "regression_control_count": 1 if config.get("control_task_id") else 0,
        "seed_schedule": [int(config.get("seed", 20260923)) + offset for offset in range(4)],
        "policy_rollouts": {
            policy: 80 if _valid_policy(destination, policy, expected_task_ids) else 0
            for policy in ("baseline", "trained")
        },
        "behavioral_improvement_supported": comparison.get(
            "behavioral_improvement_supported", False
        ),
        "catastrophic_regression_detected": comparison.get(
            "catastrophic_regression_detected", False
        ),
        "human_review_required": True,
        "comparison": comparison,
    }
    if not passed:
        value["failure_phase"] = result.get("failure_phase", status.get("stage", ""))
    _write(destination / "summary.json", value)
    return value


def run(outputs_path: Path = OUTPUTS, requested_run_id: str | None = None) -> Path:
    outputs = smoke.load_json_object(outputs_path, "RL smoke Terraform outputs")
    instance_id = smoke.terraform_output_value(outputs, "instance_id")
    source_run = _source_run()
    run_id = _run_id(requested_run_id)
    destination = ROOT / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    _write(ROOT / "latest-run.json", {"run_id": run_id})
    suite = derive_execution_suite(destination / "execution-suite")
    _write(destination / "execution-suite.json", suite.manifest)
    oracle = (
        REPOSITORY_ROOT
        / ".rl-smoke"
        / "phase8c"
        / "oracle-cache"
        / f"{suite.execution_suite_sha256}.json"
    )
    if not load_oracle_cache(
        oracle, suite.execution_suite_sha256, int(suite.manifest["task_count"])
    ):
        raise Phase9bControllerError("matching full-suite Phase 8C Oracle evidence is missing")
    _write(destination / "oracle-suite-check.json", json.loads(oracle.read_text()))
    adapter = validate_persisted_adapter(
        REPOSITORY_ROOT / ".rl-smoke" / "phase9" / "runs" / source_run,
        source_run=source_run,
        execution_suite_sha256=suite.execution_suite_sha256,
    )
    task_set_path = adapter.task_set_path
    task_set = load_task_set(task_set_path, suite.manifest)
    if len(task_set.training_ids) != 4 or len(task_set.regression_ids) != 1:
        raise Phase9bControllerError(
            "Phase 9b requires exactly four training tasks and one control"
        )
    wheel, package = build_clean_wheel(ROOT / "package")
    payload, digest = package_payload(
        wheel=wheel, suite_root=suite.root, oracle=oracle, task_set=task_set_path, adapter=adapter
    )
    _write(
        destination / "identity.json",
        {
            **package.model_dump(),
            "wheel_sha256": package.sha256,
            "payload_sha256": digest,
            **adapter.value(),
            **suite.manifest,
        },
    )
    sts, ec2, ssm = smoke._aws_clients()
    smoke.require_account(sts)
    smoke.wait_for_instance(ec2, instance_id)
    smoke.wait_for_ssm(ssm, instance_id)
    remote_dir = smoke.upload_payload(
        ssm, instance_id, run_id, payload, digest, parallel_chunks=True
    )
    result = smoke.run_shell(
        ssm,
        instance_id,
        [
            " ".join(
                (
                    f"RL_SMOKE_RUN_ID={run_id}",
                    f"PHASE9B_SOURCE_RUN={source_run}",
                    f"PHASE9B_SEED={os.environ.get('PHASE9B_SEED', '20260923')}",
                    f"bash {remote_dir}/rl/phase9b/remote_run.sh",
                    f"--run-root {remote_dir}/run",
                )
            )
        ],
        timeout=14400,
    )
    try:
        collect_evidence(ssm, instance_id, run_id, destination)
    finally:
        summary = write_summary(destination, outputs, result)
    if not result.succeeded or summary["status"] != "PASS":
        raise Phase9bControllerError(f"Phase 9b failed; evidence: {destination}")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    run_parser.add_argument("--run-id")
    evidence = sub.add_parser("evidence")
    evidence.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.command == "run":
        print(run(args.outputs, args.run_id))
    else:
        run_id = (
            smoke.validate_run_id(args.run_id)
            if args.run_id
            else str(
                smoke.load_json_object(ROOT / "latest-run.json", "latest Phase 9b run")["run_id"]
            )
        )
        print(ROOT / "runs" / run_id / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
