"""Prepare and run local real-model Harbor trials without a host proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from agentops_demo.agent.prompt import SYSTEM_PROMPTS, get_system_prompt
from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.harbor.bedrock_task import derive_bedrock_task, validate_bedrock_derivation
from agentops_demo.harbor.provenance import build_clean_wheel, current_revision, sha256_file
from agentops_demo.taskify.integrity import harbor_task_sha256
from scripts.assert_harbor_result import read_trial_result, validate_runtime_provenance
from scripts.aws_context import REGION, verified_session

MODEL = "us.anthropic.claude-sonnet-4-6"
POINTER = Path(".demo/prep/harbor-direct-demo.json")


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def build_package(output: Path, *, development: bool) -> tuple[Path, dict]:
    if not development:
        wheel, provenance = build_clean_wheel(output / "package")
        return wheel, provenance.model_dump(mode="json")
    source = hashlib.sha256()
    paths = sorted(Path("src").rglob("*.py")) + [
        Path(p) for p in ("pyproject.toml", "uv.lock", "README.md", "scripts/local_harbor_demo.py")
    ]
    for path in paths:
        source.update(str(path).encode())
        source.update(path.read_bytes())
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(output / "package")], check=True)
    wheel = next((output / "package").glob("*.whl")).resolve()
    return wheel, {
        "filename": wheel.name,
        "sha256": sha256_file(wheel),
        "source_revision": f"development-{current_revision()}-{source.hexdigest()}",
        "worktree_clean": False,
        "evidence_kind": "development",
    }


def prepare(task: Path, *, development: bool, before_prompt: str = "billing-v1") -> Path:
    from aws_bedrock_token_generator import provide_token
    from botocore.credentials import EnvProvider

    get_system_prompt(before_prompt)
    session = verified_session()
    if EnvProvider().load() is None:
        raise RuntimeError("prepare through scripts/with_aws_credentials.sh or the Make target")
    task = task.resolve()
    base_digest = harbor_task_sha256(task)
    run_id = "direct-bedrock-" + datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    output = Path(".demo/prep", run_id).resolve()
    output.mkdir(parents=True)
    wheel, package = build_package(output, development=development)
    profile = derive_bedrock_task(task, output / "task", credential_mode="bedrock-bearer-token")
    validate_bedrock_derivation(task, output / "task")
    for label, version in (("before", before_prompt), ("after", "billing-v2")):
        config = AgentConfig.model_validate(
            {
                "agent": {"source_revision": package["source_revision"]},
                "model": {"provider": "bedrock", "model_id": MODEL},
                "prompt": {"version": version},
                "tools": {"version": "billing-mcp-v1"},
                "harness": {"framework": "langgraph", "version": "v1"},
            }
        )
        (output / f"{label}-config.json").write_text(config.model_dump_json(indent=2) + "\n")
    private = output / "private"
    private.mkdir(mode=0o700)
    token_path = private / "bedrock-token"
    created = datetime.now(UTC)
    token = provide_token(
        region=REGION, aws_credentials_provider=EnvProvider(), expiry=timedelta(hours=1)
    )
    try:
        with os.fdopen(os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            f.write(token)
        manifest = {
            "run_id": run_id,
            "account_id": session.client("sts").get_caller_identity()["Account"],
            "region": REGION,
            "model": "bedrock/" + MODEL,
            "source_task": str(task),
            "base_task_sha256": base_digest,
            "execution_profile": profile.model_dump(mode="json"),
            "transport": "direct HTTPS Bedrock Converse; short-lived bearer token; no proxy",
            "wheel": str(wheel),
            "package": package,
            "token_path": str(token_path),
            "token_not_after_utc": (created + timedelta(hours=1)).isoformat(),
            "token_expiry_note": "May expire earlier when the underlying AWS session expires",
            "results": {},
        }
        save(output / "comparison.json", manifest)
        POINTER.parent.mkdir(parents=True, exist_ok=True)
        save(POINTER, {"run_dir": str(output)})
    except BaseException:
        token_path.unlink(missing_ok=True)
        raise
    print(f"READY: {output}\nToken expires within one hour. Run before, after, then cleanup.")
    return output


def active_run(run_dir: Path | None) -> Path:
    return run_dir.resolve() if run_dir else Path(json.loads(POINTER.read_text())["run_dir"])


def harbor_command(output: Path, manifest: dict, label: str) -> list[str]:
    return [
        str(Path(sys.executable).parent / "harbor"),
        "run",
        "-p",
        str(output / "task"),
        "-a",
        "agentops_demo.harbor.agent:LangGraphBillingAgent",
        "-m",
        manifest["model"],
        "--env",
        "docker",
        "--ak",
        f"agent_config_path={output / (label + '-config.json')}",
        "--ak",
        "package_path=" + manifest["wheel"],
        "--ak",
        "bedrock_token_path=" + manifest["token_path"],
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--yes",
        "--job-name",
        label,
        "--jobs-dir",
        str(output / "jobs"),
    ]


def run_trial(output: Path, label: str, *, debug: bool = False) -> None:
    manifest_path = output / "comparison.json"
    manifest = json.loads(manifest_path.read_text())
    token_path = Path(manifest["token_path"])
    if not token_path.is_file() or datetime.now(UTC) >= datetime.fromisoformat(
        manifest["token_not_after_utc"]
    ):
        raise RuntimeError("Bedrock token is missing or expired; prepare a fresh demo")
    if (output / "jobs" / label).exists():
        raise RuntimeError(
            "trial already exists; prepare a fresh demo instead of overwriting evidence"
        )
    if sha256_file(Path(manifest["wheel"])) != manifest["package"]["sha256"]:
        raise RuntimeError("prepared wheel has changed")
    command = harbor_command(output, manifest, label)
    if debug:
        command.append("--debug")
    # Harbor itself needs no host AWS authority. Only the private token file is uploaded.
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    env["UV_NO_SYNC"] = "1"
    started = time.monotonic()
    row = {"command": command, "accepted": False}
    manifest["results"][label] = row
    try:
        completed = subprocess.run(command, env=env, check=False)
        row.update(
            returncode=completed.returncode, duration_seconds=round(time.monotonic() - started, 3)
        )
        if completed.returncode:
            raise RuntimeError("Harbor process failed; inspect retained job logs")
        verify_trial(output, manifest, label)
    finally:
        save(manifest_path, manifest)
    print(json.dumps(row, indent=2))


def verify_trial(output: Path, manifest: dict, label: str) -> None:
    from harbor.utils.trajectory_validator import TrajectoryValidator

    row = manifest["results"][label]
    job = output / "jobs" / label
    observed = read_trial_result("originating", job)
    row["observed"] = observed.model_dump(mode="json")
    config = AgentConfig.model_validate_json((output / f"{label}-config.json").read_text())
    validate_runtime_provenance(
        observed,
        config=config,
        package_sha256=manifest["package"]["sha256"],
        model=manifest["model"],
    )
    if observed.reward is None:
        raise RuntimeError("trial has no valid numeric reward; inspect exception.txt")
    trajectory_path = next(job.glob("*/agent/trajectory.json"))
    validator = TrajectoryValidator()
    if not validator.validate(trajectory_path):
        raise RuntimeError(f"invalid ATIF: {validator.get_errors()}")
    trajectory = json.loads(trajectory_path.read_text())
    calls = [
        {"request_id": step["extra"]["bedrock_request_id"], "tokens": step.get("metrics", {})}
        for step in trajectory["steps"]
        if step.get("extra", {}).get("bedrock_request_id")
    ]
    if not calls:
        raise RuntimeError("trajectory has no real Bedrock request IDs")
    if harbor_task_sha256(Path(manifest["source_task"])) != manifest["base_task_sha256"]:
        raise RuntimeError("source task changed")
    row.update(
        accepted=True,
        trajectory=str(trajectory_path),
        bedrock_calls=calls,
        runtime_provenance_verified=True,
    )


def cleanup(output: Path) -> None:
    path = output / "comparison.json"
    manifest = json.loads(path.read_text())
    Path(manifest["token_path"]).unlink(missing_ok=True)
    manifest["host_token_removed"] = True
    save(path, manifest)
    print("Removed the local Bedrock token. Evidence retained at " + str(output))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--task", type=Path, default=Path("benchmarks/disputed-refund"))
    prep.add_argument("--before-prompt", choices=sorted(SYSTEM_PROMPTS), default="billing-v1")
    prep.add_argument(
        "--development", action="store_true", help="Explicitly allow and label uncommitted source"
    )
    run = sub.add_parser("run")
    run.add_argument("label", choices=("before", "after"))
    run.add_argument("--debug", action="store_true", help="Enable Harbor debug logging")
    verify = sub.add_parser("verify")
    verify.add_argument("label", choices=("before", "after"))
    for command in (run, verify, sub.add_parser("cleanup")):
        command.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(args.task, development=args.development, before_prompt=args.before_prompt)
    elif args.action == "run":
        run_trial(active_run(args.run_dir), args.label, debug=args.debug)
    elif args.action == "verify":
        output = active_run(args.run_dir)
        path = output / "comparison.json"
        manifest = json.loads(path.read_text())
        verify_trial(output, manifest, args.label)
        save(path, manifest)
        print(json.dumps(manifest["results"][args.label], indent=2))
    else:
        cleanup(active_run(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
