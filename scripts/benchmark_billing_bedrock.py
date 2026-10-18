"""Opt-in Harbor execution of the billing catalog against one Bedrock candidate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from agentops_demo.benchmark.catalog import DEFAULT_CATALOG, render_catalog_sync
from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.harbor.bedrock_credentials import assume_bedrock_role
from agentops_demo.harbor.bedrock_task import derive_bedrock_task
from agentops_demo.harbor.provenance import build_clean_wheel, sha256_file
from scripts.billing_bedrock_results import collect, summarize


def run(
    *,
    catalog: Path,
    candidate_config: Path,
    role_arn: str,
    output: Path,
    attempts: int = 1,
    policy_mode: str = "permissive",
    task_ids: list[str] | None = None,
) -> dict[str, object]:
    if attempts < 1 or policy_mode not in {"permissive", "enforced"}:
        raise ValueError("invalid attempts or policy mode")
    config = AgentConfig.model_validate_json(candidate_config.read_text())
    if config.model.provider != "bedrock":
        raise ValueError("Bedrock benchmark candidate config must use model.provider='bedrock'")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rendered = output / "tasks"
    entries = render_catalog_sync(catalog, rendered)
    if task_ids:
        if len(set(task_ids)) != len(task_ids) or not set(task_ids) <= {
            e.scenario.id for e in entries
        }:
            raise ValueError("unknown or duplicate task IDs")
        entries = [e for e in entries if e.scenario.id in task_ids]
    derived = output / "bedrock-tasks"
    for entry in entries:
        derive_bedrock_task(rendered / entry.scenario.id, derived / entry.scenario.id)

    from scripts.aws_context import expected_account_id, verified_session

    session = verified_session()
    if role_arn.split(":")[4] != expected_account_id():
        raise ValueError("Bedrock role must belong to the expected account")

    wheel, package = build_clean_wheel(output / "package")
    if config.agent.source_revision != package.source_revision:
        raise ValueError("candidate source revision does not match clean package")
    manifest = {
        "schema_version": "2",
        "run_id": output.name,
        "candidate_fingerprint": config.fingerprint(),
        "model": f"bedrock/{config.model.model_id}",
        "prompt": config.prompt.version,
        "policy_mode": policy_mode,
        "package_sha256": package.sha256,
        "source_revision": package.source_revision,
        "candidate_config_sha256": sha256_file(candidate_config),
        "task_ids": [e.scenario.id for e in entries],
        "attempts": attempts,
        "task_source_hashes": {e.scenario.id: sha256_file(e.path) for e in entries},
        "execution_profile": "bedrock",
        "concurrency": 1,
        "inference": {"temperature": 0.0, "max_tokens": 1024, "provider_seed": None},
        "sdk_total_attempts": 1,
        "harbor_retries": 0,
        "region": "us-west-2",
        "account": role_arn.split(":")[4],
        "scope": "historical regression catalog",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="agentops-bedrock-") as temporary:
        credentials_path = Path(temporary) / "credentials"
        with os.fdopen(
            os.open(credentials_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
        ) as stream:
            stream.write("")
        for entry in entries:
            credentials = assume_bedrock_role(
                session.client("sts"), role_arn=role_arn, duration_seconds=3600
            )
            credentials_path.write_text(credentials.credentials_file())
            job = output / "jobs" / entry.scenario.id
            command = [
                "uv",
                "run",
                "--group",
                "harbor",
                "harbor",
                "run",
                "-p",
                str(derived / entry.scenario.id),
                "-a",
                "agentops_demo.harbor.agent:LangGraphBillingAgent",
                "-m",
                f"bedrock/{config.model.model_id}",
                "--env",
                "docker",
                "--ak",
                f"agent_config_path={candidate_config.resolve()}",
                "--ak",
                f"package_path={wheel}",
                "--ak",
                f"bedrock_credentials_path={credentials_path}",
                "--ak",
                f"billing_policy_mode={policy_mode}",
                "--ak",
                "inference_temperature=0",
                "--ak",
                "inference_max_tokens=1024",
                "--ak",
                "inference_total_attempts=1",
                "--n-attempts",
                str(attempts),
                "--n-concurrent",
                "1",
                "--max-retries",
                "0",
                "--yes",
                "--job-name",
                entry.scenario.id,
                "--jobs-dir",
                str(job.parent),
            ]
            import subprocess

            completed = subprocess.run(command, check=False)
            rows.extend(
                collect(job, entry.scenario.model_dump(mode="json"), attempts, completed.returncode)
            )
            (output / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
            if completed.returncode or any(r["infrastructure_failure"] for r in rows):
                break
    summary = summarize(rows)
    summary["planned_rollouts"] = len(entries) * attempts
    summary["wall_seconds"] = time.perf_counter() - started
    summary["successful_rollouts_per_minute"] = summary["successful_rollouts"] / (
        summary["wall_seconds"] / 60
    )
    summary["complete"] = len(rows) == len(entries) * attempts
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return {"manifest": manifest, "results": rows, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--policy-mode", choices=["permissive", "enforced"], default="permissive")
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args(argv)
    output = args.output or Path(".artifacts/benchmarks/billing-bedrock") / time.strftime(
        "%Y%m%d%H%M%S"
    )
    result = run(
        catalog=args.catalog,
        candidate_config=args.candidate_config,
        role_arn=args.role_arn,
        output=output,
        attempts=args.attempts,
        policy_mode=args.policy_mode,
        task_ids=args.task_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    summary = result["summary"]
    return int(
        bool(
            summary["infrastructure_failures"]
            or summary["child_process_failures"]
            or not summary["complete"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
