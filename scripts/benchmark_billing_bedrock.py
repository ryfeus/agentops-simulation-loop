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


def run(*, catalog: Path, candidate_config: Path, role_arn: str, output: Path) -> dict[str, object]:
    config = AgentConfig.model_validate_json(candidate_config.read_text())
    if config.model.provider != "bedrock":
        raise ValueError("Bedrock benchmark candidate config must use model.provider='bedrock'")
    output.mkdir(parents=True, exist_ok=True)
    rendered = output / "tasks"
    entries = render_catalog_sync(catalog, rendered)
    derived = output / "bedrock-tasks"
    for entry in entries:
        derive_bedrock_task(rendered / entry.scenario.id, derived / entry.scenario.id)

    from scripts.aws_context import expected_account_id, verified_session

    session = verified_session()
    if role_arn.split(":")[4] != expected_account_id():
        raise ValueError("Bedrock role must belong to the expected account")

    credentials = assume_bedrock_role(
        session.client("sts"), role_arn=role_arn, duration_seconds=3600
    )
    wheel, package = build_clean_wheel(output / "package")
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="agentops-bedrock-") as temporary:
        credentials_path = Path(temporary) / "credentials"
        with os.fdopen(
            os.open(credentials_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
        ) as stream:
            stream.write(credentials.credentials_file())
        for entry in entries:
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
                "--n-attempts",
                "1",
                "--n-concurrent",
                "1",
                "--yes",
                "--job-name",
                entry.scenario.id,
                "--jobs-dir",
                str(job.parent),
            ]
            import subprocess

            completed = subprocess.run(command, check=False)
            rows.append({"task_id": entry.scenario.id, "returncode": completed.returncode})
    manifest = {
        "schema_version": "1",
        "run_id": output.name,
        "candidate_fingerprint": config.fingerprint(),
        "model": f"bedrock/{config.model.model_id}",
        "package_sha256": package.sha256,
        "candidate_config_sha256": sha256_file(candidate_config),
        "task_ids": [entry.scenario.id for entry in entries],
        "execution_profile": "bedrock",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return {"manifest": manifest, "results": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path(".artifacts/benchmarks/billing-bedrock") / time.strftime(
        "%Y%m%d%H%M%S"
    )
    print(
        json.dumps(
            run(
                catalog=args.catalog,
                candidate_config=args.candidate_config,
                role_arn=args.role_arn,
                output=output,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
