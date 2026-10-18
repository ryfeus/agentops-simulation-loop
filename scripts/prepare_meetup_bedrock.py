"""Write the fixed three-arm configuration and shell commands without calling AWS."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from agentops_demo.benchmark.catalog import load_catalog
from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.harbor.provenance import current_revision

MODEL = "us.anthropic.claude-sonnet-4-6"
SMOKE_TASKS = ["disputed-direct-refund", "paid-refund-direct", "target-similar-invoice-ids"]


def prepare(output: Path) -> None:
    output = output.resolve()
    entries = load_catalog()
    assert len(entries) == 25 and set(SMOKE_TASKS) <= {e.scenario.id for e in entries}
    output.mkdir(parents=True, exist_ok=False)
    for arm, prompt, policy in (
        ("A", "billing-v1", "permissive"),
        ("B", "billing-v2", "permissive"),
        ("C", "billing-v2", "enforced"),
    ):
        candidate = AgentConfig.model_validate(
            {
                "agent": {"source_revision": current_revision()},
                "model": {"provider": "bedrock", "model_id": MODEL},
                "prompt": {"version": prompt},
                "tools": {"version": "billing-mcp-v1"},
                "harness": {"framework": "langgraph", "version": "v1"},
            }
        )
        path = output / f"{arm}.json"
        path.write_text(candidate.model_dump_json(indent=2) + "\n")
        for phase, attempts in (("smoke", 1), ("full", 8)):
            command = [
                "bash",
                "scripts/with_aws_credentials.sh",
                "uv",
                "run",
                "--locked",
                "--group",
                "harbor",
                "--extra",
                "aws",
                "--extra",
                "bedrock",
                "python",
                "-m",
                "scripts.benchmark_billing_bedrock",
                "--candidate-config",
                str(path),
                "--policy-mode",
                policy,
                "--attempts",
                str(attempts),
                "--output",
                str(output / f"{phase}-{arm}"),
            ]
            if phase == "smoke":
                for task in SMOKE_TASKS:
                    command.extend(["--task-id", task])
            script = output / f"{phase}-{arm}.sh"
            script.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                ': "${EXPECTED_AWS_ACCOUNT_ID:?Set the verified account ID}"\n'
                ': "${HARBOR_BEDROCK_ROLE_ARN:?Set the existing inference-only role ARN}"\n'
                + shlex.join(command)
                + ' --role-arn "$HARBOR_BEDROCK_ROLE_ARN"\n'
            )
    (output / "experiment.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "smoke_tasks": SMOKE_TASKS,
                "smoke_rollouts": 9,
                "full_rollouts": 600,
                "temperature": 0.0,
                "max_tokens": 1024,
                "sdk_total_attempts": 1,
                "concurrency": 1,
                "provider_seed": None,
                "harbor_retries": 0,
                "status": "prepared; no inference executed",
                "recreate_after_commit": True,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
