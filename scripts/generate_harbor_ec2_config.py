"""Materialize a deterministic Harbor 0.22 EC2 JobConfig and non-secret run manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentops_demo.scale.contracts import Ec2ExecutionConfig
from scripts.harbor_ec2 import (
    DEFAULT_ROOT,
    EXPECTED_REGION,
    environment_instance_type,
    environment_int,
    load_json_object,
    phase_tags,
    read_ssh_public_key,
    terraform_output_value,
    validate_run_id,
)

DEFAULT_OUTPUTS = DEFAULT_ROOT / "terraform-outputs.json"
# Harbor's task verifier timeout defaults to 60 seconds.  A separately provisioned
# EC2 verifier needs enough time to install Docker on an empty ephemeral Ubuntu
# host before the no-network verifier itself can begin.
EC2_VERIFIER_TIMEOUT_MULTIPLIER = 12.0


def load_terraform_outputs(path: Path = DEFAULT_OUTPUTS) -> dict[str, Any]:
    """Read a captured `terraform output -json` payload, never Terraform state directly."""

    return load_json_object(path, "Harbor EC2 Terraform outputs")


def execution_config(
    *,
    outputs: Mapping[str, Any],
    run_id: str,
    task_source: str,
    environ: Mapping[str, str] | None = None,
) -> Ec2ExecutionConfig:
    env = environ or os.environ
    if task_source not in {"taskify", "static"}:
        raise ValueError("task source must be taskify or static")
    private_key = env.get("HARBOR_EC2_SSH_PRIVATE_KEY", "")
    if not private_key:
        raise ValueError("HARBOR_EC2_SSH_PRIVATE_KEY must be set")
    # Validate both halves of the local keypair before emitting a JobConfig.
    read_ssh_public_key(Path(private_key).expanduser())
    region = terraform_output_value(outputs, "aws_region")
    if region != EXPECTED_REGION:
        raise ValueError(f"Terraform output aws_region must be {EXPECTED_REGION}")
    return Ec2ExecutionConfig(
        vpc_id=terraform_output_value(outputs, "harbor_vpc_id"),
        subnet_id=terraform_output_value(outputs, "harbor_subnet_id"),
        security_group_id=terraform_output_value(outputs, "harbor_security_group_id"),
        key_name=terraform_output_value(outputs, "harbor_key_name"),
        ami_id=terraform_output_value(outputs, "harbor_ami_id"),
        instance_type=environment_instance_type(env),
        attempts=environment_int("HARBOR_EC2_ATTEMPTS", default=1, environ=env),
        concurrency=environment_int("HARBOR_EC2_CONCURRENCY", default=1, environ=env),
        run_id=validate_run_id(run_id),
        task_source=task_source,
        tags=phase_tags(run_id),
    )


def job_config(
    *,
    execution: Ec2ExecutionConfig,
    task: Path,
    jobs_dir: Path,
    job_name: str,
    model: str,
    agent_config_path: Path | None = None,
    package_path: Path | None = None,
    ssh_key_path: Path,
) -> dict[str, Any]:
    """Return the single Harbor JobConfig source of EC2 launch truth.

    Credentials belong to the controller process; neither this config's agent nor
    the generated task receives AWS, Bedrock, DSQL, or production credentials.
    """

    kwargs: dict[str, Any] = {
        "region": execution.region,
        "ami_id": execution.ami_id,
        "instance_type": execution.instance_type,
        "subnet_id": execution.subnet_id,
        "security_group_ids": [execution.security_group_id],
        "key_name": execution.key_name,
        "ssh_key_path": str(ssh_key_path.expanduser().resolve()),
        "ssh_user": execution.ssh_user,
        "launch_mode": execution.launch_mode,
        "use_public_ip": execution.use_public_ip,
        "root_volume_type": execution.root_volume_type,
        "root_volume_size_gb": execution.root_volume_size_gb,
        "bootstrap_docker": execution.bootstrap_docker,
        "tags": execution.tags,
    }
    agent: dict[str, Any]
    if agent_config_path is None:
        agent = {"name": "oracle", "model_name": model}
    else:
        if package_path is None:
            raise ValueError("installed EC2 agent requires a package path")
        agent = {
            "import_path": "agentops_demo.harbor.agent:LangGraphBillingAgent",
            "model_name": model,
            "kwargs": {
                "agent_config_path": str(agent_config_path.resolve()),
                "package_path": str(package_path.resolve()),
            },
        }
    return {
        "job_name": job_name,
        "jobs_dir": str(jobs_dir.resolve()),
        "n_attempts": execution.attempts,
        "n_concurrent_trials": execution.concurrency,
        "verifier_timeout_multiplier": EC2_VERIFIER_TIMEOUT_MULTIPLIER,
        "environment": {"type": "ec2", "delete": True, "kwargs": kwargs},
        "tasks": [{"path": str(task.resolve())}],
        "agents": [agent],
    }


def write_config(
    *,
    output: Path,
    execution: Ec2ExecutionConfig,
    task: Path,
    jobs_dir: Path,
    job_name: str,
    model: str,
    agent_config_path: Path | None,
    package_path: Path | None,
    ssh_key_path: Path,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = job_config(
        execution=execution,
        task=task,
        jobs_dir=jobs_dir,
        job_name=job_name,
        model=model,
        agent_config_path=agent_config_path,
        package_path=package_path,
        ssh_key_path=ssh_key_path,
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def capture_outputs(output: Path) -> Path:
    result = subprocess.run(
        ("terraform", "-chdir=infra/terraform", "output", "-json"),
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("Terraform output must be a JSON object")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--capture-outputs", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-source", choices=("taskify", "static"), required=True)
    parser.add_argument("--agent-config-path", type=Path)
    parser.add_argument("--package-path", type=Path)
    args = parser.parse_args(argv)
    if args.capture_outputs:
        capture_outputs(args.outputs)
    execution = execution_config(
        outputs=load_terraform_outputs(args.outputs),
        run_id=args.run_id,
        task_source=args.task_source,
    )
    private_key = Path(os.environ["HARBOR_EC2_SSH_PRIVATE_KEY"])
    write_config(
        output=args.output,
        execution=execution,
        task=args.task,
        jobs_dir=args.jobs_dir,
        job_name=args.job_name,
        model=args.model,
        agent_config_path=args.agent_config_path,
        package_path=args.package_path,
        ssh_key_path=private_key,
    )
    manifest = args.output.parent / "execution.json"
    manifest.write_text(
        json.dumps(execution.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
