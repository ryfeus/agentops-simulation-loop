"""Select Terraform var files without discarding an existing AgentCore deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.agentcore_package import CONFIG_PATH, MANIFEST_PATH
from scripts.evaluation_deployment import (
    EVALUATION_MANIFEST,
    EVALUATION_TFVARS,
    OBSERVABILITY_EVIDENCE,
    PACKAGE_MANIFEST,
    PACKAGE_PATH,
    validate_evaluation_deployment,
)
from scripts.generate_agentcore_tfvars import DEFAULT_OUTPUT, validate_deployment_tfvars

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_TFVARS = REPOSITORY_ROOT / "infra" / "terraform" / "environments" / "dev.tfvars"
HARBOR_EC2_TFVARS = REPOSITORY_ROOT / ".harbor-ec2" / "foundation.auto.tfvars.json"


def validate_harbor_ec2_tfvars(path: Path) -> None:
    """Accept only a complete generated public-key Terraform payload."""

    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Harbor EC2 tfvars {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Harbor EC2 tfvars must be a JSON object")
    expected = {"harbor_ec2_enabled", "harbor_controller_cidr", "harbor_ssh_public_key"}
    if set(payload) != expected or payload.get("harbor_ec2_enabled") is not True:
        raise ValueError("Harbor EC2 tfvars are incomplete or invalid")
    cidr = payload.get("harbor_controller_cidr")
    public_key = payload.get("harbor_ssh_public_key")
    if not isinstance(cidr, str) or not cidr or cidr == "0.0.0.0/0":
        raise ValueError("Harbor EC2 tfvars have an invalid controller CIDR")
    if not isinstance(public_key, str) or not public_key.startswith("ssh-"):
        raise ValueError("Harbor EC2 tfvars have an invalid public key")


def deployment_var_files(
    *,
    deployment_tfvars: Path = DEFAULT_OUTPUT,
    manifest: Path = MANIFEST_PATH,
    config: Path = CONFIG_PATH,
    foundation_tfvars: Path = FOUNDATION_TFVARS,
    evaluation_tfvars: Path = EVALUATION_TFVARS,
    evaluation_manifest: Path = EVALUATION_MANIFEST,
    evaluator_package: Path = PACKAGE_PATH,
    evaluator_package_manifest: Path = PACKAGE_MANIFEST,
    observability_evidence: Path = OBSERVABILITY_EVIDENCE,
    harbor_ec2_tfvars: Path = HARBOR_EC2_TFVARS,
) -> list[Path]:
    """Return safe Terraform inputs or reject incomplete/stale deployment state."""

    tfvars_exists = deployment_tfvars.is_file()
    manifest_exists = manifest.is_file()
    selected = [foundation_tfvars]
    runtime_deployed = tfvars_exists or manifest_exists
    if tfvars_exists != manifest_exists:
        missing = manifest if tfvars_exists else deployment_tfvars
        raise ValueError(f"incomplete AgentCore deployment state: missing {missing}")
    if runtime_deployed and not config.is_file():
        raise ValueError(f"incomplete AgentCore deployment state: missing {config}")
    if runtime_deployed:
        validate_deployment_tfvars(
            deployment_tfvars=deployment_tfvars,
            manifest=manifest,
            config=config,
        )
        selected.append(deployment_tfvars.resolve())

    evaluation_tfvars_exists = evaluation_tfvars.is_file()
    evaluation_manifest_exists = evaluation_manifest.is_file()
    if evaluation_tfvars_exists != evaluation_manifest_exists:
        missing = evaluation_manifest if evaluation_tfvars_exists else evaluation_tfvars
        raise ValueError(f"incomplete AgentCore evaluation state: missing {missing}")
    if evaluation_tfvars_exists:
        if not runtime_deployed:
            raise ValueError("AgentCore evaluation state requires a runtime deployment")
        validate_evaluation_deployment(
            tfvars_path=evaluation_tfvars,
            manifest_path=evaluation_manifest,
            config_path=config,
            runtime_manifest_path=manifest,
            package_path=evaluator_package,
            package_manifest_path=evaluator_package_manifest,
            evidence_path=observability_evidence,
        )
        selected.append(evaluation_tfvars.resolve())
    if harbor_ec2_tfvars.exists():
        if not harbor_ec2_tfvars.is_file():
            raise ValueError("Harbor EC2 tfvars path is not a file")
        validate_harbor_ec2_tfvars(harbor_ec2_tfvars)
        selected.append(harbor_ec2_tfvars.resolve())
    elif harbor_ec2_tfvars.parent.is_dir() and any(
        harbor_ec2_tfvars.parent.glob("**/execution.json")
    ):
        raise ValueError("incomplete Harbor EC2 generated state: missing foundation tfvars")
    return selected


def terraform_var_file_args(**paths: Path) -> list[str]:
    return [f"-var-file={path}" for path in deployment_var_files(**paths)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(" ".join(terraform_var_file_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
