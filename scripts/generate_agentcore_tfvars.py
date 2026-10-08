"""Generate Terraform runtime inputs from verified AgentCore provenance."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.harbor.provenance import is_sha256
from scripts.agentcore_package import CONFIG_PATH, MANIFEST_PATH, load_config, load_manifest

DEFAULT_OUTPUT = Path(".agentcore/deployment.auto.tfvars.json")
ECR_IMAGE_URI_PATTERN = re.compile(
    r"^[0-9]{12}\.dkr\.ecr\.us-west-2\.amazonaws\.com/"
    r"[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$"
)


def deployment_payload(config: AgentConfig, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Create Terraform values after validating deployment provenance."""

    if manifest.get("source_revision") != config.agent.source_revision:
        raise ValueError("manifest and AgentConfig source revisions differ")
    agent_config = manifest.get("agent_config")
    if not isinstance(agent_config, Mapping):
        raise ValueError("manifest AgentConfig metadata is invalid")
    fingerprint = agent_config.get("fingerprint")
    if not isinstance(fingerprint, str) or not is_sha256(fingerprint):
        raise ValueError("manifest AgentConfig fingerprint is invalid")
    if fingerprint != config.fingerprint():
        raise ValueError("manifest and AgentConfig fingerprints differ")

    container = manifest.get("container")
    if not isinstance(container, Mapping):
        raise ValueError("manifest container metadata is invalid")
    digest = container.get("digest")
    repository = container.get("repository")
    uri = container.get("uri")
    if not isinstance(digest, str) or not (
        digest.startswith("sha256:") and is_sha256(digest.removeprefix("sha256:"))
    ):
        raise ValueError("manifest image digest is invalid")
    if not isinstance(repository, str) or not repository:
        raise ValueError("manifest image repository is invalid")
    if not isinstance(uri, str) or uri != f"{repository}@{digest}":
        raise ValueError("manifest immutable image URI does not match its repository and digest")
    if not ECR_IMAGE_URI_PATTERN.fullmatch(uri):
        raise ValueError("manifest immutable image URI is invalid")

    return {
        "agent_image_uri": uri,
        "agent_config_json": json.dumps(
            config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ),
    }


def load_tfvars(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read AgentCore deployment tfvars {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("AgentCore deployment tfvars must be a JSON object")
    return value


def validate_deployment_tfvars(*, deployment_tfvars: Path, manifest: Path, config: Path) -> None:
    """Require generated Terraform values to exactly match config and manifest."""

    expected = deployment_payload(load_config(config), load_manifest(manifest))
    if load_tfvars(deployment_tfvars) != expected:
        raise ValueError("AgentCore deployment tfvars do not match config and manifest")


def write_tfvars(output: Path = DEFAULT_OUTPUT) -> Path:
    config = load_config(CONFIG_PATH)
    manifest = load_manifest(MANIFEST_PATH)
    payload = deployment_payload(config, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(f"Wrote {write_tfvars(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
