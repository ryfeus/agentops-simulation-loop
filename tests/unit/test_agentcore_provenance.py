from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentops_demo.contracts.agent_config import AgentConfig
from scripts.generate_agentcore_config import production_config
from scripts.generate_agentcore_tfvars import deployment_payload, write_tfvars


def manifest_for(config: AgentConfig) -> dict[str, object]:
    digest = "sha256:" + "b" * 64
    repository = "123456789012.dkr.ecr.us-west-2.amazonaws.com/agentops-demo"
    return {
        "source_revision": config.agent.source_revision,
        "agent_config": {"fingerprint": config.fingerprint()},
        "container": {
            "digest": digest,
            "repository": repository,
            "uri": f"{repository}@{digest}",
        },
    }


def test_production_config_has_authoritative_runtime_identity() -> None:
    config = production_config("a" * 40, "us.anthropic.claude-sonnet-4-6")
    assert config.agent.source_revision == "a" * 40
    assert config.model.provider == "bedrock"
    assert config.prompt.version == "billing-v1"
    assert config.tools.version == "billing-mcp-v1"
    assert config.harness.framework == "langgraph"


def test_tfvars_bind_config_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = production_config("a" * 40, "model")
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "deployment.json"
    config_path.write_text(config.model_dump_json())
    manifest_path.write_text(json.dumps(manifest_for(config)))
    monkeypatch.setattr("scripts.generate_agentcore_tfvars.CONFIG_PATH", config_path)
    monkeypatch.setattr("scripts.generate_agentcore_tfvars.MANIFEST_PATH", manifest_path)
    written = write_tfvars(output)
    value = json.loads(written.read_text())
    assert value["agent_image_uri"].endswith("@sha256:" + "b" * 64)
    assert AgentConfig.model_validate_json(value["agent_config_json"]) == config


def test_tfvars_reject_fingerprint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = production_config("a" * 40, "model")
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    config_path.write_text(config.model_dump_json())
    manifest = manifest_for(config)
    manifest["agent_config"] = {"fingerprint": "c" * 64}
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr("scripts.generate_agentcore_tfvars.CONFIG_PATH", config_path)
    monkeypatch.setattr("scripts.generate_agentcore_tfvars.MANIFEST_PATH", manifest_path)
    with pytest.raises(ValueError, match="fingerprints differ"):
        write_tfvars(tmp_path / "out.json")


def test_deployment_payload_rejects_mismatched_immutable_uri() -> None:
    config = production_config("a" * 40, "model")
    manifest = manifest_for(config)
    container = manifest["container"]
    assert isinstance(container, dict)
    container["uri"] = "123456789012.dkr.ecr.us-west-2.amazonaws.com/other@sha256:" + "b" * 64
    with pytest.raises(ValueError, match="does not match"):
        deployment_payload(config, manifest)


def test_deployment_payload_rejects_non_ecr_uri() -> None:
    config = production_config("a" * 40, "model")
    manifest = manifest_for(config)
    container = manifest["container"]
    assert isinstance(container, dict)
    container["repository"] = "example.invalid/repository"
    container["uri"] = "example.invalid/repository@sha256:" + "b" * 64
    with pytest.raises(ValueError, match="immutable image URI is invalid"):
        deployment_payload(config, manifest)
