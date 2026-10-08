from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_agentcore_config import production_config
from scripts.generate_agentcore_tfvars import deployment_payload
from scripts.terraform_deployment import (
    deployment_var_files,
    terraform_var_file_args,
    validate_harbor_ec2_tfvars,
)


def valid_deployment(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config = production_config("a" * 40, "model")
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    tfvars_path = tmp_path / "deployment.auto.tfvars.json"
    foundation_path = tmp_path / "dev.tfvars"
    digest = "sha256:" + "b" * 64
    repository = "123456789012.dkr.ecr.us-west-2.amazonaws.com/agentops-demo"
    manifest = {
        "source_revision": config.agent.source_revision,
        "agent_config": {"fingerprint": config.fingerprint()},
        "container": {
            "repository": repository,
            "digest": digest,
            "uri": f"{repository}@{digest}",
        },
    }
    config_path.write_text(config.model_dump_json())
    manifest_path.write_text(json.dumps(manifest))
    tfvars_path.write_text(json.dumps(deployment_payload(config, manifest)))
    foundation_path.write_text("")
    return tfvars_path, manifest_path, config_path, foundation_path


def select(paths: tuple[Path, Path, Path, Path]) -> list[Path]:
    tfvars, manifest, config, foundation = paths
    return deployment_var_files(
        deployment_tfvars=tfvars,
        manifest=manifest,
        config=config,
        foundation_tfvars=foundation,
        evaluation_tfvars=tfvars.parent / "missing-evaluation.tfvars.json",
        evaluation_manifest=tfvars.parent / "missing-evaluation-manifest.json",
        harbor_ec2_tfvars=tfvars.parent / "missing-harbor-ec2.tfvars.json",
    )


def test_no_generated_deployment_state_selects_foundation_only(tmp_path: Path) -> None:
    foundation = tmp_path / "dev.tfvars"
    config = tmp_path / "config.json"
    config.write_text("config alone is intentionally ignored")
    assert deployment_var_files(
        deployment_tfvars=tmp_path / "missing-tfvars.json",
        manifest=tmp_path / "missing-manifest.json",
        config=config,
        foundation_tfvars=foundation,
        evaluation_tfvars=tmp_path / "missing-evaluation.tfvars.json",
        evaluation_manifest=tmp_path / "missing-evaluation-manifest.json",
        harbor_ec2_tfvars=tmp_path / "missing-harbor-ec2.tfvars.json",
    ) == [foundation]


def test_valid_deployment_state_preserves_runtime(tmp_path: Path) -> None:
    paths = valid_deployment(tmp_path)
    assert select(paths) == [paths[3], paths[0].resolve()]
    assert terraform_var_file_args(
        deployment_tfvars=paths[0],
        manifest=paths[1],
        config=paths[2],
        foundation_tfvars=paths[3],
        evaluation_tfvars=paths[0].parent / "missing-evaluation.tfvars.json",
        evaluation_manifest=paths[0].parent / "missing-evaluation-manifest.json",
        harbor_ec2_tfvars=paths[0].parent / "missing-harbor-ec2.tfvars.json",
    ) == [f"-var-file={paths[3]}", f"-var-file={paths[0].resolve()}"]


@pytest.mark.parametrize("missing_index", [0, 1, 2])
def test_incomplete_deployment_state_fails(tmp_path: Path, missing_index: int) -> None:
    paths = valid_deployment(tmp_path)
    paths[missing_index].unlink()
    with pytest.raises(ValueError, match="incomplete AgentCore deployment state"):
        select(paths)


def test_source_revision_mismatch_fails(tmp_path: Path) -> None:
    paths = valid_deployment(tmp_path)
    manifest = json.loads(paths[1].read_text())
    manifest["source_revision"] = "c" * 40
    paths[1].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source revisions differ"):
        select(paths)


def test_fingerprint_mismatch_fails(tmp_path: Path) -> None:
    paths = valid_deployment(tmp_path)
    manifest = json.loads(paths[1].read_text())
    manifest["agent_config"]["fingerprint"] = "c" * 64
    paths[1].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprints differ"):
        select(paths)


def test_invalid_image_digest_fails(tmp_path: Path) -> None:
    paths = valid_deployment(tmp_path)
    manifest = json.loads(paths[1].read_text())
    manifest["container"]["digest"] = "sha256:not-a-digest"
    paths[1].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="image digest is invalid"):
        select(paths)


def test_noncanonical_tfvars_fail(tmp_path: Path) -> None:
    paths = valid_deployment(tmp_path)
    tfvars = json.loads(paths[0].read_text())
    tfvars["agent_image_uri"] = tfvars["agent_image_uri"].replace("agentops-demo", "other")
    paths[0].write_text(json.dumps(tfvars))
    with pytest.raises(ValueError, match="tfvars do not match"):
        select(paths)


def test_malformed_generated_file_fails(tmp_path: Path) -> None:
    paths = valid_deployment(tmp_path)
    paths[0].write_text("not-json")
    with pytest.raises(ValueError, match="cannot read AgentCore deployment tfvars"):
        select(paths)


def test_partial_evaluation_deployment_state_fails(tmp_path: Path) -> None:
    paths = valid_deployment(tmp_path)
    evaluation_tfvars = tmp_path / "evaluation.tfvars.json"
    evaluation_tfvars.write_text("{}")
    with pytest.raises(ValueError, match="incomplete AgentCore evaluation state"):
        deployment_var_files(
            deployment_tfvars=paths[0],
            manifest=paths[1],
            config=paths[2],
            foundation_tfvars=paths[3],
            evaluation_tfvars=evaluation_tfvars,
            evaluation_manifest=tmp_path / "missing-manifest.json",
        )


def test_evaluation_state_without_runtime_fails(tmp_path: Path) -> None:
    evaluation_tfvars = tmp_path / "evaluation.tfvars.json"
    evaluation_manifest = tmp_path / "evaluation-manifest.json"
    evaluation_tfvars.write_text("{}")
    evaluation_manifest.write_text("{}")
    with pytest.raises(ValueError, match="requires a runtime"):
        deployment_var_files(
            deployment_tfvars=tmp_path / "missing-runtime-tfvars.json",
            manifest=tmp_path / "missing-runtime-manifest.json",
            config=tmp_path / "config.json",
            foundation_tfvars=tmp_path / "dev.tfvars",
            evaluation_tfvars=evaluation_tfvars,
            evaluation_manifest=evaluation_manifest,
        )


def test_complete_harbor_ec2_tfvars_are_composed_with_runtime_state(tmp_path: Path) -> None:
    paths = valid_deployment(tmp_path)
    harbor_tfvars = tmp_path / ".harbor-ec2" / "foundation.auto.tfvars.json"
    harbor_tfvars.parent.mkdir()
    harbor_tfvars.write_text(
        json.dumps(
            {
                "harbor_ec2_enabled": True,
                "harbor_controller_cidr": "198.51.100.10/32",
                "harbor_ssh_public_key": "ssh-ed25519 AAAA operator",
            }
        )
    )
    selected = deployment_var_files(
        deployment_tfvars=paths[0],
        manifest=paths[1],
        config=paths[2],
        foundation_tfvars=paths[3],
        evaluation_tfvars=tmp_path / "missing-evaluation.tfvars.json",
        evaluation_manifest=tmp_path / "missing-evaluation-manifest.json",
        harbor_ec2_tfvars=harbor_tfvars,
    )
    assert selected[-1] == harbor_tfvars.resolve()


def test_incomplete_harbor_ec2_tfvars_fail_closed(tmp_path: Path) -> None:
    tfvars = tmp_path / "foundation.auto.tfvars.json"
    tfvars.write_text(json.dumps({"harbor_ec2_enabled": True}))
    with pytest.raises(ValueError, match="incomplete or invalid"):
        validate_harbor_ec2_tfvars(tfvars)
