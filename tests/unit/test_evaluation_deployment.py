from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_evaluator_lambda import build_package
from scripts.evaluation_deployment import (
    hydrate_deployment,
    prepare_deployment,
    validate_evaluation_deployment,
)
from scripts.generate_agentcore_config import production_config


def prepared(tmp_path: Path) -> dict[str, Path]:
    revision = "a" * 40
    config = production_config(revision, "model")
    paths = {
        "config": tmp_path / "config.json",
        "runtime": tmp_path / "runtime.json",
        "package": tmp_path / "evaluator.zip",
        "package_manifest": tmp_path / "package.json",
        "evidence": tmp_path / "evidence.json",
        "tfvars": tmp_path / "evaluation.tfvars.json",
        "manifest": tmp_path / "manifest.json",
    }
    paths["config"].write_text(config.model_dump_json())
    paths["runtime"].write_text(
        json.dumps(
            {
                "source_revision": revision,
                "agent_config": {"fingerprint": config.fingerprint()},
            }
        )
    )
    build_package(
        output=paths["package"],
        manifest=paths["package_manifest"],
        revision=revision,
    )
    paths["evidence"].write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source_revision": revision,
                "agent_config_fingerprint": config.fingerprint(),
                "session_id": "session",
                "trace_id": "trace",
                "service_name": "agentops_demo_dev.DEFAULT",
                "trace_log_group": "/aws/runtime",
                "instrumentation_scope": "openinference.instrumentation.langchain",
                "required_tool": "get_invoice",
            }
        )
    )
    prepare_deployment(
        config_path=paths["config"],
        runtime_manifest_path=paths["runtime"],
        package_path=paths["package"],
        package_manifest_path=paths["package_manifest"],
        evidence_path=paths["evidence"],
        tfvars_path=paths["tfvars"],
        manifest_path=paths["manifest"],
        outputs={
            "agentcore_runtime_arn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/id",
            "agentcore_runtime_service_name": "agentops_demo_dev.DEFAULT",
            "agentcore_trace_log_group": "/aws/runtime",
        },
    )
    return paths


def validate(paths: dict[str, Path]) -> None:
    validate_evaluation_deployment(
        tfvars_path=paths["tfvars"],
        manifest_path=paths["manifest"],
        config_path=paths["config"],
        runtime_manifest_path=paths["runtime"],
        package_path=paths["package"],
        package_manifest_path=paths["package_manifest"],
        evidence_path=paths["evidence"],
    )


def test_prepared_evaluation_is_canonical_and_valid(tmp_path: Path) -> None:
    paths = prepared(tmp_path)
    validate(paths)
    assert json.loads(paths["manifest"].read_text())["state"] == "prepared"
    assert json.loads(paths["tfvars"].read_text())["agentcore_evaluation_enabled"] is True


def test_hydrated_evaluation_requires_complete_outputs(tmp_path: Path) -> None:
    paths = prepared(tmp_path)
    outputs = {
        "dispute_policy_lambda_arn": "arn:lambda",
        "dispute_policy_evaluator_id": "evaluator-id",
        "dispute_policy_evaluator_arn": "arn:evaluator",
        "online_evaluation_config_id": "online-id",
        "online_evaluation_config_arn": "arn:online",
        "online_evaluation_execution_status": "ENABLED",
    }
    result = hydrate_deployment(manifest_path=paths["manifest"], outputs=outputs)
    assert result["state"] == "deployed"
    validate(paths)
    assert result["online_evaluation"]["result_log_group"].endswith("online-id")


@pytest.mark.parametrize("name", ["package", "package_manifest", "evidence"])
def test_missing_evaluation_input_fails_closed(tmp_path: Path, name: str) -> None:
    paths = prepared(tmp_path)
    paths[name].unlink()
    with pytest.raises(ValueError):
        validate(paths)


def test_tampered_tfvars_package_and_evidence_fail(tmp_path: Path) -> None:
    paths = prepared(tmp_path)
    paths["tfvars"].write_text(json.dumps({"agentcore_evaluation_enabled": False}))
    with pytest.raises(ValueError, match="tfvars"):
        validate(paths)

    paths = prepared(tmp_path)
    paths["package"].write_bytes(paths["package"].read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="package"):
        validate(paths)

    paths = prepared(tmp_path)
    evidence = json.loads(paths["evidence"].read_text())
    evidence["agent_config_fingerprint"] = "b" * 64
    paths["evidence"].write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="fingerprint"):
        validate(paths)
