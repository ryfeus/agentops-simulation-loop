"""Prepare, validate, and hydrate provenance-bound Phase 4 evaluation deployment state."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.agentcore_package import CONFIG_PATH, MANIFEST_PATH, load_config, load_manifest
from scripts.build_evaluator_lambda import OUTPUT as PACKAGE_PATH
from scripts.build_evaluator_lambda import PACKAGE_MANIFEST
from scripts.dsql_admin import terraform_outputs
from scripts.find_agentcore_trace import DEFAULT_EVIDENCE as OBSERVABILITY_EVIDENCE

EVALUATION_TFVARS = Path(".agentcore/evaluation.auto.tfvars.json")
EVALUATION_MANIFEST = Path(".agentcore/evaluation/manifest.json")
EVALUATOR_NAME = "DisputePolicyCompliance"
ONLINE_NAME = "DisputePolicyOnline"


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot read evaluator package {path}: {exc}") from exc


def evaluation_tfvars(package_sha256: str) -> dict[str, object]:
    return {
        "agentcore_evaluation_enabled": True,
        "evaluator_package_sha256": package_sha256,
    }


def _validate_inputs(
    *,
    config_path: Path,
    runtime_manifest_path: Path,
    package_path: Path,
    package_manifest_path: Path,
    evidence_path: Path,
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    runtime_manifest = load_manifest(runtime_manifest_path)
    source_revision = config.agent.source_revision
    fingerprint = config.fingerprint()
    if runtime_manifest.get("source_revision") != source_revision:
        raise ValueError("evaluation and runtime source revisions differ")
    runtime_config = runtime_manifest.get("agent_config")
    if not isinstance(runtime_config, Mapping) or runtime_config.get("fingerprint") != fingerprint:
        raise ValueError("evaluation and runtime AgentConfig fingerprints differ")
    package_manifest = _load_object(package_manifest_path, "evaluator package manifest")
    package = package_manifest.get("package")
    if package_manifest.get("schema_version") != "1" or not isinstance(package, Mapping):
        raise ValueError("evaluator package manifest is invalid")
    if package_manifest.get("source_revision") != source_revision:
        raise ValueError("evaluator package source revision differs from runtime")
    digest = _sha256(package_path)
    if package.get("sha256") != digest or package.get("path") != str(package_path):
        raise ValueError("evaluator package manifest does not match package bytes")
    evidence = _load_object(evidence_path, "observability evidence")
    if evidence.get("schema_version") != "1":
        raise ValueError("observability evidence schema is unsupported")
    if evidence.get("source_revision") != source_revision:
        raise ValueError("observability evidence source revision differs from runtime")
    if evidence.get("agent_config_fingerprint") != fingerprint:
        raise ValueError("observability evidence AgentConfig fingerprint differs from runtime")
    if evidence.get("instrumentation_scope") != "openinference.instrumentation.langchain":
        raise ValueError("observability evidence has an unsupported instrumentation scope")
    if evidence.get("required_tool") != "get_invoice":
        raise ValueError("observability evidence does not prove get_invoice tracing")
    return source_revision, fingerprint, package_manifest, evidence


def prepare_deployment(
    *,
    config_path: Path = CONFIG_PATH,
    runtime_manifest_path: Path = MANIFEST_PATH,
    package_path: Path = PACKAGE_PATH,
    package_manifest_path: Path = PACKAGE_MANIFEST,
    evidence_path: Path = OBSERVABILITY_EVIDENCE,
    tfvars_path: Path = EVALUATION_TFVARS,
    manifest_path: Path = EVALUATION_MANIFEST,
    outputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_revision, fingerprint, package_manifest, evidence = _validate_inputs(
        config_path=config_path,
        runtime_manifest_path=runtime_manifest_path,
        package_path=package_path,
        package_manifest_path=package_manifest_path,
        evidence_path=evidence_path,
    )
    values = dict(outputs or terraform_outputs())
    runtime_arn = values.get("agentcore_runtime_arn")
    service_name = values.get("agentcore_runtime_service_name")
    trace_log_group = values.get("agentcore_trace_log_group")
    if not all(
        isinstance(value, str) and value for value in (runtime_arn, service_name, trace_log_group)
    ):
        raise ValueError("Terraform outputs do not identify a deployed observable runtime")
    if (
        evidence.get("service_name") != service_name
        or evidence.get("trace_log_group") != trace_log_group
    ):
        raise ValueError("observability evidence does not match deployed runtime identity")
    digest = str(package_manifest["package"]["sha256"])
    manifest = {
        "schema_version": "1",
        "state": "prepared",
        "source_revision": source_revision,
        "runtime": {
            "arn": runtime_arn,
            "service_name": service_name,
            "trace_log_group": trace_log_group,
            "agent_config_fingerprint": fingerprint,
        },
        "observability": {
            "session_id": evidence["session_id"],
            "trace_id": evidence["trace_id"],
        },
        "evaluator": {
            "name": EVALUATOR_NAME,
            "level": "TRACE",
            "lambda_zip_sha256": digest,
        },
        "online_evaluation": {
            "name": ONLINE_NAME,
            "sampling_percentage": 100.0,
            "session_timeout_minutes": 1,
        },
    }
    tfvars_path.parent.mkdir(parents=True, exist_ok=True)
    tfvars_path.write_text(json.dumps(evaluation_tfvars(digest), indent=2, sort_keys=True) + "\n")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def validate_evaluation_deployment(
    *,
    tfvars_path: Path = EVALUATION_TFVARS,
    manifest_path: Path = EVALUATION_MANIFEST,
    config_path: Path = CONFIG_PATH,
    runtime_manifest_path: Path = MANIFEST_PATH,
    package_path: Path = PACKAGE_PATH,
    package_manifest_path: Path = PACKAGE_MANIFEST,
    evidence_path: Path = OBSERVABILITY_EVIDENCE,
) -> None:
    source_revision, fingerprint, package_manifest, evidence = _validate_inputs(
        config_path=config_path,
        runtime_manifest_path=runtime_manifest_path,
        package_path=package_path,
        package_manifest_path=package_manifest_path,
        evidence_path=evidence_path,
    )
    digest = str(package_manifest["package"]["sha256"])
    if _load_object(tfvars_path, "evaluation tfvars") != evaluation_tfvars(digest):
        raise ValueError("evaluation tfvars do not match the evaluator package")
    manifest = _load_object(manifest_path, "evaluation manifest")
    if manifest.get("schema_version") != "1" or manifest.get("state") not in {
        "prepared",
        "deployed",
    }:
        raise ValueError("evaluation manifest state is invalid")
    if manifest.get("source_revision") != source_revision:
        raise ValueError("evaluation manifest source revision differs from runtime")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("agent_config_fingerprint") != fingerprint:
        raise ValueError("evaluation manifest runtime provenance is invalid")
    if not all(
        isinstance(runtime.get(key), str) and runtime[key]
        for key in (
            "arn",
            "service_name",
            "trace_log_group",
        )
    ):
        raise ValueError("evaluation manifest runtime identity is incomplete")
    observability = manifest.get("observability")
    if not isinstance(observability, Mapping) or observability != {
        "session_id": evidence.get("session_id"),
        "trace_id": evidence.get("trace_id"),
    }:
        raise ValueError("evaluation manifest observability evidence differs")
    evaluator = manifest.get("evaluator")
    if not isinstance(evaluator, Mapping) or any(
        (
            evaluator.get("name") != EVALUATOR_NAME,
            evaluator.get("level") != "TRACE",
            evaluator.get("lambda_zip_sha256") != digest,
        )
    ):
        raise ValueError("evaluation manifest evaluator identity is invalid")
    online = manifest.get("online_evaluation")
    if not isinstance(online, Mapping) or any(
        (
            online.get("name") != ONLINE_NAME,
            online.get("sampling_percentage") != 100.0,
            online.get("session_timeout_minutes") != 1,
        )
    ):
        raise ValueError("evaluation manifest online configuration is invalid")
    if manifest["state"] == "deployed":
        required = (
            "id",
            "arn",
            "lambda_arn",
            "status",
        )
        if not all(isinstance(evaluator.get(key), str) and evaluator[key] for key in required):
            raise ValueError("deployed evaluator identity is incomplete")
        if not all(
            isinstance(online.get(key), str) and online[key]
            for key in ("id", "arn", "execution_status", "result_log_group")
        ):
            raise ValueError("deployed online evaluation identity is incomplete")
    elif any(key in evaluator for key in ("id", "arn", "lambda_arn", "status")) or any(
        key in online for key in ("id", "arn", "execution_status", "result_log_group")
    ):
        raise ValueError("prepared evaluation manifest contains deployed identity")


def hydrate_deployment(
    *, manifest_path: Path = EVALUATION_MANIFEST, outputs: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    manifest = _load_object(manifest_path, "evaluation manifest")
    values = dict(outputs or terraform_outputs())
    required = {
        "lambda_arn": "dispute_policy_lambda_arn",
        "id": "dispute_policy_evaluator_id",
        "arn": "dispute_policy_evaluator_arn",
    }
    for target, source in required.items():
        value = values.get(source)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Terraform output {source} is missing")
        manifest["evaluator"][target] = value
    manifest["evaluator"]["status"] = "ACTIVE"
    for target, source in {
        "id": "online_evaluation_config_id",
        "arn": "online_evaluation_config_arn",
        "execution_status": "online_evaluation_execution_status",
    }.items():
        value = values.get(source)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Terraform output {source} is missing")
        manifest["online_evaluation"][target] = value
    manifest["online_evaluation"]["result_log_group"] = (
        "/aws/bedrock-agentcore/evaluations/results/" + manifest["online_evaluation"]["id"]
    )
    manifest["state"] = "deployed"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hydrate", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.hydrate:
        result = hydrate_deployment()
    elif args.validate:
        validate_evaluation_deployment()
        result = {"status": "valid"}
    else:
        result = prepare_deployment()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
