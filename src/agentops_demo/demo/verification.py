"""Independent verification for a complete, secret-free Phase 7 evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agentops_demo.demo.contracts import DemoReport, StageResult
from agentops_demo.demo.lineage import validate_lineage
from agentops_demo.scale.contracts import CandidateGate
from agentops_demo.taskify.reproduction import ReproductionReport
from agentops_demo.taskify.trace import trace_sha256

_STAGES = (
    (2, "production_failure"),
    (3, "taskify"),
    (4, "local_regression"),
    (5, "ec2_parity"),
    (6, "ec2_scale"),
    (7, "finalize"),
)
_SECRET_MARKERS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "PRIVATE KEY-----",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(left: object, right: object) -> bool:
    return left.model_dump(mode="json") == right.model_dump(mode="json")  # type: ignore[attr-defined]


def _verify_stage(root: Path, number: int, name: str) -> None:
    path = root / "stages" / f"{number:02d}-{name.replace('_', '-')}.json"
    result = StageResult.model_validate_json(path.read_text())
    if result.status != "PASS":
        raise AssertionError(f"stage {name} did not pass")
    for artifact_name, digest in result.artifacts.items():
        artifact = root / result.artifact_paths[artifact_name]
        if not artifact.is_file() or _sha(artifact) != digest:
            raise AssertionError(f"stage {name} artifact {artifact_name} is missing or altered")


def verify_demo_bundle(root: Path) -> DemoReport:
    """Reload all evidence; the final report is a summary, never root of trust."""

    report = DemoReport.model_validate_json((root / "demo-report.json").read_text())
    for number, name in _STAGES:
        _verify_stage(root, number, name)
    failure = root / "production/failure.json"
    trace = root / "production/trace.json"
    if _sha(failure) != report.source.failure_sha256:
        raise AssertionError("frozen production failure digest does not match report")
    if trace_sha256(json.loads(trace.read_text())) != report.source.trace_sha256:
        raise AssertionError("frozen production trace digest does not match report")
    scenarios = list((root / "taskify").glob("*/scenario.yaml"))
    if len(scenarios) != 1:
        raise AssertionError("demo bundle must contain exactly one dynamic Scenario")
    scenario = scenarios[0]
    scenario_dir = scenario.parent
    local = ReproductionReport.model_validate_json((scenario_dir / "reproduction.json").read_text())
    parity = CandidateGate.model_validate_json(
        (root / "ec2/parity/candidate-gate.json").read_text()
    )
    scale = CandidateGate.model_validate_json((root / "ec2/scale/candidate-gate.json").read_text())
    if not _same(local, report.local_validation):
        raise AssertionError("reproduction evidence differs from final report")
    if not _same(parity, report.ec2_parity) or not _same(scale, report.ec2_scale):
        raise AssertionError("EC2 evidence differs from final report")
    lineage = validate_lineage(report.source, str(scenario), local, parity, scale)
    if not lineage.passed or not _same(lineage, report.lineage):
        raise AssertionError("demo lineage does not match underlying evidence")
    if not report.accepted or not report.cleanup.passed:
        raise AssertionError("demo evidence is not accepted")
    for path in root.rglob("*"):
        if path.is_file():
            text = path.read_text(errors="ignore")
            if any(marker in text for marker in _SECRET_MARKERS):
                raise AssertionError("demo evidence contains a credential marker")
    return report
