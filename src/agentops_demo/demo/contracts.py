"""Fail-closed, secret-free contracts for a Phase 7 demo run."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from agentops_demo.contracts._base import ContractModel
from agentops_demo.scale.contracts import CandidateGate, CleanupSummary
from agentops_demo.taskify.reproduction import CalibrationIdentity, ReproductionReport

DemoMode = Literal["live", "replay"]
DemoStage = Literal[
    "preflight",
    "production_failure",
    "taskify",
    "local_regression",
    "ec2_parity",
    "ec2_scale",
    "finalize",
]


class StageResult(ContractModel):
    stage: DemoStage
    status: Literal["PASS", "FAIL"]
    started_at: str
    finished_at: str
    artifacts: dict[str, str] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_artifacts(self) -> StageResult:
        if set(self.artifacts) != set(self.artifact_paths):
            raise ValueError("stage artifact paths and hashes must have identical keys")
        return self


class ProductionFailureIdentity(ContractModel):
    source: Literal["LIVE", "REPLAY"]
    model: str
    session_id: str
    trace_id: str
    source_revision: str
    agent_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LineageSummary(ContractModel):
    trace_id: str
    session_id: str
    scenario_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    harbor_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool


class DemoReport(ContractModel):
    schema_version: Literal["1"] = "1"
    run_id: str = Field(min_length=1)
    mode: DemoMode
    started_at: str
    finished_at: str
    git_revision: str
    source: ProductionFailureIdentity
    calibration: CalibrationIdentity
    local_validation: ReproductionReport
    ec2_parity: CandidateGate
    ec2_scale: CandidateGate
    lineage: LineageSummary
    cleanup: CleanupSummary
    accepted: bool
    failure_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_acceptance(self) -> DemoReport:
        accepted = (
            self.local_validation.status == "VALIDATED"
            and self.ec2_parity.accepted
            and self.ec2_parity.local_ec2_parity is True
            and self.ec2_scale.accepted
            and self.ec2_scale.eligible is True
            and self.lineage.passed
            and self.cleanup.passed
        )
        if self.accepted != accepted:
            raise ValueError("demo accepted claim does not match linked evidence")
        if accepted and self.failure_reasons:
            raise ValueError("accepted demo cannot contain failure reasons")
        if not accepted and not self.failure_reasons:
            raise ValueError("rejected demo requires failure reasons")
        if self.mode == "live" and self.source.source != "LIVE":
            raise ValueError("live demo must record live production evidence")
        if self.mode == "replay" and self.source.source != "REPLAY":
            raise ValueError("replay demo must be explicitly labeled as replay")
        if self.calibration != self.local_validation.calibration:
            raise ValueError("demo calibration identity must match local reproduction evidence")
        if self.accepted and self.mode == "live":
            if self.local_validation.source.model != self.source.model:
                raise ValueError("accepted live demo source model must match local evidence")
            calibration = self.local_validation.calibration
            if (
                calibration is None
                or calibration.kind != "trajectory_calibration"
                or calibration.model != "scripted/bad"
            ):
                raise ValueError("accepted live demo requires scripted/bad trajectory calibration")
        return self
