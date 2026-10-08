"""Strict, serializable result contract for generated Harbor replay."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from agentops_demo.contracts._base import ContractModel
from agentops_demo.harbor.provenance import PackageProvenance

ReproductionStatus = Literal[
    "CANDIDATE", "VALIDATED", "NOT_REPRODUCED", "NEEDS_REVIEW", "UNSUPPORTED"
]
ReplayMode = Literal["exact_source", "calibration"]


class SourceProvenance(ContractModel):
    source_revision: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    agent_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(default="scripted/bad", min_length=1)


class CalibrationIdentity(ContractModel):
    """The isolated behavior used to calibrate a production-derived Scenario."""

    kind: Literal["source_replay", "trajectory_calibration"]
    model: str = Field(min_length=1)
    agent_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_tools: list[str] = Field(min_length=1)
    derived_from_observed_failure: bool


class TrialResult(ContractModel):
    name: Literal["oracle", "known-good", "originating"]
    attempted: bool
    reward: float | None = None
    tool_calls: list[str] = Field(default_factory=list)
    job_dir: str | None = None
    expected_reward: float | None = None
    expected_tool_calls: list[str] | None = None
    reward_matches: bool | None = None
    tools_match: bool | None = None
    config_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    package_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model: str | None = None

    @model_validator(mode="after")
    def observations_match_flags(self) -> TrialResult:
        if self.reward_matches is not None and self.expected_reward is None:
            raise ValueError("reward match requires an expected reward")
        if self.tools_match is not None and self.expected_tool_calls is None:
            raise ValueError("tool match requires expected tool calls")
        return self


class HarborTaskIdentity(ContractModel):
    """Digest of the exact rendered no-network Harbor task validated in Phase 5."""

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReproductionReport(ContractModel):
    schema_version: Literal["1"] = "1"
    status: ReproductionStatus
    replay_mode: ReplayMode
    source_kind: str = Field(min_length=1)
    source: SourceProvenance
    calibration: CalibrationIdentity | None = None
    execution_package: PackageProvenance | None = None
    harbor_task: HarborTaskIdentity | None = None
    attempts: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    trials: list[TrialResult] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_semantics(self) -> ReproductionReport:
        if self.status == "UNSUPPORTED":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("UNSUPPORTED requires a reason")
            if self.calibration is not None:
                raise ValueError("UNSUPPORTED must not claim a calibration")
            if self.trials or self.attempts != 0 or self.failure_count != 0:
                raise ValueError("UNSUPPORTED must not claim replay execution")
            return self
        if self.calibration is None:
            raise ValueError("supported reproduction evidence requires a calibration identity")
        if self.replay_mode == "exact_source":
            if self.execution_package is None:
                raise ValueError("exact-source replay requires an execution package")
            if self.execution_package.source_revision != self.source.source_revision:
                raise ValueError("exact-source package revision must match source revision")
        if self.source.model.startswith("bedrock/"):
            if self.replay_mode != "calibration":
                raise ValueError("Bedrock source evidence may only use calibration replay")
            if self.calibration.kind != "trajectory_calibration":
                raise ValueError("Bedrock source requires trajectory calibration")
            if self.calibration.model != "scripted/bad":
                raise ValueError("Bedrock trajectory calibration must use scripted/bad")
            if not self.calibration.derived_from_observed_failure:
                raise ValueError(
                    "Bedrock calibration must be derived from observed failure evidence"
                )
            if self.calibration.agent_config_fingerprint == self.source.agent_config_fingerprint:
                raise ValueError("Bedrock calibration must have a distinct AgentConfig fingerprint")
        elif self.calibration.kind == "trajectory_calibration":
            raise ValueError("trajectory calibration is reserved for supported Bedrock evidence")
        elif self.calibration.agent_config_fingerprint != self.source.agent_config_fingerprint:
            raise ValueError(
                "source replay calibration must retain the source AgentConfig fingerprint"
            )
        trials = {trial.name: trial for trial in self.trials}
        if len(trials) != len(self.trials):
            raise ValueError("reproduction report has duplicate trial names")
        originating = trials.get("originating")
        expected_attempts = int(originating is not None and originating.attempted)
        expected_failures = int(
            originating is not None and originating.attempted and originating.reward == 0.0
        )
        if (self.attempts, self.failure_count) != (expected_attempts, expected_failures):
            raise ValueError("source replay counters do not match originating trial")
        if self.status == "VALIDATED":
            if self.harbor_task is None:
                raise ValueError("VALIDATED requires rendered Harbor task provenance")
            if not _is_reward(trials.get("oracle"), 1.0):
                raise ValueError("VALIDATED requires Oracle reward 1")
            if not _is_reward(trials.get("known-good"), 1.0):
                raise ValueError("VALIDATED requires known-good reward 1")
            if not _is_reward(originating, 0.0) or self.failure_count < 1:
                raise ValueError("VALIDATED requires a reproduced originating failure")
        if self.status == "NOT_REPRODUCED":
            if originating is None or not originating.attempted or originating.reward is None:
                raise ValueError("NOT_REPRODUCED requires a numeric originating result")
            if originating.reward == 0.0:
                raise ValueError("NOT_REPRODUCED cannot claim a reproduced source failure")
        if self.status == "NEEDS_REVIEW" and not (isinstance(self.reason, str) and self.reason):
            raise ValueError("NEEDS_REVIEW requires a diagnostic reason")
        return self


def _is_reward(trial: TrialResult | None, expected: float) -> bool:
    return trial is not None and trial.attempted and trial.reward == expected
