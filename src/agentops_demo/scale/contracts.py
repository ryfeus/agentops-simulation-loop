"""Strict, credential-free contracts for the Phase 6 execution plane."""

from __future__ import annotations

from statistics import fmean
from typing import Literal

from pydantic import Field, model_validator

from agentops_demo.contracts._base import ContractModel
from agentops_demo.harbor.provenance import PackageProvenance

AllowedInstanceType = Literal["m7i-flex.large", "m7i.large", "t3.large"]
TaskSource = Literal["taskify", "static"]
RunPurpose = Literal["oracle_smoke", "calibration_smoke", "parity", "candidate_gate"]


class Ec2ExecutionConfig(ContractModel):
    """Non-secret EC2 inputs materialized from Terraform outputs."""

    schema_version: Literal["1"] = "1"
    region: Literal["us-west-2"] = "us-west-2"
    launch_mode: Literal["ephemeral"] = "ephemeral"
    use_public_ip: Literal[True] = True
    ssh_user: Literal["ubuntu"] = "ubuntu"
    root_volume_type: Literal["gp3"] = "gp3"
    root_volume_size_gb: Literal[30] = 30
    bootstrap_docker: Literal[True] = True
    vpc_id: str = Field(min_length=1)
    subnet_id: str = Field(min_length=1)
    security_group_id: str = Field(min_length=1)
    key_name: str = Field(min_length=1)
    ami_id: str = Field(min_length=1)
    instance_type: AllowedInstanceType = "m7i-flex.large"
    attempts: int = Field(default=1, ge=1, le=4)
    concurrency: int = Field(default=1, ge=1, le=4)
    run_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    task_source: TaskSource
    tags: dict[str, str]

    @model_validator(mode="after")
    def require_phase_tags(self) -> Ec2ExecutionConfig:
        required = {"Project": "agentops-demo", "Phase": "6", "RunId": self.run_id}
        if {key: self.tags.get(key) for key in required} != required:
            raise ValueError("EC2 execution tags must identify the Phase 6 project and run")
        return self


class Ec2Trial(ContractModel):
    """One observed EC2 trial; controller failures may be retained=False."""

    name: str = Field(min_length=1)
    requested_reward: float
    retained: bool = True
    reward: float | None = None
    completed: bool
    provenance_matches: bool = False
    infrastructure_failure: str | None = None
    tool_calls: list[str] = Field(default_factory=list)
    config_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    package_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model: str | None = None
    worker_instance_ids: list[str] = Field(default_factory=list)
    elapsed_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_observation(self) -> Ec2Trial:
        if self.completed and (not self.retained or self.reward is None):
            raise ValueError("completed trial requires retained numeric Harbor evidence")
        if self.reward is not None and not self.completed:
            raise ValueError("numeric reward requires a completed trial")
        return self


class TrialAggregate(ContractModel):
    """Purpose-aware summary that never hides incomplete execution."""

    name: str = Field(min_length=1)
    requested: int = Field(ge=0)
    retained: int = Field(ge=0)
    completed: int = Field(ge=0)
    benchmark_passes: int = Field(ge=0)
    benchmark_failures: int = Field(ge=0)
    infrastructure_failures: int = Field(ge=0)
    rewards: list[float] = Field(default_factory=list)
    mean_reward: float | None = None
    provenance_matches: bool


class BenchmarkIdentity(ContractModel):
    """Immutable task identity linking EC2 execution to its source evidence."""

    task_source: TaskSource
    harbor_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_id: str | None = None
    scenario_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    phase5_status: str | None = None
    phase5_replay_mode: str | None = None
    trace_id: str | None = None
    session_id: str | None = None

    @model_validator(mode="after")
    def validate_source_identity(self) -> BenchmarkIdentity:
        values = (
            self.scenario_id,
            self.scenario_sha256,
            self.phase5_status,
            self.phase5_replay_mode,
            self.trace_id,
            self.session_id,
        )
        if self.task_source == "taskify":
            if not all(isinstance(value, str) and value for value in values):
                raise ValueError("taskify benchmark identity requires Phase 5 trace provenance")
        elif any(value is not None for value in values):
            raise ValueError("static benchmark identity must not claim Phase 5 provenance")
        return self


class CleanupSummary(ContractModel):
    workers_observed: list[str] = Field(default_factory=list)
    workers_remaining: list[str] = Field(default_factory=list)
    passed: bool
    error: str | None = None

    @model_validator(mode="after")
    def validate_cleanup(self) -> CleanupSummary:
        if self.passed and (self.workers_remaining or self.error is not None):
            raise ValueError("cleanup cannot pass with workers remaining or an observation error")
        return self


class CandidateIdentity(ContractModel):
    name: Literal["known-good"]
    model: Literal["scripted/correct"]
    agent_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def expected_trial_counts(purpose: RunPurpose, execution: Ec2ExecutionConfig) -> dict[str, int]:
    """Return exact required Harbor cardinality for a run purpose."""

    if purpose == "oracle_smoke":
        return {"oracle": execution.attempts}
    if purpose == "calibration_smoke":
        return {"scripted-bad": execution.attempts}
    if purpose == "candidate_gate":
        return {"known-good": execution.attempts}
    return {
        "oracle": execution.attempts,
        "known-good": execution.attempts,
        "scripted-bad": execution.attempts,
    }


class CandidateGate(ContractModel):
    """A fail-closed EC2 acceptance report; it never promotes a candidate."""

    schema_version: Literal["1"] = "1"
    run_id: str = Field(min_length=1)
    purpose: RunPurpose
    task_source: TaskSource
    execution: Ec2ExecutionConfig
    benchmark: BenchmarkIdentity
    execution_package: PackageProvenance
    candidate: CandidateIdentity | None = None
    trials: list[Ec2Trial]
    aggregates: list[TrialAggregate]
    cleanup: CleanupSummary
    local_ec2_parity: bool | None = None
    accepted: bool
    eligible: bool | None
    failure_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_acceptance_claim(self) -> CandidateGate:
        if (
            self.task_source != self.execution.task_source
            or self.task_source != self.benchmark.task_source
        ):
            raise ValueError("report task source must match execution and benchmark identity")
        expected = expected_trial_counts(self.purpose, self.execution)
        if any(trial.name not in expected for trial in self.trials):
            raise ValueError("report contains a trial outside the requested purpose")
        aggregates = {aggregate.name: aggregate for aggregate in self.aggregates}
        if set(aggregates) != set(expected):
            raise ValueError("report aggregates must match purpose-required trial names")
        if any(aggregates[name].requested != count for name, count in expected.items()):
            raise ValueError("aggregate requested cardinality does not match purpose")
        for name, aggregate in aggregates.items():
            observed = [trial for trial in self.trials if trial.name == name]
            retained = [trial for trial in observed if trial.retained]
            completed = [trial for trial in retained if trial.completed]
            rewards = [trial.reward for trial in completed if trial.reward is not None]
            passes = [
                trial
                for trial in completed
                if trial.infrastructure_failure is None and trial.reward == trial.requested_reward
            ]
            failures = [
                trial
                for trial in completed
                if trial.infrastructure_failure is None and trial.reward != trial.requested_reward
            ]
            expected_values = (
                len(retained),
                len(completed),
                len(passes),
                len(failures),
                sum(trial.infrastructure_failure is not None for trial in observed),
                rewards,
                bool(retained) and all(trial.provenance_matches for trial in retained),
            )
            actual_values = (
                aggregate.retained,
                aggregate.completed,
                aggregate.benchmark_passes,
                aggregate.benchmark_failures,
                aggregate.infrastructure_failures,
                aggregate.rewards,
                aggregate.provenance_matches,
            )
            if actual_values != expected_values:
                raise ValueError(f"aggregate {name} does not match raw trial evidence")
            expected_mean = fmean(rewards) if rewards else None
            if aggregate.mean_reward != expected_mean:
                raise ValueError(f"aggregate {name} mean reward does not match raw evidence")
        if self.purpose == "candidate_gate":
            if (self.accepted, self.eligible) not in {(True, True), (False, False)}:
                raise ValueError("candidate gate requires accepted and eligible to match")
            if self.candidate is None:
                raise ValueError("candidate gate requires candidate identity")
            candidate_trials = [
                trial
                for trial in self.trials
                if trial.name == self.candidate.name and trial.retained
            ]
            if any(
                trial.provenance_matches
                and (
                    trial.package_sha256 != self.candidate.package_sha256
                    or trial.config_fingerprint != self.candidate.agent_config_fingerprint
                    or trial.model != self.candidate.model
                )
                for trial in candidate_trials
            ):
                raise ValueError("candidate provenance match claim is not homogeneous")
        elif self.eligible is not None:
            raise ValueError("only a candidate gate may claim eligibility")
        if self.purpose == "parity" and self.local_ec2_parity is None:
            raise ValueError("parity report requires local EC2 parity evidence")
        if self.purpose != "parity" and self.local_ec2_parity is not None:
            raise ValueError("only parity may include local EC2 parity evidence")
        if self.accepted:
            if self.failure_reasons:
                raise ValueError("accepted report cannot contain failure reasons")
            if not self.cleanup.passed:
                raise ValueError("accepted report requires worker cleanup")
            if self.purpose == "parity" and self.local_ec2_parity is not True:
                raise ValueError("accepted parity report requires matching local evidence")
            for aggregate in self.aggregates:
                if (
                    aggregate.requested != aggregate.retained
                    or aggregate.retained != aggregate.completed
                    or aggregate.benchmark_passes != aggregate.requested
                    or aggregate.benchmark_failures != 0
                    or aggregate.infrastructure_failures != 0
                    or not aggregate.provenance_matches
                ):
                    raise ValueError("accepted report contains incomplete or invalid aggregate")
        elif not self.failure_reasons:
            raise ValueError("rejected report requires structured failure reasons")
        return self
