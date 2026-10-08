"""Deterministic, purpose-specific aggregation for retained Harbor EC2 outcomes."""

from __future__ import annotations

from statistics import fmean

from agentops_demo.harbor.provenance import PackageProvenance
from agentops_demo.scale.contracts import (
    BenchmarkIdentity,
    CandidateGate,
    CandidateIdentity,
    CleanupSummary,
    Ec2ExecutionConfig,
    Ec2Trial,
    RunPurpose,
    TaskSource,
    TrialAggregate,
    expected_trial_counts,
)


def aggregate_trials(
    *, purpose: RunPurpose, execution: Ec2ExecutionConfig, trials: list[Ec2Trial]
) -> list[TrialAggregate]:
    """Summarize raw evidence without treating infrastructure failures as rewards."""

    aggregates: list[TrialAggregate] = []
    for name, requested in expected_trial_counts(purpose, execution).items():
        observed = [trial for trial in trials if trial.name == name]
        retained_trials = [trial for trial in observed if trial.retained]
        completed_trials = [trial for trial in retained_trials if trial.completed]
        rewards = [trial.reward for trial in completed_trials if trial.reward is not None]
        passes = [
            trial
            for trial in completed_trials
            if trial.infrastructure_failure is None and trial.reward == trial.requested_reward
        ]
        failures = [
            trial
            for trial in completed_trials
            if trial.infrastructure_failure is None and trial.reward != trial.requested_reward
        ]
        aggregates.append(
            TrialAggregate(
                name=name,
                requested=requested,
                retained=len(retained_trials),
                completed=len(completed_trials),
                benchmark_passes=len(passes),
                benchmark_failures=len(failures),
                infrastructure_failures=sum(
                    trial.infrastructure_failure is not None for trial in observed
                ),
                rewards=rewards,
                mean_reward=fmean(rewards) if rewards else None,
                provenance_matches=bool(retained_trials)
                and all(trial.provenance_matches for trial in retained_trials),
            )
        )
    return aggregates


def candidate_gate(
    *,
    run_id: str,
    purpose: RunPurpose,
    task_source: TaskSource,
    execution: Ec2ExecutionConfig,
    benchmark: BenchmarkIdentity,
    execution_package: PackageProvenance,
    candidate: CandidateIdentity | None,
    trials: list[Ec2Trial],
    cleanup: CleanupSummary,
    local_ec2_parity: bool | None,
) -> CandidateGate:
    """Build an honest report: every semantic failure stays observable and rejects it."""

    aggregates = aggregate_trials(purpose=purpose, execution=execution, trials=trials)
    reasons: list[str] = []
    for aggregate in aggregates:
        if aggregate.retained != aggregate.requested:
            reasons.append(
                f"{aggregate.name} retained {aggregate.retained} trials; "
                f"expected {aggregate.requested}"
            )
        if aggregate.completed != aggregate.requested:
            reasons.append(
                f"{aggregate.name} completed {aggregate.completed} trials; "
                f"expected {aggregate.requested}"
            )
        if aggregate.benchmark_failures:
            reasons.append(
                f"{aggregate.name} has {aggregate.benchmark_failures} benchmark reward failures"
            )
        if aggregate.infrastructure_failures:
            reasons.append(
                f"{aggregate.name} has {aggregate.infrastructure_failures} infrastructure failures"
            )
        if not aggregate.provenance_matches:
            reasons.append(f"{aggregate.name} provenance did not match")
    if not cleanup.passed:
        reasons.append("worker cleanup did not pass")
    if purpose == "parity" and local_ec2_parity is not True:
        reasons.append("local EC2 parity did not match")
    accepted = not reasons
    eligible = accepted if purpose == "candidate_gate" else None
    return CandidateGate(
        run_id=run_id,
        purpose=purpose,
        task_source=task_source,
        execution=execution,
        benchmark=benchmark,
        execution_package=execution_package,
        candidate=candidate,
        trials=trials,
        aggregates=aggregates,
        cleanup=cleanup,
        local_ec2_parity=local_ec2_parity,
        accepted=accepted,
        eligible=eligible,
        failure_reasons=reasons,
    )
