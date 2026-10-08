"""Strict evidence contracts consumed by taskification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from agentops_demo.contracts._base import ContractModel
from agentops_demo.contracts.scenario import BenchmarkMetadata


class FailureEvidenceError(ValueError):
    pass


class FailureEvaluator(ContractModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    label: Literal["FAIL"]
    value: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    explanation: str | None = None


class FailureSource(ContractModel):
    runtime_arn: str | None = None
    service_name: str = Field(min_length=1)
    trace_log_group: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    result_log_group: str = Field(min_length=1)
    result_timestamp: str | int | None = None
    instruction: str = Field(min_length=1)


class FailureCandidate(ContractModel):
    source_revision: str = Field(min_length=1)
    agent_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    tools_version: str = Field(min_length=1)
    harness_framework: str = Field(min_length=1)
    harness_version: str = Field(min_length=1)


class FailureEvidence(ContractModel):
    schema_version: Literal["1"]
    evaluator: FailureEvaluator
    source: FailureSource
    candidate: FailureCandidate
    trajectory: list[str] = Field(default_factory=list)
    trace_span_count: int | None = Field(default=None, ge=1)
    benchmark_context: BenchmarkMetadata | None = None

    @model_validator(mode="after")
    def fail_is_zero(self) -> FailureEvidence:
        if self.evaluator.value != 0.0:
            raise ValueError("FAIL evidence must have a 0.0 evaluator value")
        return self
