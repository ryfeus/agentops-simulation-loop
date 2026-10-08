"""Strict one-turn AgentCore request and response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from agentops_demo.contracts._base import ContractModel
from agentops_demo.contracts.billing import BillingToolName
from agentops_demo.contracts.scenario import BenchmarkMetadata

CandidateSelector = Literal["bedrock", "scripted-bad", "scripted-correct"]


class AgentCoreRequest(ContractModel):
    instruction: str = Field(min_length=1)
    evaluation_context: BenchmarkMetadata | None = None
    candidate: CandidateSelector | None = None

    @field_validator("instruction")
    @classmethod
    def instruction_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("instruction must not be blank")
        return value


class AgentToolCall(ContractModel):
    name: BillingToolName
    arguments: dict[str, Any]


class AgentCoreResponse(ContractModel):
    final_response: str
    tool_calls: list[AgentToolCall]
    completed_tools: list[BillingToolName]
    agent_config_fingerprint: str
    source_revision: str
