"""Explicit execution-profile identity for isolated and real-model Harbor tasks."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from agentops_demo.contracts._base import ContractModel


class HarborExecutionProfile(ContractModel):
    name: Literal["isolated", "bedrock"]
    base_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    agent_network_mode: Literal["no-network", "public"]
    verifier_network_mode: Literal["no-network"]
    credential_mode: Literal["sts-assume-role"] | None = None

    @model_validator(mode="after")
    def validate_security_boundary(self) -> HarborExecutionProfile:
        if self.name == "isolated":
            if self.agent_network_mode != "no-network" or self.credential_mode is not None:
                raise ValueError("isolated profile must remain no-network and credential-free")
        elif self.agent_network_mode != "public" or self.credential_mode != "sts-assume-role":
            raise ValueError("Bedrock profile requires public agent egress and STS credentials")
        return self
