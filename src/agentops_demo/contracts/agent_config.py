"""Version identity for an evaluated agent candidate."""

from __future__ import annotations

import hashlib
import json

from pydantic import Field

from agentops_demo.contracts._base import ContractModel


class AgentIdentity(ContractModel):
    """Source code identity for the candidate."""

    source_revision: str = Field(min_length=1)


class ModelConfig(ContractModel):
    """Model provider and provider-specific model identifier."""

    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)


class PromptConfig(ContractModel):
    """Prompt contract version."""

    version: str = Field(min_length=1)


class ToolsConfig(ContractModel):
    """MCP tool contract version."""

    version: str = Field(min_length=1)


class HarnessConfig(ContractModel):
    """Agent harness identity."""

    framework: str = Field(min_length=1)
    version: str = Field(min_length=1)


class AgentConfig(ContractModel):
    """Complete, secret-free identity of an evaluated agent candidate."""

    agent: AgentIdentity
    model: ModelConfig
    prompt: PromptConfig
    tools: ToolsConfig
    harness: HarnessConfig

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 identity for the logical configuration."""

        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
