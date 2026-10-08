"""Local-only configuration for terminal and browser billing-agent clients."""

from __future__ import annotations

import os

from agentops_demo.agentcore.candidates import SUPPORTED_CANDIDATES, resolve_candidate
from agentops_demo.agentcore.request import CandidateSelector
from agentops_demo.contracts.agent_config import AgentConfig

DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-6"


def local_candidate_from_environment(candidate: str | None = None) -> CandidateSelector:
    """Return the supported launch-time local candidate selector."""

    selected = candidate or os.getenv("AGENTCORE_CANDIDATE", "scripted-correct")
    if selected not in SUPPORTED_CANDIDATES:
        raise ValueError(f"unsupported local candidate {selected!r}")
    return selected  # type: ignore[return-value]


def local_agent_config_from_environment(candidate: str | None = None) -> AgentConfig:
    """Build one effective local AgentConfig without exposing credentials to a UI."""

    selected = local_candidate_from_environment(candidate)
    configured_model_id = os.getenv("BEDROCK_MODEL_ID")
    if selected == "bedrock" and not configured_model_id:
        raise RuntimeError("BEDROCK_MODEL_ID is required for the Bedrock local chat candidate")
    base = AgentConfig.model_validate(
        {
            "agent": {"source_revision": os.getenv("AGENT_SOURCE_REVISION", "local")},
            "model": {
                "provider": "bedrock",
                "model_id": configured_model_id or DEFAULT_BEDROCK_MODEL_ID,
            },
            "prompt": {"version": os.getenv("AGENT_PROMPT_VERSION", "billing-v1")},
            "tools": {"version": "billing-mcp-v1"},
            "harness": {"framework": "langgraph", "version": "v1"},
        }
    )
    return resolve_candidate(base, selected)
