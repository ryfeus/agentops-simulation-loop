"""Trusted invocation-time AgentCore candidate selection and runtime ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from agentops_demo.agent.runtime import AgentSessionRuntime
from agentops_demo.agentcore.request import CandidateSelector
from agentops_demo.contracts.agent_config import AgentConfig

SUPPORTED_CANDIDATES: tuple[CandidateSelector, ...] = (
    "bedrock",
    "scripted-bad",
    "scripted-correct",
)

RuntimeFactory = Callable[..., Awaitable[AgentSessionRuntime]]


class CandidateResolutionError(ValueError):
    """Raised when a trusted selector is unavailable for the deployment."""


class SessionCandidateConflictError(ValueError):
    """Raised when a runtime session attempts to change its candidate."""


def resolve_candidate(
    deployed_config: AgentConfig, selector: CandidateSelector | None
) -> AgentConfig:
    """Return an immutable effective AgentConfig for one allowed selector."""

    if selector is None:
        return deployed_config
    if selector == "bedrock":
        if deployed_config.model.provider != "bedrock":
            raise CandidateResolutionError("the deployed default candidate is not Bedrock")
        return deployed_config
    model_id = {"scripted-bad": "bad", "scripted-correct": "correct"}[selector]
    value = deployed_config.model_dump(mode="json")
    value["model"] = {"provider": "scripted", "model_id": model_id}
    return AgentConfig.model_validate(value)


class CandidateRuntimePool:
    """Lazily create and retain one session runtime per candidate fingerprint."""

    def __init__(self, *, mcp_url: str, runtime_factory: RuntimeFactory) -> None:
        self._mcp_url = mcp_url
        self._runtime_factory = runtime_factory
        self._runtimes: dict[str, AgentSessionRuntime] = {}
        self._lock = asyncio.Lock()

    async def runtime_for(self, config: AgentConfig) -> AgentSessionRuntime:
        fingerprint = config.fingerprint()
        async with self._lock:
            runtime = self._runtimes.get(fingerprint)
            if runtime is None:
                runtime = await self._runtime_factory(config=config, mcp_url=self._mcp_url)
                self._runtimes[fingerprint] = runtime
            return runtime


class SessionCandidateBindings:
    """Bind each process-local AgentCore session to one effective candidate."""

    def __init__(self) -> None:
        self._fingerprints: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def bind_or_validate(self, *, session_id: str, fingerprint: str) -> None:
        async with self._lock:
            bound = self._fingerprints.get(session_id)
            if bound is None:
                self._fingerprints[session_id] = fingerprint
                return
            if bound != fingerprint:
                raise SessionCandidateConflictError(
                    "AgentCore runtime session is already bound to a different candidate"
                )
