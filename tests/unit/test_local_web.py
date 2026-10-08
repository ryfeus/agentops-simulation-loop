from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agentops_demo.agent.graph import build_agent_with_tools
from agentops_demo.agent.local import (
    DEFAULT_BEDROCK_MODEL_ID,
    local_agent_config_from_environment,
    local_candidate_from_environment,
)
from agentops_demo.agent.runtime import AgentSessionRuntime
from agentops_demo.agent.scripted import create_scripted_model
from agentops_demo.agentcore.candidates import resolve_candidate
from agentops_demo.web.server import WebServerSettings, create_web_app


def test_scripted_local_candidate_needs_no_bedrock_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)

    config = local_agent_config_from_environment("scripted-correct")

    assert local_candidate_from_environment("scripted-correct") == "scripted-correct"
    assert config.model.provider == "scripted"
    assert config.model.model_id == "correct"
    assert config.fingerprint() != local_agent_config_from_environment("scripted-bad").fingerprint()


def test_bedrock_local_candidate_requires_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    with pytest.raises(RuntimeError, match="BEDROCK_MODEL_ID"):
        local_agent_config_from_environment("bedrock")

    monkeypatch.setenv("BEDROCK_MODEL_ID", "configured-model")
    assert local_agent_config_from_environment("bedrock").model.model_id == "configured-model"
    assert DEFAULT_BEDROCK_MODEL_ID


def test_web_health_and_cors_expose_only_safe_effective_metadata(agent_config) -> None:
    config = resolve_candidate(agent_config, "scripted-correct")
    settings = WebServerSettings(candidate="scripted-correct", config=config, mcp_url="unused")

    async def create_runtime(**_kwargs: object) -> AgentSessionRuntime:
        return AgentSessionRuntime(config=config, graph=SimpleNamespace(nodes={}))  # type: ignore[arg-type]

    with TestClient(create_web_app(settings, runtime_factory=create_runtime)) as client:
        response = client.get("/health")
        cors = client.options(
            "/agent",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        rejected = client.options(
            "/agent",
            headers={
                "Origin": "https://example.invalid",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.json() == {
        "status": "ok",
        "candidate": "scripted-correct",
        "model": "scripted/correct",
        "agent_config_fingerprint": config.fingerprint(),
        "source_revision": "abc123",
    }
    assert cors.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in rejected.headers


def test_web_runtime_bootstrap_failure_is_concise(agent_config) -> None:
    settings = WebServerSettings(candidate="scripted-bad", config=agent_config, mcp_url="unused")

    async def fail_runtime(**_kwargs: object) -> AgentSessionRuntime:
        raise ValueError("secret diagnostic")

    with (
        pytest.raises(RuntimeError, match="Unable to start the local agent runtime"),
        TestClient(create_web_app(settings, runtime_factory=fail_runtime)),
    ):
        pass


def test_web_adapter_streams_scripted_tool_events(agent_config) -> None:
    config = resolve_candidate(agent_config, "scripted-correct")
    settings = WebServerSettings(candidate="scripted-correct", config=config, mcp_url="unused")

    @tool
    async def get_invoice(invoice_id: str) -> dict[str, str]:
        """Get an invoice."""

        return {"invoice_id": invoice_id, "status": "disputed"}

    @tool
    async def refund_invoice(invoice_id: str, reason: str) -> dict[str, str]:
        """Refund an invoice."""

        return {"invoice_id": invoice_id, "reason": reason}

    @tool
    async def escalate_dispute(invoice_id: str, reason: str) -> dict[str, str]:
        """Escalate an invoice dispute."""

        return {"invoice_id": invoice_id, "reason": reason}

    async def create_runtime(**_kwargs: object) -> AgentSessionRuntime:
        graph = build_agent_with_tools(
            config=config,
            model=create_scripted_model("correct"),
            tools=[get_invoice, refund_invoice, escalate_dispute],
            checkpointer=InMemorySaver(),
        )
        return AgentSessionRuntime(config=config, graph=graph)

    with TestClient(create_web_app(settings, runtime_factory=create_runtime)) as client:
        response = client.post(
            "/agent",
            headers={"Accept": "text/event-stream"},
            json={
                "threadId": "browser-thread",
                "runId": "browser-run",
                "tools": [],
                "context": [],
                "forwardedProps": {},
                "messages": [{"id": "user-1", "role": "user", "content": "refund inv-123"}],
            },
        )

    assert response.status_code == 200, response.text
    assert "TOOL_CALL_START" in response.text
    assert "get_invoice" in response.text
    assert "RUN_FINISHED" in response.text


def test_local_default_is_scripted_without_credentials(monkeypatch):
    monkeypatch.delenv("AGENTCORE_CANDIDATE", raising=False)
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    assert local_candidate_from_environment() == "scripted-correct"
    assert local_agent_config_from_environment().model.provider == "scripted"
