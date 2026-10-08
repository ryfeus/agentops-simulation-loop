from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentops_demo.agentcore.server import create_agentcore_app
from agentops_demo.agentcore.settings import AgentCoreSettings
from agentops_demo.billing.sqlite_repository import (
    SQLiteBillingRepository,
    initialize_database,
)
from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.validation.scenario import load_scenario
from tests.conftest import GOLDEN_SCENARIO


def available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def scripted_config() -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "agent": {"source_revision": "integration-test"},
            "model": {"provider": "scripted", "model_id": "correct"},
            "prompt": {"version": "billing-v2"},
            "tools": {"version": "billing-mcp-v1"},
            "harness": {"framework": "langgraph", "version": "v1"},
        }
    )


def test_agentcore_http_to_langgraph_mcp_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "billing.db"
    scenario = load_scenario(GOLDEN_SCENARIO)
    asyncio.run(initialize_database(database_path, scenario.initial_state))
    port = available_port()
    monkeypatch.setenv("BILLING_REPOSITORY_BACKEND", "sqlite")
    monkeypatch.setenv("BILLING_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("BILLING_POLICY_MODE", "permissive")
    settings = AgentCoreSettings(config=scripted_config(), mcp_port=port)

    with TestClient(create_agentcore_app(settings)) as client:
        response = client.post(
            "/invocations",
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "integration-session"},
            json={"instruction": scenario.instruction},
        )

    assert response.status_code == 200
    assert response.json()["completed_tools"] == ["get_invoice", "escalate_dispute"]
    repository = SQLiteBillingRepository(database_path)
    assert asyncio.run(repository.mutation_counts("inv-123")) == (0, 1)
    invoice = asyncio.run(repository.get_invoice("inv-123"))
    assert invoice is not None
    assert invoice.status == "disputed"
