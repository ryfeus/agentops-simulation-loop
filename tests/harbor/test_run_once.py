from __future__ import annotations

import inspect
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agentops_demo.cli import run_once
from agentops_demo.contracts.agent_config import AgentConfig


class FakeGraph:
    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        assert state["messages"][0]["content"] == "instruction"
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "1", "name": "get_invoice", "args": {"invoice_id": "inv-123"}}
                    ],
                ),
                ToolMessage(content="{}", tool_call_id="1", name="get_invoice"),
                AIMessage(content="done"),
            ]
        }


async def test_one_shot_runner_emits_config_identity_and_trajectory(
    agent_config: AgentConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_once, "create_model", lambda config: object())

    async def fake_build_agent(**kwargs: object) -> FakeGraph:
        assert kwargs["config"] == agent_config
        return FakeGraph()

    monkeypatch.setattr(run_once, "build_agent", fake_build_agent)
    result = await run_once.invoke_once(
        config=agent_config, instruction="instruction", mcp_url="http://localhost/mcp"
    )

    assert result == {
        "final_response": "done",
        "tool_calls": [{"name": "get_invoice", "arguments": {"invoice_id": "inv-123"}}],
        "completed_tools": ["get_invoice"],
        "agent_config_fingerprint": agent_config.fingerprint(),
    }


def test_one_shot_runner_has_no_verifier_or_state_dependency() -> None:
    source = inspect.getsource(run_once).lower()
    assert "sqlite" not in source
    assert "expected_invariant" not in source
    assert "validation.scenario" not in source
