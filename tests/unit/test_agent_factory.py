from __future__ import annotations

from copy import deepcopy

import pytest
from langchain_core.tools import tool

from agentops_demo.agent import graph as graph_module
from agentops_demo.agent.graph import build_agent_with_tools, validate_runtime_config
from agentops_demo.agent.prompt import (
    BASELINE_SYSTEM_PROMPT,
    IMPROVED_SYSTEM_PROMPT,
    get_system_prompt,
)
from agentops_demo.agent.scripted import create_scripted_model
from agentops_demo.contracts.agent_config import AgentConfig


@tool
async def get_invoice(invoice_id: str) -> dict[str, str]:
    """Get an invoice."""

    return {"id": invoice_id, "status": "disputed"}


@tool
async def refund_invoice(invoice_id: str, reason: str) -> dict[str, object]:
    """Refund an invoice."""

    return {"success": True, "invoice_id": invoice_id, "reason": reason}


@tool
async def escalate_dispute(invoice_id: str, reason: str) -> dict[str, object]:
    """Escalate an invoice."""

    return {"success": True, "invoice_id": invoice_id, "reason": reason}


def test_prompt_versions_keep_policy_difference_explicit() -> None:
    assert get_system_prompt("billing-v1") == BASELINE_SYSTEM_PROMPT
    assert "disputed invoice" not in BASELINE_SYSTEM_PROMPT
    assert get_system_prompt("billing-v2") == IMPROVED_SYSTEM_PROMPT
    assert "must not be refunded" in IMPROVED_SYSTEM_PROMPT


def test_unknown_prompt_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported prompt version"):
        get_system_prompt("unknown")


def test_agent_factory_builds_compiled_graph_from_supplied_dependencies(
    agent_config: AgentConfig,
) -> None:
    validate_runtime_config(agent_config)
    graph = build_agent_with_tools(
        config=agent_config,
        model=create_scripted_model("bad"),
        tools=[get_invoice, refund_invoice, escalate_dispute],
    )

    assert graph.name == "billing_agent"


def test_agent_factory_rejects_tool_contract_drift(agent_config: AgentConfig) -> None:
    with pytest.raises(ValueError, match="billing tools must be exactly"):
        build_agent_with_tools(
            config=agent_config,
            model=create_scripted_model("bad"),
            tools=[get_invoice],
        )


@pytest.mark.parametrize(
    ("version", "expected_prompt"),
    [
        ("billing-v1", BASELINE_SYSTEM_PROMPT),
        ("billing-v2", IMPROVED_SYSTEM_PROMPT),
    ],
)
def test_agent_factory_uses_configured_prompt(
    agent_config_data: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    expected_prompt: str,
) -> None:
    agent_config_data["prompt"] = {"version": version}
    config = AgentConfig.model_validate(agent_config_data)
    captured: dict[str, object] = {}

    def capture_create_agent(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(graph_module, "create_agent", capture_create_agent)
    graph = build_agent_with_tools(
        config=config,
        model=create_scripted_model("bad"),
        tools=[get_invoice, refund_invoice, escalate_dispute],
    )

    assert graph is not None
    assert captured["system_prompt"] == expected_prompt


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("tools", {"version": "billing-mcp-v999"}, "unsupported tools version"),
        ("prompt", {"version": "billing-v999"}, "unsupported prompt version"),
        (
            "harness",
            {"framework": "other", "version": "v1"},
            "unsupported harness framework",
        ),
        (
            "harness",
            {"framework": "langgraph", "version": "v999"},
            "unsupported harness version",
        ),
    ],
)
def test_runtime_config_rejects_unsupported_identity(
    agent_config_data: dict[str, object],
    section: str,
    value: dict[str, str],
    message: str,
) -> None:
    data = deepcopy(agent_config_data)
    data[section] = value
    config = AgentConfig.model_validate(data)

    with pytest.raises(ValueError, match=message):
        build_agent_with_tools(
            config=config,
            model=create_scripted_model("bad"),
            tools=[get_invoice, refund_invoice, escalate_dispute],
        )
