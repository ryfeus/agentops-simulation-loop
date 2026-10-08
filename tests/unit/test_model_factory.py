from __future__ import annotations

import sys

import pytest
from langchain_core.messages import HumanMessage

from agentops_demo.agent.model import create_model
from agentops_demo.agent.scripted import ScriptedBillingModel
from agentops_demo.contracts.agent_config import AgentConfig
from scripts.generate_agentcore_config import production_config


def test_unsupported_provider_is_rejected(agent_config_data: dict[str, object]) -> None:
    agent_config_data["model"] = {"provider": "unsupported", "model_id": "model-v1"}
    config = AgentConfig.model_validate(agent_config_data)

    with pytest.raises(ValueError, match="unsupported model provider"):
        create_model(config)


def test_missing_optional_bedrock_extra_has_actionable_error(
    agent_config: AgentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "langchain_aws", None)

    with pytest.raises(RuntimeError, match=r"bedrock.*extra"):
        create_model(agent_config)


@pytest.mark.parametrize("model_id", ["bad", "correct", "noop"])
def test_scripted_model_is_selected_from_agent_config(
    agent_config_data: dict[str, object], model_id: str
) -> None:
    agent_config_data["model"] = {"provider": "scripted", "model_id": model_id}

    model = create_model(AgentConfig.model_validate(agent_config_data))

    assert isinstance(model, ScriptedBillingModel)
    assert model.mode == model_id


def test_noop_scripted_model_emits_no_tool_calls() -> None:
    response = create_model(
        AgentConfig.model_validate(
            {
                "agent": {"source_revision": "test"},
                "model": {"provider": "scripted", "model_id": "noop"},
                "prompt": {"version": "billing-v1"},
                "tools": {"version": "billing-mcp-v1"},
                "harness": {"framework": "langgraph", "version": "v1"},
            }
        )
    ).invoke([HumanMessage(content="Refund inv-201.")])

    assert response.tool_calls == []


def test_unknown_scripted_model_is_rejected(agent_config_data: dict[str, object]) -> None:
    agent_config_data["model"] = {"provider": "scripted", "model_id": "unknown"}

    with pytest.raises(ValueError, match="unsupported scripted model"):
        create_model(AgentConfig.model_validate(agent_config_data))


def test_production_config_allows_only_explicit_bad_scripted_calibration() -> None:
    assert production_config("a" * 40, "bad", "scripted").model.provider == "scripted"
    with pytest.raises(ValueError, match="only model_id=bad"):
        production_config("a" * 40, "correct", "scripted")
    with pytest.raises(ValueError, match="unsupported production model provider"):
        production_config("a" * 40, "model", "other")
