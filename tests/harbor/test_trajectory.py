from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from harbor.models.trajectories.trajectory import Trajectory
from harbor.viewer.server import create_app
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentops_demo.agent.prompt import get_system_prompt
from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.harbor.trajectory import to_atif, write_trajectory


def test_atif_preserves_messages_parallel_calls_and_actual_tool_responses(
    agent_config: AgentConfig,
) -> None:
    messages = [
        HumanMessage(content="Check both invoices"),
        AIMessage(
            content=[{"type": "text", "text": "Checking both."}],
            tool_calls=[
                {"id": "a", "name": "get_invoice", "args": {"invoice_id": "inv-123"}},
                {"id": "b", "name": "get_invoice", "args": {"invoice_id": "inv-999"}},
            ],
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "input_token_details": {"cache_read": 25},
            },
            response_metadata={
                "ResponseMetadata": {
                    "RequestId": "real-request-id",
                    "HTTPHeaders": {"authorization": "must-not-export"},
                },
            },
        ),
        ToolMessage(content="Not found", tool_call_id="b", status="error"),
        ToolMessage(content=[{"type": "text", "text": '{"status":"disputed"}'}], tool_call_id="a"),
        AIMessage(content="Invoice inv-123 is disputed; inv-999 was not found."),
    ]

    trajectory = Trajectory.model_validate(
        to_atif(config=agent_config, instruction="Check both invoices", messages=messages)
    )

    assert trajectory.schema_version == "ATIF-v1.7"
    assert trajectory.session_id
    assert trajectory.agent.model_name == "bedrock/model-v1"
    assert trajectory.agent.extra["agent_config_fingerprint"] == agent_config.fingerprint()
    assert [step.step_id for step in trajectory.steps] == [1, 2, 3, 4]
    assert [step.source for step in trajectory.steps] == ["system", "user", "agent", "agent"]
    assert trajectory.steps[0].message == get_system_prompt(agent_config.prompt.version)
    assert trajectory.steps[1].message == "Check both invoices"
    action = trajectory.steps[2]
    assert action.message == "Checking both."
    assert action.extra == {"bedrock_request_id": "real-request-id"}
    assert "must-not-export" not in trajectory.model_dump_json()
    assert [
        (call.tool_call_id, call.function_name, call.arguments) for call in action.tool_calls
    ] == [
        ("a", "get_invoice", {"invoice_id": "inv-123"}),
        ("b", "get_invoice", {"invoice_id": "inv-999"}),
    ]
    assert [(result.source_call_id, result.content) for result in action.observation.results] == [
        ("b", "Not found"),
        ("a", '{"status":"disputed"}'),
    ]
    assert action.observation.results[0].extra == {"status": "error"}
    assert trajectory.steps[-1].message == messages[-1].content
    assert trajectory.final_metrics.total_prompt_tokens == 100
    assert trajectory.final_metrics.total_completion_tokens == 10
    assert trajectory.final_metrics.total_cached_tokens == 25


def test_atif_does_not_invent_tool_results_or_usage(agent_config: AgentConfig) -> None:
    trajectory = Trajectory.model_validate(
        to_atif(
            config=agent_config,
            instruction="Look up the invoice",
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "pending", "name": "get_invoice", "args": {"invoice_id": "inv-123"}}
                    ],
                )
            ],
        )
    )
    assert trajectory.steps[1].message == "Look up the invoice"
    assert trajectory.steps[-1].observation is None
    assert trajectory.steps[-1].metrics is None
    assert trajectory.final_metrics.total_prompt_tokens is None


def test_atif_rejects_unlinked_tool_response(agent_config: AgentConfig) -> None:
    with pytest.raises(ValueError, match="no matching call"):
        to_atif(
            config=agent_config,
            instruction="Check invoice",
            messages=[ToolMessage(content="{}", tool_call_id="missing")],
        )


def test_harbor_viewer_reads_exported_trajectory(tmp_path: Path, agent_config: AgentConfig) -> None:
    output = tmp_path / "jobs" / "demo" / "trial" / "agent" / "trajectory.json"
    write_trajectory(
        output,
        config=agent_config,
        instruction="Hello",
        messages=[HumanMessage(content="Hello"), AIMessage(content="Hi")],
    )
    with TestClient(create_app(tmp_path / "jobs")) as client:
        response = client.get("/api/jobs/demo/trials/trial/trajectory")
    assert response.status_code == 200
    trajectory = Trajectory.model_validate(response.json())
    assert trajectory.steps[-1].message == "Hi"
