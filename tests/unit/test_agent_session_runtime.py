from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver

from agentops_demo.agent.graph import build_agent_with_tools
from agentops_demo.cli import run_once


class SessionAwareFakeGraph:
    def __init__(self) -> None:
        self.instructions: dict[str, list[str]] = {}
        self.messages: dict[str, list[object]] = {}

    async def aget_state(self, config: dict[str, dict[str, str]]) -> SimpleNamespace:
        session_id = config["configurable"]["thread_id"]
        return SimpleNamespace(values={"messages": self.messages.get(session_id, [])})

    async def ainvoke(
        self, state: dict[str, Any], *, config: dict[str, dict[str, str]]
    ) -> dict[str, Any]:
        session_id = config["configurable"]["thread_id"]
        history = self.instructions.setdefault(session_id, [])
        history.append(state["messages"][0]["content"])
        messages = self.messages.setdefault(session_id, [])
        messages.extend(
            [
                state["messages"][0],
                AIMessage(content=" | ".join(history)),
            ]
        )
        return {"messages": messages}


class HistoryCountingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: list[BaseTool], **_kwargs: object) -> HistoryCountingModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **_kwargs: object,
    ) -> ChatResult:
        del stop, run_manager
        count = sum(message.type in {"human", "user"} for message in messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(count)))])


@tool
async def get_invoice(invoice_id: str) -> dict[str, str]:
    """Get an invoice."""

    return {"id": invoice_id, "status": "disputed"}


@tool
async def refund_invoice(invoice_id: str, reason: str) -> dict[str, str]:
    """Refund an invoice."""

    return {"id": invoice_id, "reason": reason}


@tool
async def escalate_dispute(invoice_id: str, reason: str) -> dict[str, str]:
    """Escalate a dispute."""

    return {"id": invoice_id, "reason": reason}


@pytest.mark.asyncio
async def test_session_runtime_uses_agentcore_session_as_langgraph_thread(agent_config) -> None:
    graph = SessionAwareFakeGraph()
    runtime = run_once.AgentSessionRuntime(config=agent_config, graph=graph)  # type: ignore[arg-type]

    first = await runtime.invoke(session_id="session-a", instruction="remember inv-123")
    second = await runtime.invoke(session_id="session-a", instruction="refund it")
    isolated = await runtime.invoke(session_id="session-b", instruction="what did I say?")

    assert first["final_response"] == "remember inv-123"
    assert second["final_response"] == "remember inv-123 | refund it"
    assert isolated["final_response"] == "what did I say?"


@pytest.mark.asyncio
async def test_in_memory_checkpoint_retains_only_the_matching_session_history(agent_config) -> None:
    graph = build_agent_with_tools(
        config=agent_config,
        model=HistoryCountingModel(responses=[]),
        tools=[get_invoice, refund_invoice, escalate_dispute],
        checkpointer=InMemorySaver(),
    )
    runtime = run_once.AgentSessionRuntime(config=agent_config, graph=graph)

    first = await runtime.invoke(session_id="session-a", instruction="first")
    second = await runtime.invoke(session_id="session-a", instruction="second")
    isolated = await runtime.invoke(session_id="session-b", instruction="first")
    assert first["final_response"] == "1"
    assert second["final_response"] == "2"
    assert isolated["final_response"] == "1"


@pytest.mark.asyncio
async def test_session_runtime_factory_builds_one_in_memory_checkpointer(
    agent_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = SessionAwareFakeGraph()
    captured: dict[str, object] = {}
    monkeypatch.setattr(run_once, "create_model", lambda _config: object())

    async def fake_build_agent(**kwargs: object) -> SessionAwareFakeGraph:
        captured.update(kwargs)
        return graph

    monkeypatch.setattr(run_once, "build_agent", fake_build_agent)
    runtime = await run_once.create_session_runtime(
        config=agent_config,
        mcp_url="http://localhost/mcp",
    )

    assert isinstance(captured["checkpointer"], InMemorySaver)
    assert await runtime.invoke(session_id="session-a", instruction="hello") == {
        "final_response": "hello",
        "tool_calls": [],
        "completed_tools": [],
        "agent_config_fingerprint": agent_config.fingerprint(),
    }


@pytest.mark.asyncio
async def test_session_runtime_rejects_blank_session_id(agent_config) -> None:
    runtime = run_once.AgentSessionRuntime(
        config=agent_config,
        graph=SessionAwareFakeGraph(),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="must not be blank"):
        await runtime.invoke(session_id="   ", instruction="hello")
