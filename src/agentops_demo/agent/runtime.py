"""Reusable checkpoint-backed local runtime for the billing LangGraph agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from agentops_demo.agent.graph import build_agent
from agentops_demo.agent.model import create_model
from agentops_demo.contracts.agent_config import AgentConfig

ModelFactory = Callable[[AgentConfig], BaseChatModel]
GraphFactory = Callable[..., Awaitable[CompiledStateGraph]]


def serialize_result(
    *, config: AgentConfig, state: dict[str, Any], message_start: int = 0
) -> dict[str, Any]:
    """Return the stable response shape from a completed graph state."""

    messages: Sequence[Any] = state["messages"][message_start:]
    if not messages:
        raise RuntimeError("agent graph did not emit messages for the invocation")
    tool_calls = [
        {"name": call["name"], "arguments": call["args"]}
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    completed_tools = [
        message.name
        for message in messages
        if isinstance(message, ToolMessage) and message.name is not None
    ]
    final = messages[-1]
    final_response = final.text if isinstance(final, AIMessage) else str(final.content)
    return {
        "final_response": final_response,
        "tool_calls": tool_calls,
        "completed_tools": completed_tools,
        "agent_config_fingerprint": config.fingerprint(),
    }


class AgentSessionRuntime:
    """A process-local LangGraph runtime keyed by caller-provided thread IDs."""

    def __init__(self, *, config: AgentConfig, graph: CompiledStateGraph) -> None:
        self._config = config
        self._graph = graph

    @property
    def config(self) -> AgentConfig:
        """The immutable effective agent identity for this runtime."""

        return self._config

    @property
    def graph(self) -> CompiledStateGraph:
        """The checkpoint-backed graph for protocol adapters such as AG-UI."""

        return self._graph

    async def invoke(self, *, session_id: str, instruction: str) -> dict[str, Any]:
        """Run one turn using ``session_id`` as LangGraph's ``thread_id``."""

        if not session_id.strip():
            raise ValueError("session_id must not be blank")
        graph_config = {"configurable": {"thread_id": session_id}}
        before = await self._graph.aget_state(graph_config)
        previous_messages = before.values.get("messages", [])
        if not isinstance(previous_messages, list):
            raise RuntimeError("agent graph checkpoint has invalid messages state")
        message_start = len(previous_messages)
        state = await self._graph.ainvoke(
            {"messages": [{"role": "user", "content": instruction}]},
            config=graph_config,
        )
        return serialize_result(
            config=self._config,
            state=state,
            message_start=message_start,
        )


async def create_session_runtime(
    *,
    config: AgentConfig,
    mcp_url: str,
    model_factory: ModelFactory = create_model,
    graph_factory: GraphFactory = build_agent,
) -> AgentSessionRuntime:
    """Build one model, graph, and in-memory checkpointer for a local process."""

    model = model_factory(config)
    graph = await graph_factory(
        config=config,
        model=model,
        mcp_url=mcp_url,
        checkpointer=InMemorySaver(),
    )
    return AgentSessionRuntime(config=config, graph=graph)


async def invoke_once(
    *,
    config: AgentConfig,
    instruction: str,
    mcp_url: str,
    model_factory: ModelFactory = create_model,
    graph_factory: GraphFactory = build_agent,
) -> dict[str, Any]:
    """Invoke the configured graph once without retaining conversation state."""

    model = model_factory(config)
    graph = await graph_factory(config=config, model=model, mcp_url=mcp_url)
    state = await graph.ainvoke({"messages": [{"role": "user", "content": instruction}]})
    return serialize_result(config=config, state=state)
