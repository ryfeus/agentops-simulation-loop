"""Side-effect-free factories for the LangGraph-backed billing agent."""

from __future__ import annotations

from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from agentops_demo.agent.prompt import get_system_prompt
from agentops_demo.contracts.agent_config import AgentConfig

EXPECTED_TOOL_NAMES = {"get_invoice", "refund_invoice", "escalate_dispute"}
SUPPORTED_HARNESS_FRAMEWORK = "langgraph"
SUPPORTED_HARNESS_VERSION = "v1"
SUPPORTED_TOOLS_VERSION = "billing-mcp-v1"


def validate_runtime_config(config: AgentConfig) -> None:
    """Reject configuration identities this runtime cannot faithfully execute."""

    if config.harness.framework != SUPPORTED_HARNESS_FRAMEWORK:
        raise ValueError(
            f"unsupported harness framework {config.harness.framework!r}; "
            f"expected {SUPPORTED_HARNESS_FRAMEWORK!r}"
        )
    if config.harness.version != SUPPORTED_HARNESS_VERSION:
        raise ValueError(
            f"unsupported harness version {config.harness.version!r}; "
            f"expected {SUPPORTED_HARNESS_VERSION!r}"
        )
    if config.tools.version != SUPPORTED_TOOLS_VERSION:
        raise ValueError(
            f"unsupported tools version {config.tools.version!r}; "
            f"expected {SUPPORTED_TOOLS_VERSION!r}"
        )
    get_system_prompt(config.prompt.version)


def build_agent_with_tools(
    *,
    config: AgentConfig,
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build the graph from injected model and MCP-shaped LangChain tools."""

    validate_runtime_config(config)
    tool_names = {tool.name for tool in tools}
    if tool_names != EXPECTED_TOOL_NAMES:
        raise ValueError(
            f"billing tools must be exactly {sorted(EXPECTED_TOOL_NAMES)}; "
            f"received {sorted(tool_names)}"
        )
    return create_agent(
        model=model,
        tools=list(tools),
        system_prompt=get_system_prompt(config.prompt.version),
        name="billing_agent",
        checkpointer=checkpointer,
    )


async def build_agent(
    *,
    config: AgentConfig,
    model: BaseChatModel,
    mcp_url: str,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Load billing tools through MCP HTTP and build the reusable graph."""

    validate_runtime_config(config)
    client = MultiServerMCPClient(
        {
            "billing": {
                "transport": "http",
                "url": mcp_url,
            }
        }
    )
    tools = await client.get_tools()
    return build_agent_with_tools(
        config=config,
        model=model,
        tools=tools,
        checkpointer=checkpointer,
    )
