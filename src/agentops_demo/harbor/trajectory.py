"""Export actual LangGraph messages as Harbor-readable ATIF without importing Harbor."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agentops_demo.agent.prompt import get_system_prompt
from agentops_demo.contracts.agent_config import AgentConfig


def to_atif(
    *, config: AgentConfig, instruction: str, messages: Sequence[BaseMessage]
) -> dict[str, Any]:
    """Keep tool responses on their calling step, as required by ATIF references."""

    steps: list[dict[str, Any]] = [
        {"step_id": 1, "source": "system", "message": get_system_prompt(config.prompt.version)}
    ]
    if not messages or not isinstance(messages[0], HumanMessage):
        steps.append({"step_id": 2, "source": "user", "message": instruction})
    calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in calls:
                raise ValueError(f"tool response has no matching call: {message.tool_call_id}")
            step = calls[message.tool_call_id]
            observation = step.setdefault("observation", {"results": []})
            observation["results"].append(
                {
                    "source_call_id": message.tool_call_id,
                    "content": (
                        message.content
                        if isinstance(message.content, str)
                        else message.text or json.dumps(message.content, sort_keys=True)
                    ),
                    "extra": {"status": message.status},
                }
            )
            continue

        if isinstance(message, AIMessage):
            step = {"step_id": len(steps) + 1, "source": "agent", "message": message.text}
            # Retain only non-secret invocation evidence from provider metadata.
            request_id = message.response_metadata.get("ResponseMetadata", {}).get("RequestId")
            if request_id:
                step["extra"] = {"bedrock_request_id": request_id}
            if message.tool_calls:
                step["tool_calls"] = []
                for call in message.tool_calls:
                    call_id = call["id"]
                    if not call_id or call_id in calls:
                        raise ValueError("tool calls must have unique, nonempty IDs")
                    calls[call_id] = step
                    step["tool_calls"].append(
                        {
                            "tool_call_id": call_id,
                            "function_name": call["name"],
                            "arguments": call["args"],
                        }
                    )
            if message.usage_metadata is not None:
                usage = message.usage_metadata
                metrics = {
                    "prompt_tokens": usage["input_tokens"],
                    "completion_tokens": usage["output_tokens"],
                }
                cached = usage.get("input_token_details", {}).get("cache_read")
                if cached is not None:
                    metrics["cached_tokens"] = cached
                step["metrics"] = metrics
        elif isinstance(message, HumanMessage | SystemMessage):
            step = {
                "step_id": len(steps) + 1,
                "source": "user" if isinstance(message, HumanMessage) else "system",
                "message": message.text,
            }
        else:
            raise ValueError(f"unsupported trajectory message: {message.type}")
        steps.append(step)

    final_metrics = {"total_steps": len(steps)}
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
        values = [step["metrics"][key] for step in steps if key in step.get("metrics", {})]
        if values:
            final_metrics[f"total_{key}"] = sum(values)
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": str(uuid4()),
        "agent": {
            "name": "langgraph-billing",
            "version": config.harness.version,
            "model_name": f"{config.model.provider}/{config.model.model_id}",
            "extra": {
                "agent_config_fingerprint": config.fingerprint(),
                "source_revision": config.agent.source_revision,
                "prompt_version": config.prompt.version,
            },
        },
        "steps": steps,
        "final_metrics": final_metrics,
    }


def write_trajectory(
    path: Path, *, config: AgentConfig, instruction: str, messages: Sequence[BaseMessage]
) -> None:
    """Write the conventional trajectory artifact alongside agent diagnostics."""

    trajectory = to_atif(config=config, instruction=instruction, messages=messages)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n")
