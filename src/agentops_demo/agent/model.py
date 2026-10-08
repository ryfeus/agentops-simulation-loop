"""Minimal external-model construction kept separate from graph logic."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agentops_demo.contracts.agent_config import AgentConfig


def create_model(config: AgentConfig, *, region_name: str | None = None) -> BaseChatModel:
    """Create the configured external model, importing optional providers lazily."""

    if config.model.provider == "scripted":
        from agentops_demo.agent.scripted import create_scripted_model

        if config.model.model_id not in {"bad", "correct", "noop"}:
            raise ValueError(f"unsupported scripted model {config.model.model_id!r}")
        return create_scripted_model(config.model.model_id)

    if config.model.provider != "bedrock":
        raise ValueError(f"unsupported model provider {config.model.provider!r}")
    try:
        from langchain_aws import ChatBedrockConverse
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Bedrock support is not installed; run with the 'bedrock' project extra"
        ) from exc

    kwargs: dict[str, Any] = {"model_id": config.model.model_id}
    if region_name:
        kwargs["region_name"] = region_name
    return ChatBedrockConverse(**kwargs)
