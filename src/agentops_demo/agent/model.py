"""Minimal external-model construction kept separate from graph logic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from agentops_demo.contracts.agent_config import AgentConfig


def create_model(
    config: AgentConfig, *, region_name: str | None = None, aws_session: Any | None = None
) -> BaseChatModel:
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
    if value := os.getenv("BILLING_MODEL_TEMPERATURE"):
        temperature = float(value)
        if not 0 <= temperature <= 1:
            raise ValueError("temperature must be between zero and one")
        kwargs["temperature"] = temperature
    if value := os.getenv("BILLING_MODEL_MAX_TOKENS"):
        if int(value) < 1:
            raise ValueError("max tokens must be positive")
        kwargs["max_tokens"] = int(value)
    if value := os.getenv("BILLING_MODEL_TOTAL_ATTEMPTS"):
        from botocore.config import Config

        if int(value) < 1:
            raise ValueError("total attempts must be positive")
        kwargs["config"] = Config(retries={"mode": "standard", "total_max_attempts": int(value)})
    if token_file := os.getenv("AWS_BEARER_TOKEN_BEDROCK_FILE"):
        token = Path(token_file).read_text().strip()
        if not token or any(character.isspace() for character in token):
            raise ValueError("Bedrock token file must contain one nonempty token")
        # Pass a redacted secret directly to the client; never put the token in
        # Harbor command arguments, AgentConfig, or trajectory metadata.
        kwargs["bedrock_api_key"] = SecretStr(token)
    if region_name:
        kwargs["region_name"] = region_name
    if aws_session is not None:
        client_kwargs: dict[str, Any] = {}
        if region_name:
            client_kwargs["region_name"] = region_name
        if "config" in kwargs:
            client_kwargs["config"] = kwargs["config"]
        kwargs["client"] = aws_session.client("bedrock-runtime", **client_kwargs)
        kwargs["bedrock_client"] = aws_session.client("bedrock", **client_kwargs)
    return ChatBedrockConverse(**kwargs)
