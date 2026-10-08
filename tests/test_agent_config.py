from __future__ import annotations

import copy
import re

import pytest
from pydantic import ValidationError

from agentops_demo.contracts.agent_config import AgentConfig


def test_valid_config_parses(agent_config_data: dict[str, object]) -> None:
    config = AgentConfig.model_validate(agent_config_data)

    assert config.model.provider == "bedrock"
    assert config.harness.framework == "langgraph"


def test_fingerprint_is_deterministic(agent_config_data: dict[str, object]) -> None:
    first = AgentConfig.model_validate(agent_config_data)
    reordered = AgentConfig.model_validate(dict(reversed(list(agent_config_data.items()))))

    assert first.fingerprint() == reordered.fingerprint()
    assert re.fullmatch(r"[0-9a-f]{64}", first.fingerprint())


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("agent", "source_revision", "def456"),
        ("model", "model_id", "model-v2"),
        ("prompt", "version", "billing-v2"),
        ("tools", "version", "billing-mcp-v2"),
        ("harness", "version", "v2"),
    ],
)
def test_versioned_field_changes_fingerprint(
    agent_config_data: dict[str, object], section: str, field: str, replacement: str
) -> None:
    changed_data = copy.deepcopy(agent_config_data)
    changed_data[section][field] = replacement  # type: ignore[index]

    original = AgentConfig.model_validate(agent_config_data)
    changed = AgentConfig.model_validate(changed_data)

    assert original.fingerprint() != changed.fingerprint()


def test_missing_section_is_rejected(agent_config_data: dict[str, object]) -> None:
    del agent_config_data["tools"]

    with pytest.raises(ValidationError):
        AgentConfig.model_validate(agent_config_data)


def test_blank_version_is_rejected(agent_config_data: dict[str, object]) -> None:
    agent_config_data["prompt"] = {"version": "   "}

    with pytest.raises(ValidationError):
        AgentConfig.model_validate(agent_config_data)


def test_secret_field_is_rejected(agent_config_data: dict[str, object]) -> None:
    agent_config_data["model"]["api_key"] = "not-allowed"  # type: ignore[index]

    with pytest.raises(ValidationError):
        AgentConfig.model_validate(agent_config_data)
