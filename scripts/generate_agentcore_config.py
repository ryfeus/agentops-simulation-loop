"""Generate the exact production AgentConfig from a clean Git revision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agentops_demo.contracts.agent_config import AgentConfig
from scripts.generate_harbor_configs import current_revision, require_clean_worktree

DEFAULT_OUTPUT = Path(".agentcore/config/agent-config.json")
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
SUPPORTED_PROVIDERS = {"bedrock", "scripted"}


def production_config(revision: str, model_id: str, model_provider: str = "bedrock") -> AgentConfig:
    if model_provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"unsupported production model provider: {model_provider}")
    if model_provider == "scripted" and model_id != "bad":
        raise ValueError("production scripted calibration supports only model_id=bad")
    return AgentConfig.model_validate(
        {
            "agent": {"source_revision": revision},
            "model": {"provider": model_provider, "model_id": model_id},
            "prompt": {"version": "billing-v1"},
            "tools": {"version": "billing-mcp-v1"},
            "harness": {"framework": "langgraph", "version": "v1"},
        }
    )


def write_config(output: Path, model_id: str, model_provider: str = "bedrock") -> AgentConfig:
    require_clean_worktree()
    revision = current_revision()
    config = production_config(revision, model_id, model_provider)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", DEFAULT_MODEL))
    parser.add_argument("--model-provider", default=os.getenv("MODEL_PROVIDER", "bedrock"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    config = write_config(args.output, args.model_id, args.model_provider)
    print(f"Source revision: {config.agent.source_revision}")
    print(f"AgentConfig fingerprint: {config.fingerprint()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
