"""Generate deterministic Harbor candidate configs from the current Git revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.harbor.provenance import current_revision, require_clean_worktree


def candidate_config(revision: str, mode: str) -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "agent": {"source_revision": revision},
            "model": {"provider": "scripted", "model_id": mode},
            "prompt": {"version": "billing-v1"},
            "tools": {"version": "billing-mcp-v1"},
            "harness": {"framework": "langgraph", "version": "v1"},
        }
    )


def write_configs(output_dir: Path, revision: str) -> dict[str, AgentConfig]:
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = {mode: candidate_config(revision, mode) for mode in ("bad", "correct", "noop")}
    for mode, config in configs.items():
        content = json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True)
        (output_dir / f"{mode}.json").write_text(f"{content}\n")
    return configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".harbor/configs"))
    args = parser.parse_args(argv)
    require_clean_worktree()
    revision = current_revision()
    configs = write_configs(args.output_dir, revision)
    for mode, config in configs.items():
        print(f"{mode}: {config.fingerprint()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
