"""Derive a public-agent/no-network-verifier task without altering benchmark semantics."""

from __future__ import annotations

import shutil
from pathlib import Path

from agentops_demo.harbor.execution_profile import HarborExecutionProfile
from agentops_demo.taskify.integrity import harbor_task_sha256

_ALLOWED_CHANGED = {"task.toml", "environment/requirements.txt", "execution-profile.json"}


def derive_bedrock_task(base: Path, output: Path) -> HarborExecutionProfile:
    if output.exists():
        raise FileExistsError(f"derived task already exists: {output}")
    shutil.copytree(base, output, symlinks=False)
    task = output / "task.toml"
    content = task.read_text()
    content = content.replace('network_mode = "no-network"', 'network_mode = "public"', 1)
    content = content.replace("timeout_sec = 120", "timeout_sec = 300", 1)
    task.write_text(content)
    profile = HarborExecutionProfile(
        name="bedrock",
        base_task_sha256=harbor_task_sha256(base),
        execution_task_sha256=harbor_task_sha256(output),
        agent_network_mode="public",
        verifier_network_mode="no-network",
        credential_mode="sts-assume-role",
    )
    (output / "execution-profile.json").write_text(profile.model_dump_json(indent=2) + "\n")
    return profile


def validate_bedrock_derivation(base: Path, derived: Path) -> None:
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base).as_posix()
        other = derived / relative
        if not other.is_file():
            raise ValueError(f"derived task omits {relative}")
        if relative not in _ALLOWED_CHANGED and path.read_bytes() != other.read_bytes():
            raise ValueError(f"derived task changed protected benchmark file {relative}")
