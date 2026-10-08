"""Derive the explicit shared-verifier execution task used only by Phase 8B."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from agentops_demo.taskify.integrity import harbor_task_sha256, validate_harbor_task

EXECUTION_PROFILE = "trl-harbor-shared-verifier"


@dataclass(frozen=True)
class ExecutionTask:
    """Lineage for a deterministic Phase 8B execution-only task copy."""

    path: Path
    task_id: str
    base_task_sha256: str
    execution_task_sha256: str
    execution_profile: str = EXECUTION_PROFILE


def _task_config(path: Path) -> dict[str, object]:
    with path.open("rb") as source:
        value = tomllib.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"task TOML is not a table: {path}")
    return value


def _shared_verifier_toml(path: Path) -> str:
    """Make the sole permitted deterministic task.toml change."""

    source = path.read_text(encoding="utf-8")
    parsed = _task_config(path)
    expected = copy.deepcopy(parsed)
    verifier = expected.get("verifier")
    if not isinstance(verifier, dict) or verifier.get("environment_mode") != "separate":
        raise ValueError("Phase 8B canonical task must declare a separate verifier")
    verifier["environment_mode"] = "shared"
    verifier.pop("environment", None)

    lines: list[str] = []
    section = ""
    skipping_verifier_environment = False
    replaced_mode = False
    removed_environment = False
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            skipping_verifier_environment = section == "verifier.environment"
            if skipping_verifier_environment:
                removed_environment = True
                continue
        if skipping_verifier_environment:
            continue
        if section == "verifier" and stripped.startswith("environment_mode"):
            newline = "\n" if line.endswith("\n") else ""
            lines.append('environment_mode = "shared"' + newline)
            replaced_mode = True
        else:
            lines.append(line)
    if not replaced_mode or not removed_environment:
        raise ValueError("Phase 8B could not deterministically derive the shared verifier task")

    rendered = "".join(lines)
    derived = tomllib.loads(rendered)
    if derived != expected:
        raise ValueError("shared verifier derivation changed task.toml beyond verifier topology")
    return rendered


def _assert_semantic_files_unchanged(base: Path, execution: Path) -> None:
    base_files = {path.relative_to(base) for path in base.rglob("*") if path.is_file()}
    execution_files = {
        path.relative_to(execution) for path in execution.rglob("*") if path.is_file()
    }
    if base_files != execution_files:
        raise ValueError("execution task file set differs from canonical task")
    for relative in sorted(base_files):
        if relative == Path("task.toml"):
            continue
        if (base / relative).read_bytes() != (execution / relative).read_bytes():
            raise ValueError(f"execution task changed canonical semantic file: {relative}")


def derive_execution_task(base: Path, output: Path) -> ExecutionTask:
    """Copy *base* and change only its verifier topology for TRL HarborEnv."""

    validate_harbor_task(base)
    if output.exists():
        raise FileExistsError(f"execution task output already exists: {output}")
    shutil.copytree(base, output, symlinks=False)
    task_toml = output / "task.toml"
    task_toml.write_text(_shared_verifier_toml(task_toml), encoding="utf-8")
    validate_harbor_task(output)
    _assert_semantic_files_unchanged(base, output)
    return ExecutionTask(
        path=output,
        task_id=base.name,
        base_task_sha256=harbor_task_sha256(base),
        execution_task_sha256=harbor_task_sha256(output),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    identity = derive_execution_task(args.base, args.output)
    value = asdict(identity)
    value["path"] = str(identity.path)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
