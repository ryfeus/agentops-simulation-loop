"""Oracle-gate the derived Phase 8C execution suite and cache exact lineage."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from .execution_suite import derive_execution_suite


def load_oracle_cache(path: Path, execution_suite_sha256: str, task_count: int) -> bool:
    """Return true only for a complete PASS cache matching this exact suite."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and value.get("schema_version") == "1"
        and value.get("status") == "PASS"
        and value.get("execution_suite_sha256") == execution_suite_sha256
        and value.get("task_count") == task_count
        and value.get("passed_tasks") == task_count
        and value.get("failed_tasks") == []
    )


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check(*, stage: Path, cache_root: Path, execute: bool = True) -> dict[str, Any]:
    suite = derive_execution_suite(stage / "suite")
    cache = cache_root / f"{suite.execution_suite_sha256}.json"
    if load_oracle_cache(cache, suite.execution_suite_sha256, int(suite.manifest["task_count"])):
        return {"status": "PASS", "cached": True, "cache": str(cache), **suite.manifest}
    if not execute:
        return {"status": "MISSING", "cached": False, "cache": str(cache), **suite.manifest}
    failures: list[dict[str, Any]] = []
    for task_id in suite.manifest["task_ids"]:
        task = suite.root / "tasks" / task_id
        completed = subprocess.run(
            [
                "harbor",
                "run",
                "-p",
                str(task),
                "-a",
                "oracle",
                "-m",
                "oracle",
                "--env",
                "docker",
                "--n-attempts",
                "1",
                "--n-concurrent",
                "1",
                "--yes",
                "--job-name",
                task_id,
                "--jobs-dir",
                str(stage / "jobs"),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        reward = None
        for reward_file in sorted((stage / "jobs" / task_id).rglob("reward.json")):
            try:
                reward = float(json.loads(reward_file.read_text())["reward"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
        if completed.returncode != 0 or reward != 1.0:
            failures.append(
                {
                    "task_id": task_id,
                    "returncode": completed.returncode,
                    "reward": reward,
                    "output": completed.stdout[-4000:],
                }
            )
    value = {
        "schema_version": "1",
        "execution_suite_sha256": suite.execution_suite_sha256,
        "task_count": suite.manifest["task_count"],
        "passed_tasks": int(suite.manifest["task_count"]) - len(failures),
        "failed_tasks": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    if value["status"] == "PASS":
        _write(cache, value)
    return {"cached": False, "cache": str(cache), **value, "suite": suite.manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    value = check(stage=args.stage, cache_root=args.cache_root, execute=not args.check_only)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
