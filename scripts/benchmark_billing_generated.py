"""Regenerate or verify the checked-in Harbor snapshot for the billing catalog."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from agentops_demo.benchmark.catalog import DEFAULT_CATALOG, render_catalog_sync
from agentops_demo.taskify.integrity import harbor_task_sha256

DEFAULT_TASKS = Path("benchmarks/billing/tasks")


def _task_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError(f"billing Harbor task snapshot is missing: {root}")
    entries = sorted(root.iterdir())
    if any(not entry.is_dir() or entry.is_symlink() for entry in entries):
        raise ValueError(f"billing Harbor task snapshot contains a non-directory entry: {root}")
    return {entry.name: harbor_task_sha256(entry) for entry in entries}


def check(*, catalog: Path, tasks: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="agentops-billing-generated-") as temporary:
        generated = Path(temporary) / "tasks"
        render_catalog_sync(catalog, generated)
        expected = _task_hashes(generated)
    actual = _task_hashes(tasks)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    changed = sorted(
        task_id for task_id in set(expected) & set(actual) if expected[task_id] != actual[task_id]
    )
    return {
        "status": "CURRENT" if not missing and not unexpected and not changed else "STALE",
        "task_count": len(expected),
        "missing": missing,
        "unexpected": unexpected,
        "changed": changed,
    }


def regenerate(*, catalog: Path, tasks: Path) -> dict[str, object]:
    entries = render_catalog_sync(catalog, tasks)
    return {"status": "RENDERED", "task_count": len(entries), "tasks": str(tasks)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    result = (
        check(catalog=args.catalog, tasks=args.tasks)
        if args.check
        else regenerate(catalog=args.catalog, tasks=args.tasks)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"CURRENT", "RENDERED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
