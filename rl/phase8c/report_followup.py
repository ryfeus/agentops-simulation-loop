"""Merge immutable Phase 8C canonical and follow-up rollout evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .report import build_reports


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def merge(*, rollout_paths: list[Path], suite_path: Path, destination: Path) -> dict[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    rows = [row for path in rollout_paths for row in _rows(path)]
    identities = {
        (str(row.get("run_id")), str(row.get("task_id")), int(row.get("attempt", -1)))
        for row in rows
    }
    if len(identities) != len(rows) or any(identity[2] < 0 for identity in identities):
        raise ValueError("duplicate or invalid provenance-aware rollout identity")
    source_runs = sorted({str(row["run_id"]) for row in rows})
    attempts = max((int(row["attempt"]) for row in rows), default=-1) + 1
    reports = build_reports(rows, attempts_per_task=attempts, destination=destination)
    for name in ("task-summary.json", "phase9-candidates.json"):
        source = destination / name
        target = destination / f"combined-{name}"
        source.replace(target)
    (destination / "combined-rollouts.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
    )
    value = {
        "schema_version": "1",
        "source_runs": source_runs,
        "execution_suite_sha256": suite["execution_suite_sha256"],
        "task_count": len(reports["tasks"]),
        "attempts_per_task": attempts,
    }
    (destination / "combined-summary.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("rollouts", type=Path, nargs="+")
    args = parser.parse_args()
    print(
        json.dumps(
            merge(rollout_paths=args.rollouts, suite_path=args.suite, destination=args.output),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
