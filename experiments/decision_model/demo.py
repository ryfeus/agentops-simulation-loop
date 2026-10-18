"""Offline terminal demo of saved evidence; no model load or AWS calls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("reports/decision-model"))
    args = parser.parse_args()
    with (args.results / "metrics.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["slice"] == "all"]
    print("Fixed decision replay (deferral is not task completion):")
    for row in rows:
        print(
            f"{row['arm']:20} accuracy {float(row['accuracy']):6.1%}  "
            f"prohibited proposals {row['prohibited_proposals']:>3}  "
            f"defer {float(row['defer_rate']):6.1%}"
        )
    examples = args.results / "examples.json"
    if examples.exists():
        for example in json.loads(examples.read_text()):
            print("\n" + json.dumps(example, indent=2))
    print("\nSee reports/decision-model-results.md for execution outcomes and limitations.")


if __name__ == "__main__":
    main()
