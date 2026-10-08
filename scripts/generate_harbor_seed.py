"""Generate the disputed-refund Harbor SQLite seed from the canonical Scenario."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import tempfile
from pathlib import Path

from agentops_demo.billing.sqlite_repository import initialize_database
from agentops_demo.validation.scenario import load_scenario

DEFAULT_SCENARIO = Path("scenarios/disputed-refund/scenario.yaml")
DEFAULT_OUTPUT = Path("benchmarks/disputed-refund/environment/seed.sql")


async def render_seed(scenario_path: Path) -> str:
    scenario = load_scenario(scenario_path)
    with tempfile.TemporaryDirectory(prefix="agentops-seed-") as temporary:
        database = Path(temporary) / "billing.db"
        await initialize_database(database, scenario.initial_state)
        with sqlite3.connect(database) as connection:
            lines = list(connection.iterdump())
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    content = asyncio.run(render_seed(args.scenario))
    if args.check:
        if not args.output.is_file() or args.output.read_text() != content:
            raise SystemExit(f"Harbor seed is stale: run with --output {args.output}")
        print(f"OK {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
