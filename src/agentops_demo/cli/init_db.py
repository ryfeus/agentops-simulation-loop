"""Initialize the local billing database from a normalized scenario."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from agentops_demo.billing.sqlite_repository import initialize_database
from agentops_demo.validation.scenario import load_scenario

DEFAULT_SCENARIO = Path("scenarios/disputed-refund/scenario.yaml")
DEFAULT_DATABASE = Path("data/billing.db")


async def initialize_from_scenario(scenario_path: Path, database_path: Path) -> str:
    scenario = load_scenario(scenario_path)
    await initialize_database(database_path, scenario.initial_state)
    return scenario.id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialize the local billing demo database")
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(os.getenv("BILLING_DATABASE_PATH", DEFAULT_DATABASE)),
    )
    args = parser.parse_args(argv)
    scenario_id = asyncio.run(initialize_from_scenario(args.scenario, args.database))
    print(f"Loaded scenario {scenario_id!r} into {args.database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
