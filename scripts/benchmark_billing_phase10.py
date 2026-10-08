"""Generate, check, render, and accept the deterministic Phase 10 billing corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentops_demo.benchmark.phase10_variants import PHASE10, check, generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PHASE10)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "generate",
        "check",
        "stats",
        "render-smoke",
        "render",
        "oracle",
        "calibrate",
        "acceptance",
    ):
        subcommands.add_parser(name)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            manifest = generate(args.root)
            result = {
                "status": "GENERATED",
                "task_count": manifest["task_count"],
                "corpus_sha256": manifest["corpus_sha256"],
            }
        elif args.command == "check":
            result = check(args.root)
        elif args.command == "stats":
            if check(args.root)["status"] != "CURRENT":
                raise ValueError("Phase 10 snapshot is stale; run generate")
            result = json.loads((args.root / "generated" / "corpus-stats.json").read_text())
        elif args.command in {"render-smoke", "render", "oracle", "calibrate", "acceptance"}:
            from agentops_demo.benchmark.phase10_acceptance import run_command

            result = run_command(args.command, args.root)
        else:
            parser.error("unknown command")
    except (OSError, ValueError, RuntimeError) as exc:
        result = {"status": "FAIL", "error": str(exc)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"STALE", "FAIL", "MISSING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
