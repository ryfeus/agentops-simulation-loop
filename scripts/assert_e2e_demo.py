"""Independently verify a final Phase 7 evidence bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentops_demo.demo.contracts import DemoReport
from agentops_demo.demo.verification import verify_demo_bundle


def assert_demo(root: Path) -> DemoReport:
    return verify_demo_bundle(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    assert_demo(args.run)
    print("END-TO-END AGENTOPS LOOP: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
