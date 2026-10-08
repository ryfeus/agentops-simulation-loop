"""Assert the two reports that constitute a successful Phase 6 demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentops_demo.scale.contracts import CandidateGate


def _report(path: Path) -> CandidateGate:
    return CandidateGate.model_validate_json(path.read_text())


def assert_acceptance(parity_path: Path, scale_path: Path) -> None:
    parity = _report(parity_path)
    scale = _report(scale_path)
    if (
        parity.purpose != "parity"
        or not parity.accepted
        or parity.eligible is not None
        or parity.local_ec2_parity is not True
    ):
        raise ValueError("Phase 6 parity acceptance report did not pass")
    known_good = next((item for item in scale.aggregates if item.name == "known-good"), None)
    if (
        scale.purpose != "candidate_gate"
        or not scale.accepted
        or scale.eligible is not True
        or known_good is None
        or (known_good.requested, known_good.retained, known_good.completed) != (4, 4, 4)
        or known_good.infrastructure_failures != 0
        or not scale.cleanup.passed
    ):
        raise ValueError("Phase 6 scale acceptance report did not pass")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--scale", type=Path, required=True)
    args = parser.parse_args(argv)
    assert_acceptance(args.parity, args.scale)
    print("PHASE 6 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
