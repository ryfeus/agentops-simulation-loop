"""List only Phase 6 Harbor EC2 workers; this command never changes AWS state."""

from __future__ import annotations

import argparse
import json
import os

import boto3

from scripts.harbor_ec2 import EXPECTED_REGION
from scripts.harbor_ec2_controller import discover_workers, require_account


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    session = boto3.Session(
        profile_name=os.getenv("AWS_PROFILE", "default"), region_name=EXPECTED_REGION
    )
    require_account(session.client("sts"))
    instances = discover_workers(session.client("ec2"), args.run_id)
    print(json.dumps(instances, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
