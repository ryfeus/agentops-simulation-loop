"""Terminate only explicitly selected, Phase 6-tagged ephemeral Harbor workers."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from typing import Any

import boto3

from scripts.harbor_ec2 import EXPECTED_REGION
from scripts.harbor_ec2_controller import discover_workers, require_account, tagged_for_run


def eligible_for_cleanup(
    instances: list[dict[str, Any]],
    *,
    run_id: str | None,
    older_than_hours: float | None,
    now: dt.datetime,
) -> list[str]:
    if (run_id is None) == (older_than_hours is None):
        raise ValueError("provide exactly one of --run-id or --older-than-hours")
    if older_than_hours is not None and older_than_hours <= 0:
        raise ValueError("--older-than-hours must be positive")
    selected: list[str] = []
    cutoff = now - dt.timedelta(hours=older_than_hours or 0)
    for instance in instances:
        instance_id = instance.get("InstanceId")
        if not isinstance(instance_id, str):
            continue
        tags = {item.get("Key"): item.get("Value") for item in instance.get("Tags", [])}
        if tags.get("Project") != "agentops-demo" or tags.get("Phase") != "6":
            continue
        if run_id is not None:
            if tagged_for_run(instance, run_id):
                selected.append(instance_id)
            continue
        launched = instance.get("LaunchTime")
        if isinstance(launched, dt.datetime) and launched <= cutoff:
            selected.append(instance_id)
    return sorted(selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--run-id")
    selected.add_argument("--older-than-hours", type=float)
    parser.add_argument(
        "--yes", action="store_true", help="actually terminate the selected workers"
    )
    args = parser.parse_args(argv)
    session = boto3.Session(
        profile_name=os.getenv("AWS_PROFILE", "default"), region_name=EXPECTED_REGION
    )
    require_account(session.client("sts"))
    ec2 = session.client("ec2")
    instances = discover_workers(ec2, args.run_id)
    ids = eligible_for_cleanup(
        instances,
        run_id=args.run_id,
        older_than_hours=args.older_than_hours,
        now=dt.datetime.now(dt.UTC),
    )
    if not ids:
        print("No matching Phase 6 workers")
        return 0
    if not args.yes:
        print("Would terminate: " + " ".join(ids))
        return 0
    ec2.terminate_instances(InstanceIds=ids)
    print("Terminated: " + " ".join(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
