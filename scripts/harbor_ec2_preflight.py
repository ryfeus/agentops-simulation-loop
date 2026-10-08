"""Validate the Phase 6 EC2 foundation and DryRun worker launch permission."""

from __future__ import annotations

import argparse
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from scripts.generate_harbor_ec2_config import execution_config, load_terraform_outputs
from scripts.harbor_ec2 import EXPECTED_REGION, expected_account_id, read_ssh_public_key
from scripts.harbor_ec2_controller import require_account, required_output_resources, route_has_igw


def preflight(
    *,
    sts_client: Any,
    ec2_client: Any,
    outputs: dict[str, Any],
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    env = environ or dict(os.environ)
    require_account(sts_client, expected_account_id(env))
    harbor_version = version("harbor")
    if not harbor_version.startswith("0.22."):
        raise RuntimeError(f"Harbor EC2 preflight requires Harbor 0.22.x, found {harbor_version}")
    resources = required_output_resources(outputs)
    private_key = env.get("HARBOR_EC2_SSH_PRIVATE_KEY", "")
    if not private_key:
        raise ValueError("HARBOR_EC2_SSH_PRIVATE_KEY must be set")
    read_ssh_public_key(Path(private_key).expanduser())
    execution = execution_config(
        outputs=outputs,
        run_id="preflight",
        task_source="static",
        environ=env,
    )
    if execution.region != EXPECTED_REGION:
        raise ValueError("EC2 execution region is invalid")
    subnets = ec2_client.describe_subnets(SubnetIds=[resources["harbor_subnet_id"]]).get(
        "Subnets", []
    )
    if len(subnets) != 1 or not subnets[0].get("MapPublicIpOnLaunch"):
        raise RuntimeError("Harbor subnet must assign public IPv4 addresses")
    route_tables = ec2_client.describe_route_tables(
        Filters=[{"Name": "association.subnet-id", "Values": [resources["harbor_subnet_id"]]}]
    ).get("RouteTables", [])
    if not any(route_has_igw(table.get("Routes", [])) for table in route_tables):
        raise RuntimeError("Harbor subnet lacks a default route through an internet gateway")
    ec2_client.describe_security_groups(GroupIds=[resources["harbor_security_group_id"]])
    try:
        ec2_client.run_instances(
            DryRun=True,
            ImageId=resources["harbor_ami_id"],
            InstanceType=execution.instance_type,
            MinCount=1,
            MaxCount=1,
            SubnetId=resources["harbor_subnet_id"],
            SecurityGroupIds=[resources["harbor_security_group_id"]],
            KeyName=resources["harbor_key_name"],
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code != "DryRunOperation":
            raise RuntimeError(f"EC2 RunInstances DryRun failed: {code or exc}") from exc
    else:
        raise RuntimeError("EC2 RunInstances DryRun unexpectedly launched or returned success")
    return resources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True)
    args = parser.parse_args(argv)
    session = boto3.Session(
        profile_name=os.getenv("AWS_PROFILE", "default"), region_name=EXPECTED_REGION
    )
    resources = preflight(
        sts_client=session.client("sts"),
        ec2_client=session.client("ec2"),
        outputs=load_terraform_outputs(args.outputs),
    )
    print("PASS " + " ".join(f"{key}={value}" for key, value in sorted(resources.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
