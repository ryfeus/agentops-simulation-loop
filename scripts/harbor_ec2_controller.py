"""AWS controller primitives for Phase 6; workers receive no AWS credentials."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from scripts.aws_context import require_account as require_account
from scripts.harbor_ec2 import EXPECTED_REGION, phase_tags


def worker_filters(run_id: str | None = None) -> list[dict[str, list[str] | str]]:
    filters: list[dict[str, list[str] | str]] = [
        {"Name": "tag:Project", "Values": ["agentops-demo"]},
        {"Name": "tag:Phase", "Values": ["6"]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
    ]
    if run_id is not None:
        filters.append({"Name": "tag:RunId", "Values": [run_id]})
    return filters


def discover_workers(ec2_client: Any, run_id: str | None = None) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    paginator = ec2_client.get_paginator("describe_instances")
    for page in paginator.paginate(Filters=worker_filters(run_id)):
        for reservation in page.get("Reservations", []):
            instances.extend(reservation.get("Instances", []))
    return instances


def tagged_for_run(instance: Mapping[str, Any], run_id: str) -> bool:
    tags = {item.get("Key"): item.get("Value") for item in instance.get("Tags", [])}
    required = phase_tags(run_id)
    return all(tags.get(key) == value for key, value in required.items())


def required_output_resources(outputs: Mapping[str, Any]) -> dict[str, str]:
    """Fail closed before DryRun if foundation outputs are incomplete."""

    required = ("harbor_subnet_id", "harbor_security_group_id", "harbor_key_name", "harbor_ami_id")
    values: dict[str, str] = {}
    for name in required:
        entry = outputs.get(name)
        value = entry.get("value") if isinstance(entry, Mapping) else None
        if not isinstance(value, str) or not value:
            raise ValueError(f"Terraform output {name!r} is missing or invalid")
        values[name] = value
    region = outputs.get("aws_region")
    region_value = region.get("value") if isinstance(region, Mapping) else None
    if region_value != EXPECTED_REGION:
        raise ValueError(f"Terraform output aws_region must be {EXPECTED_REGION}")
    return values


def route_has_igw(routes: Iterable[Mapping[str, Any]]) -> bool:
    return any(
        route.get("GatewayId", "").startswith("igw-")
        and route.get("DestinationCidrBlock") == "0.0.0.0/0"
        for route in routes
    )
