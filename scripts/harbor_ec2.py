"""Shared safe parsing for the opt-in Phase 6 Harbor EC2 controller tools."""

from __future__ import annotations

import ipaddress
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.aws_context import expected_account_id as expected_account_id

EXPECTED_REGION = "us-west-2"
EXPECTED_ACCOUNT_ENV = "EXPECTED_AWS_ACCOUNT_ID"
DEFAULT_ROOT = Path(".harbor-ec2")
ALLOWED_INSTANCE_TYPES = {"m7i-flex.large", "m7i.large", "t3.large"}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_controller_cidr(value: str) -> str:
    try:
        network = ipaddress.ip_network(value, strict=False)
    except ValueError as exc:
        raise ValueError("HARBOR_EC2_CONTROLLER_CIDR must be a valid CIDR") from exc
    if network.version != 4:
        raise ValueError("HARBOR_EC2_CONTROLLER_CIDR must be an IPv4 CIDR")
    if network.prefixlen == 0:
        raise ValueError("HARBOR_EC2_CONTROLLER_CIDR must not allow every address")
    return str(network)


def read_ssh_public_key(private_key: Path) -> str:
    if not private_key.is_file():
        raise ValueError("HARBOR_EC2_SSH_PRIVATE_KEY must name an existing private key")
    public_key = private_key.with_name(private_key.name + ".pub")
    try:
        value = public_key.read_text().strip()
    except OSError as exc:
        raise ValueError(f"SSH public key is unavailable: {public_key}") from exc
    fields = value.split()
    if len(fields) < 2 or not fields[0].startswith("ssh-") or not fields[1]:
        raise ValueError("SSH public key must be an OpenSSH public key")
    return value


def environment_int(name: str, *, default: int, environ: Mapping[str, str] | None = None) -> int:
    raw = (environ or os.environ).get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer in 1..4") from exc
    if not 1 <= value <= 4:
        raise ValueError(f"{name} must be an integer in 1..4")
    return value


def environment_instance_type(environ: Mapping[str, str] | None = None) -> str:
    value = (environ or os.environ).get("HARBOR_EC2_INSTANCE_TYPE", "m7i-flex.large")
    if value not in ALLOWED_INSTANCE_TYPES:
        allowed = ", ".join(sorted(ALLOWED_INSTANCE_TYPES))
        raise ValueError(f"HARBOR_EC2_INSTANCE_TYPE must be one of {allowed}")
    return value


def validate_run_id(value: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise ValueError("HARBOR_EC2_RUN_ID must be 1..128 safe filename characters")
    return value


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def terraform_output_value(outputs: Mapping[str, Any], name: str) -> str:
    entry = outputs.get(name)
    value = entry.get("value") if isinstance(entry, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Terraform output {name!r} is missing or invalid")
    return value


def phase_tags(run_id: str) -> dict[str, str]:
    return {"Project": "agentops-demo", "Phase": "6", "RunId": run_id, "ManagedBy": "harbor"}
