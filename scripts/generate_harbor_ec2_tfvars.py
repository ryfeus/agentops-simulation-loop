"""Generate public-key-only Terraform inputs for the optional Harbor EC2 foundation."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

from scripts.harbor_ec2 import DEFAULT_ROOT, read_ssh_public_key, validate_controller_cidr

DEFAULT_OUTPUT = DEFAULT_ROOT / "foundation.auto.tfvars.json"


def payload(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    env = environ or os.environ
    private_key = env.get("HARBOR_EC2_SSH_PRIVATE_KEY", "")
    if not private_key:
        raise ValueError("HARBOR_EC2_SSH_PRIVATE_KEY must be set")
    cidr = env.get("HARBOR_EC2_CONTROLLER_CIDR", "")
    if not cidr:
        raise ValueError("HARBOR_EC2_CONTROLLER_CIDR must be set")
    return {
        "harbor_controller_cidr": validate_controller_cidr(cidr),
        "harbor_ec2_enabled": True,
        "harbor_ssh_public_key": read_ssh_public_key(Path(private_key).expanduser()),
    }


def write_tfvars(output: Path = DEFAULT_OUTPUT, environ: Mapping[str, str] | None = None) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload(environ), indent=2, sort_keys=True) + "\n")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(f"Wrote {write_tfvars(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
