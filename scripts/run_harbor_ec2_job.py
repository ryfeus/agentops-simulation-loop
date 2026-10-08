"""Run one generated JobConfig with Harbor's native EC2 environment patched for HTTPS apt."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from harbor.environments.factory import _ENVIRONMENT_REGISTRY, _EnvEntry
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig


def install_https_ec2_environment() -> None:
    """Keep Harbor 0.22's EC2 lifecycle while replacing only its bootstrap class."""

    _ENVIRONMENT_REGISTRY[EnvironmentType.EC2] = _EnvEntry(
        "agentops_demo.harbor.ec2", "HttpsBootstrapEC2Environment", "ec2"
    )


async def run_config(config_path: Path) -> Path:
    from harbor.job import Job

    install_https_ec2_environment()
    config = JobConfig.model_validate_json(config_path.read_text())
    job = await Job.create(config)
    await job.run()
    return job.job_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    job_dir = asyncio.run(run_config(args.config))
    print(f"Results written to {job_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
