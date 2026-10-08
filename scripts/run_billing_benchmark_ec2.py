"""Opt-in EC2 execution of the rendered static billing benchmark suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from agentops_demo.benchmark.catalog import DEFAULT_CATALOG, render_catalog_sync
from agentops_demo.harbor.provenance import build_clean_wheel
from agentops_demo.taskify.integrity import harbor_task_sha256
from scripts.generate_harbor_configs import write_configs
from scripts.generate_harbor_ec2_config import execution_config, job_config, load_terraform_outputs
from scripts.harbor_ec2 import DEFAULT_ROOT


def run(*, outputs: Path, catalog: Path, run_id: str, candidate: str) -> dict[str, object]:
    if candidate not in {"correct", "bad"}:
        raise ValueError("candidate must be correct or bad")
    root = DEFAULT_ROOT / "billing" / run_id
    entries = render_catalog_sync(catalog, root / "tasks")
    execution = execution_config(
        outputs=load_terraform_outputs(outputs), run_id=run_id, task_source="static"
    )
    wheel, package = build_clean_wheel(root / "package")
    configs = write_configs(root / "configs", package.source_revision)
    private_key = Path(os.environ["HARBOR_EC2_SSH_PRIVATE_KEY"])
    rows: list[dict[str, object]] = []
    for entry in entries:
        task = root / "tasks" / entry.scenario.id
        job_root = root / "jobs"
        config_path = root / "job-configs" / f"{entry.scenario.id}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = job_config(
            execution=execution,
            task=task,
            jobs_dir=job_root,
            job_name=entry.scenario.id,
            model=f"scripted/{candidate}",
            agent_config_path=root / "configs" / f"{candidate}.json",
            package_path=wheel,
            ssh_key_path=private_key,
        )
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        completed = subprocess.run(
            (
                "uv",
                "run",
                "--group",
                "harbor",
                "python",
                "-m",
                "scripts.run_harbor_ec2_job",
                "--config",
                str(config_path),
            ),
            check=False,
        )
        rows.append(
            {
                "task_id": entry.scenario.id,
                "archetype": entry.scenario.benchmark.archetype if entry.scenario.benchmark else "",
                "task_sha256": harbor_task_sha256(task),
                "returncode": completed.returncode,
            }
        )
    report = {
        "schema_version": "1",
        "run_id": run_id,
        "task_source": "static",
        "candidate": {
            "model": f"scripted/{candidate}",
            "agent_config_fingerprint": configs[candidate].fingerprint(),
            "package_sha256": package.sha256,
            "source_revision": package.source_revision,
        },
        "execution": execution.model_dump(mode="json"),
        "tasks": rows,
    }
    (root / "suite-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_ROOT / "terraform-outputs.json")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--run-id", default=f"billing-{time.strftime('%Y%m%d%H%M%S')}")
    parser.add_argument("--candidate", choices=("correct", "bad"), default="correct")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                outputs=args.outputs,
                catalog=args.catalog,
                run_id=args.run_id,
                candidate=args.candidate,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
