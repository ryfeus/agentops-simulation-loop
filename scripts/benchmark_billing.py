"""Render and run the deterministic billing benchmark catalog through Harbor."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentops_demo.benchmark.catalog import (
    DEFAULT_CATALOG,
    CatalogRenderError,
    render_catalog_sync,
)
from agentops_demo.harbor.provenance import current_revision
from agentops_demo.taskify.integrity import (
    harbor_suite_sha256,
    harbor_task_sha256,
    scenario_sha256,
)
from scripts.generate_harbor_configs import write_configs


def _read_reward(job: Path) -> float | None:
    paths = sorted(job.rglob("reward.json"))
    if not paths:
        return None
    try:
        value = json.loads(paths[-1].read_text())
        return float(value["reward"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _slice(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        reward = row.get("reward")
        if isinstance(reward, (int, float)):
            grouped[str(row[key])].append(float(reward))
    return {name: sum(values) / len(values) for name, values in sorted(grouped.items())}


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _render_manifest(entries: list[Any], rendered: Path, source_revision: str) -> dict[str, Any]:
    task_identities: list[tuple[str, str, str]] = []
    tasks: dict[str, dict[str, str]] = {}
    for entry in sorted(entries, key=lambda item: item.scenario.id):
        scenario = entry.scenario
        scenario_digest = scenario_sha256(scenario)
        task_digest = harbor_task_sha256(rendered / scenario.id)
        task_identities.append((scenario.id, scenario_digest, task_digest))
        tasks[scenario.id] = {
            "archetype": scenario.benchmark.archetype,
            "difficulty": scenario.benchmark.difficulty,
            "scenario_sha256": scenario_digest,
            "harbor_task_sha256": task_digest,
        }
    return {
        "schema_version": "1",
        "benchmark": "billing",
        "source_revision": source_revision,
        "task_count": len(entries),
        "task_ids": sorted(tasks),
        "suite_sha256": harbor_suite_sha256(task_identities),
        "tasks": tasks,
    }


def run(
    *,
    catalog: Path,
    output: Path,
    candidates: list[str],
    execute: bool,
) -> dict[str, Any]:
    rendered = output / "tasks"
    source_revision = current_revision()
    entries = render_catalog_sync(catalog, rendered)
    manifest = _render_manifest(entries, rendered, source_revision)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "manifest.json", manifest)
    if not execute:
        result = {
            "status": "RENDERED",
            "catalog_count": len(entries),
            "rendered_count": len(entries),
            "failed_count": 0,
            "output": str(output),
            "manifest": manifest,
        }
        _write_json(output / "summary.json", result)
        return result

    wheel: Path | None = None
    if any(candidate != "oracle" for candidate in candidates):
        from scripts.build_harbor_agent import build, load_manifest, wheel_from_manifest

        package_manifest = build(output / "package", output / "configs")
        package = load_manifest(package_manifest)
        source_revision = package.get("source_revision")
        if not isinstance(source_revision, str):
            raise RuntimeError("Harbor package manifest has no source revision")
        wheel = wheel_from_manifest(package_manifest)
        write_configs(output / "configs", source_revision)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for entry in entries:
            job = output / "jobs" / candidate / entry.scenario.id
            command = [
                "uv",
                "run",
                "--group",
                "harbor",
                "harbor",
                "run",
                "-p",
                str(rendered / entry.scenario.id),
                "-a",
                "oracle"
                if candidate == "oracle"
                else "agentops_demo.harbor.agent:LangGraphBillingAgent",
                "-m",
                "oracle" if candidate == "oracle" else f"scripted/{candidate}",
                "--env",
                "docker",
                "--n-attempts",
                "1",
                "--n-concurrent",
                "1",
                "--yes",
                "--job-name",
                entry.scenario.id,
                "--jobs-dir",
                str(job.parent),
            ]
            if candidate != "oracle":
                if wheel is None:
                    raise RuntimeError("Harbor candidate execution package is missing")
                command.extend(
                    ("--ak", f"agent_config_path={output / 'configs' / (candidate + '.json')}")
                )
                command.extend(("--ak", f"package_path={wheel}"))
            completed = subprocess.run(command, check=False)
            rows.append(
                {
                    "task_id": entry.scenario.id,
                    "candidate": candidate,
                    "archetype": entry.scenario.benchmark.archetype
                    if entry.scenario.benchmark
                    else "",
                    "difficulty": entry.scenario.benchmark.difficulty
                    if entry.scenario.benchmark
                    else "",
                    "returncode": completed.returncode,
                    "reward": _read_reward(job),
                }
            )
    oracle_only = candidates == ["oracle"]
    oracle_failures = [row for row in rows if row["returncode"] != 0 or row["reward"] != 1.0]
    result = {
        "status": "FAILED" if oracle_only and oracle_failures else "COMPLETED",
        "overall": sum(float(row["reward"] or 0.0) for row in rows) / len(rows),
        "by_archetype": _slice(rows, "archetype"),
        "by_difficulty": _slice(rows, "difficulty"),
        "by_candidate": _slice(rows, "candidate"),
    }
    if oracle_only:
        result["oracle_failures"] = oracle_failures
    _write_json(output / "results.json", rows)
    _write_json(output / "summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate", choices=("correct", "bad", "noop"), action="append")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args(argv)
    run_id = time.strftime("%Y%m%d%H%M%S")
    output = args.output or Path(".artifacts/benchmarks/billing") / run_id
    if args.render_only and args.oracle:
        parser.error("--render-only and --oracle cannot be combined")
    if args.oracle and args.candidate:
        parser.error("--oracle does not accept --candidate")
    try:
        result = run(
            catalog=args.catalog,
            output=output,
            candidates=["oracle"] if args.oracle else args.candidate or ["correct", "bad"],
            execute=not args.render_only,
        )
    except CatalogRenderError as exc:
        output.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "FAILED",
            "catalog_count": exc.catalog_count,
            "rendered_count": exc.rendered_count,
            "failed_count": len(exc.failures),
            "failures": list(exc.failures),
            "output": str(output),
        }
        _write_json(output / "summary.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
