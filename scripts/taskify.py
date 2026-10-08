"""Create and render deterministic Harbor regression candidates from failure evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from agentops_demo.evaluation.trace_reader import parse_json_record, span_trace_id
from agentops_demo.taskify.contracts import FailureEvidence
from agentops_demo.taskify.harbor_renderer import render_harbor_task
from agentops_demo.taskify.harbor_runner import reproduce
from agentops_demo.taskify.integrity import scenario_sha256
from agentops_demo.taskify.scenario_builder import build_scenario
from agentops_demo.taskify.trace import trace_sha256
from agentops_demo.validation.scenario import dump_scenario_yaml, load_scenario, validate_scenario

DEFAULT_OUTPUT = Path(".taskify")


def _json_sha256(value: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def fetch_trace(*, log_group: str, trace_id: str, client: Any | None = None) -> dict[str, Any]:
    """Fetch only spans belonging to one exact AgentCore trace ID."""

    if client is None:
        from scripts.aws_context import verified_session

        client = verified_session().client("logs", region_name="us-west-2")
    kwargs: dict[str, Any] = {
        "logGroupName": log_group,
        "logStreamNames": ["spans"],
        "filterPattern": trace_id,
    }
    spans: list[dict[str, Any]] = []
    while True:
        response = client.filter_log_events(**kwargs)
        for event in response.get("events", []):
            try:
                span = parse_json_record(event["message"])
            except (KeyError, ValueError):
                continue
            if span_trace_id(span) == trace_id:
                spans.append(span)
        token = response.get("nextToken")
        if not isinstance(token, str) or not token:
            break
        kwargs["nextToken"] = token
    if not spans:
        raise RuntimeError(f"exact trace {trace_id!r} was not found")
    return {"evaluationInput": {"sessionSpans": spans}}


def fetch_exact_trace(evidence: FailureEvidence) -> dict[str, Any]:
    """Fetch only spans belonging to the exact trace named by failure evidence."""

    return fetch_trace(
        log_group=evidence.source.trace_log_group,
        trace_id=evidence.source.trace_id,
    )


def create(*, failure_path: Path, trace_path: Path, output_root: Path = DEFAULT_OUTPUT) -> Path:
    evidence_raw = _load_json(failure_path)
    trace = _load_json(trace_path)
    evidence = FailureEvidence.model_validate(evidence_raw)
    scenario = build_scenario(evidence, trace)
    destination = output_root / scenario.id
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(failure_path, destination / "failure.json")
    (destination / "trace.json").write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    (destination / "scenario.yaml").write_text(dump_scenario_yaml(scenario))
    config = scenario.provenance.originating_agent_config
    assert config is not None
    candidate_dir = destination / "candidate"
    candidate_dir.mkdir(exist_ok=True)
    (candidate_dir / "originating-agent-config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema_version": "1",
        "scenario_id": scenario.id,
        "source": {
            "failure_sha256": _json_sha256(evidence_raw),
            "trace_sha256": trace_sha256(trace),
            "session_id": evidence.source.session_id,
            "trace_id": evidence.source.trace_id,
        },
        "candidate": {
            "source_revision": evidence.candidate.source_revision,
            "agent_config_fingerprint": evidence.candidate.agent_config_fingerprint,
        },
        "scenario": {
            "path": "scenario.yaml",
            "sha256": scenario_sha256(scenario),
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create_parser = commands.add_parser("create")
    create_parser.add_argument("--failure", required=True, type=Path)
    create_parser.add_argument("--trace", type=Path)
    create_parser.add_argument("--offline", action="store_true")
    create_parser.add_argument("--output-root", default=DEFAULT_OUTPUT, type=Path)
    render_parser = commands.add_parser("render")
    render_parser.add_argument("--scenario", required=True, type=Path)
    render_parser.add_argument("--output", type=Path)
    validate_parser = commands.add_parser("validate-scenario")
    validate_parser.add_argument("--scenario", required=True, type=Path)
    benchmark_parser = commands.add_parser("validate-benchmark")
    benchmark_parser.add_argument("--scenario", required=True, type=Path)
    reproduce_parser = commands.add_parser("reproduce")
    reproduce_parser.add_argument("--scenario", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "create":
        if args.offline and args.trace is None:
            parser.error("--offline requires --trace")
        if args.trace is None:
            evidence = FailureEvidence.model_validate(_load_json(args.failure))
            temporary = args.output_root / ".live-trace.json"
            temporary.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(fetch_exact_trace(evidence), indent=2, sort_keys=True) + "\n"
            )
            args.trace = temporary
        destination = create(
            failure_path=args.failure, trace_path=args.trace, output_root=args.output_root
        )
        print(destination)
    elif args.command == "render":
        scenario = load_scenario(args.scenario)
        output = args.output or args.scenario.parent / "harbor"
        asyncio.run(render_harbor_task(scenario, output))
        print(output)
    elif args.command == "validate-scenario":
        validate_scenario(load_scenario(args.scenario))
        print(f"OK {args.scenario}")
    else:
        report = reproduce(args.scenario)
        print(report.status)
        if report.status != "VALIDATED":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
