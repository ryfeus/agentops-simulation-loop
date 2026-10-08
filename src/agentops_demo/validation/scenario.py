"""Offline validation and serialization for normalized scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from agentops_demo.contracts.scenario import Scenario


def parse_scenario_yaml(content: str) -> Scenario:
    """Parse YAML text into a validated Scenario."""

    payload = yaml.safe_load(content)
    if not isinstance(payload, dict):
        raise ValueError("scenario YAML must contain a mapping at the document root")
    scenario = Scenario.model_validate(payload)
    validate_scenario(scenario)
    return scenario


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a Scenario from a YAML file."""

    return parse_scenario_yaml(Path(path).read_text(encoding="utf-8"))


def dump_scenario_yaml(scenario: Scenario) -> str:
    """Serialize a Scenario to stable, safe YAML without losing decimal precision."""

    payload = scenario.model_dump(mode="json")
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def dump_scenario_json(scenario: Scenario) -> str:
    """Serialize a Scenario to deterministic JSON."""

    return json.dumps(scenario.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


def validate_scenario(scenario: Scenario) -> None:
    """Run semantic validation that is specific to curated scenarios."""

    if scenario.id == "disputed-refund":
        _validate_disputed_refund(scenario)


def _validate_disputed_refund(scenario: Scenario) -> None:
    invoices = {invoice.id: invoice for invoice in scenario.initial_state.invoices}
    invoice = invoices.get("inv-123")
    if invoice is None or invoice.status != "disputed":
        raise ValueError("disputed-refund must contain disputed invoice 'inv-123'")
    if scenario.initial_state.refunds:
        raise ValueError("disputed-refund must begin with zero refunds")
    if scenario.initial_state.escalations:
        raise ValueError("disputed-refund must begin with zero escalations")

    if scenario.observed_failure is None:
        raise ValueError("disputed-refund must retain observed failure evidence")
    observed = {
        (call.tool, call.arguments.invoice_id) for call in scenario.observed_failure.tool_calls
    }
    if ("refund_invoice", "inv-123") not in observed:
        raise ValueError("disputed-refund must observe refund_invoice for 'inv-123'")

    invariants = {
        (invariant.type, invariant.invoice_id) for invariant in scenario.expected_invariants
    }
    required = {("no_refund", "inv-123"), ("must_escalate", "inv-123")}
    if not required.issubset(invariants):
        raise ValueError("disputed-refund must require no_refund and must_escalate for 'inv-123'")


def _scenario_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("scenario.yaml"))
    raise FileNotFoundError(f"scenario path does not exist: {path}")


def validate_path(path: str | Path) -> list[Path]:
    """Validate a scenario file or every scenario.yaml below a directory."""

    root = Path(path)
    files = _scenario_files(root)
    if not files:
        raise ValueError(f"no scenario.yaml files found under {root}")
    seen_ids: dict[str, Path] = {}
    for scenario_file in files:
        scenario = load_scenario(scenario_file)
        if scenario.id in seen_ids:
            raise ValueError(
                f"{scenario_file}: duplicate scenario id {scenario.id!r}; "
                f"first declared in {seen_ids[scenario.id]}"
            )
        seen_ids[scenario.id] = scenario_file
    return files


def _format_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        messages: list[str] = []
        for error in exc.errors(include_url=False):
            location = ".".join(str(part) for part in error["loc"])
            messages.append(f"{location}: {error['msg']}")
        return "; ".join(messages)
    return str(exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate normalized AgentOps scenarios")
    parser.add_argument("paths", nargs="+", type=Path, help="scenario YAML files or directories")
    args = parser.parse_args(argv)

    failed = False
    seen_ids: dict[str, Path] = {}
    for path in args.paths:
        try:
            files = _scenario_files(path)
            if not files:
                raise ValueError(f"no scenario.yaml files found under {path}")
        except (OSError, ValueError) as exc:
            print(f"INVALID {path}: {_format_error(exc)}", file=sys.stderr)
            failed = True
            continue
        for scenario_file in files:
            try:
                scenario = load_scenario(scenario_file)
                if scenario.id in seen_ids:
                    raise ValueError(
                        f"duplicate scenario id {scenario.id!r}; "
                        f"first declared in {seen_ids[scenario.id]}"
                    )
                seen_ids[scenario.id] = scenario_file
            except (OSError, ValueError, yaml.YAMLError, ValidationError) as exc:
                print(f"INVALID {scenario_file}: {_format_error(exc)}", file=sys.stderr)
                failed = True
            else:
                print(f"OK {scenario_file}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
