from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from conftest import GOLDEN_SCENARIO

from agentops_demo.contracts.scenario import Scenario
from agentops_demo.validation.scenario import (
    dump_scenario_json,
    dump_scenario_yaml,
    load_scenario,
    main,
    parse_scenario_yaml,
    validate_scenario,
)


def test_golden_scenario_captures_failure_and_expected_outcome() -> None:
    scenario = load_scenario(GOLDEN_SCENARIO)
    invoices = {invoice.id: invoice for invoice in scenario.initial_state.invoices}

    assert scenario.provenance.source.kind == "synthetic"
    assert scenario.provenance.source.trace_ref is None
    assert scenario.provenance.originating_agent_config is not None
    assert invoices["inv-123"].status == "disputed"
    assert invoices["inv-123"].amount == Decimal("89.00")
    assert scenario.initial_state.refunds == []
    assert scenario.initial_state.escalations == []
    assert any(
        call.tool == "refund_invoice" and call.arguments.invoice_id == "inv-123"
        for call in scenario.observed_failure.tool_calls
    )
    invariants = {
        (invariant.type, invariant.invoice_id) for invariant in scenario.expected_invariants
    }
    assert ("no_refund", "inv-123") in invariants
    assert ("must_escalate", "inv-123") in invariants


def test_yaml_round_trip_preserves_semantic_equality() -> None:
    scenario = load_scenario(GOLDEN_SCENARIO)
    serialized = dump_scenario_yaml(scenario)
    loaded_payload = yaml.safe_load(serialized)

    assert loaded_payload["initial_state"]["invoices"][0]["amount"] == "89.00"
    assert parse_scenario_yaml(serialized) == scenario


def test_json_round_trip_preserves_semantic_equality() -> None:
    scenario = load_scenario(GOLDEN_SCENARIO)
    serialized = dump_scenario_json(scenario)

    assert Scenario.model_validate(json.loads(serialized)) == scenario


def test_curated_validator_rejects_golden_scenario_with_initial_refund() -> None:
    scenario = load_scenario(GOLDEN_SCENARIO)
    invalid = scenario.model_copy(
        update={
            "initial_state": scenario.initial_state.model_copy(
                update={
                    "refunds": [
                        {
                            "id": "refund-1",
                            "invoice_id": "inv-123",
                            "reason": "already refunded",
                        }
                    ]
                }
            )
        }
    )

    with pytest.raises(ValueError, match="zero refunds"):
        validate_scenario(invalid)


def test_cli_validates_scenario_directory(capsys: pytest.CaptureFixture[str]) -> None:
    result = main([str(GOLDEN_SCENARIO.parent.parent)])

    captured = capsys.readouterr()
    assert result == 0
    assert f"OK {GOLDEN_SCENARIO}" in captured.out


def test_cli_reports_invalid_scenario(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    invalid = tmp_path / "scenario.yaml"
    invalid.write_text("schema_version: '2'\n", encoding="utf-8")

    result = main([str(invalid)])

    captured = capsys.readouterr()
    assert result == 1
    assert f"INVALID {invalid}" in captured.err


def test_cli_rejects_duplicate_scenario_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scenario = load_scenario(GOLDEN_SCENARIO)
    first = tmp_path / "first" / "scenario.yaml"
    second = tmp_path / "second" / "scenario.yaml"
    first.parent.mkdir()
    second.parent.mkdir()
    content = dump_scenario_yaml(scenario)
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")

    result = main([str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 1
    assert f"INVALID {second}" in captured.err
    assert "duplicate scenario id 'disputed-refund'" in captured.err
