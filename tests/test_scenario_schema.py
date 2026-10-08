from __future__ import annotations

import copy

import pytest
import yaml
from conftest import GOLDEN_SCENARIO
from pydantic import ValidationError

from agentops_demo.contracts.scenario import Scenario
from agentops_demo.validation.scenario import load_scenario


def golden_data() -> dict[str, object]:
    payload = yaml.safe_load(GOLDEN_SCENARIO.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_golden_scenario_loads() -> None:
    assert load_scenario(GOLDEN_SCENARIO).id == "disputed-refund"


def test_provenance_parses_and_originating_config_fingerprints() -> None:
    scenario = load_scenario(GOLDEN_SCENARIO)

    assert scenario.provenance.source.kind == "synthetic"
    assert scenario.provenance.source.evaluation is not None
    assert scenario.provenance.source.evaluation.evaluator == "DisputePolicyCompliance"
    assert scenario.provenance.originating_agent_config is not None
    assert len(scenario.provenance.originating_agent_config.fingerprint()) == 64


def test_originating_agent_config_is_optional() -> None:
    data = golden_data()
    del data["provenance"]["originating_agent_config"]  # type: ignore[index]

    scenario = Scenario.model_validate(data)

    assert scenario.provenance.originating_agent_config is None


def test_scenario_contract_has_no_agent_candidate_field() -> None:
    assert "agent_config" not in Scenario.model_fields
    assert "candidate_agent_config" not in Scenario.model_fields


def test_synthetic_source_without_trace_reference_is_accepted() -> None:
    data = golden_data()
    del data["provenance"]["source"]["trace_ref"]  # type: ignore[index]

    scenario = Scenario.model_validate(data)

    assert scenario.provenance.source.trace_ref is None


def test_synthetic_source_with_trace_reference_is_rejected() -> None:
    data = golden_data()
    data["provenance"]["source"]["trace_ref"] = "trace-123"  # type: ignore[index]

    with pytest.raises(ValidationError, match=r"synthetic .* cannot contain"):
        Scenario.model_validate(data)


def test_trace_source_with_trace_reference_is_accepted() -> None:
    data = golden_data()
    data["provenance"]["source"] = {  # type: ignore[index]
        "kind": "trace",
        "trace_ref": "trace-123",
    }

    scenario = Scenario.model_validate(data)

    assert scenario.provenance.source.kind == "trace"
    assert scenario.provenance.source.trace_ref == "trace-123"
    assert scenario.provenance.source.evaluation is None


@pytest.mark.parametrize("trace_ref", [None, "", "   "])
def test_trace_source_without_nonblank_reference_is_rejected(trace_ref: str | None) -> None:
    data = golden_data()
    data["provenance"]["source"] = {  # type: ignore[index]
        "kind": "trace",
        "trace_ref": trace_ref,
    }

    with pytest.raises(ValidationError, match="require a non-empty trace reference"):
        Scenario.model_validate(data)


def test_evaluation_evidence_is_optional() -> None:
    data = golden_data()
    del data["provenance"]["source"]["evaluation"]  # type: ignore[index]

    scenario = Scenario.model_validate(data)

    assert scenario.provenance.source.evaluation is None


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_evaluation_score_outside_unit_interval_is_rejected(score: float) -> None:
    data = golden_data()
    data["provenance"]["source"]["evaluation"]["score"] = score  # type: ignore[index]

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_blank_evaluator_is_rejected() -> None:
    data = golden_data()
    data["provenance"]["source"]["evaluation"]["evaluator"] = " "  # type: ignore[index]

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


@pytest.mark.parametrize(
    "scenario_id", ["", "UPPERCASE", "../escape", "has space", "/root", "trailing-"]
)
def test_unsafe_scenario_id_is_rejected(scenario_id: str) -> None:
    data = golden_data()
    data["id"] = scenario_id

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_missing_instruction_is_rejected() -> None:
    data = golden_data()
    del data["instruction"]

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_blank_instruction_is_rejected() -> None:
    data = golden_data()
    data["instruction"] = "  "

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_unsupported_schema_version_is_rejected() -> None:
    data = golden_data()
    data["schema_version"] = "2"

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_duplicate_invoice_ids_are_rejected() -> None:
    data = golden_data()
    duplicate = copy.deepcopy(data["initial_state"]["invoices"][0])  # type: ignore[index]
    data["initial_state"]["invoices"].append(duplicate)  # type: ignore[index]

    with pytest.raises(ValidationError, match="duplicate invoice id"):
        Scenario.model_validate(data)


def test_duplicate_customer_ids_are_rejected() -> None:
    data = golden_data()
    duplicate = copy.deepcopy(data["initial_state"]["customers"][0])  # type: ignore[index]
    data["initial_state"]["customers"].append(duplicate)  # type: ignore[index]

    with pytest.raises(ValidationError, match="duplicate customer id"):
        Scenario.model_validate(data)


def test_duplicate_refund_ids_are_rejected() -> None:
    data = golden_data()
    refund = {"id": "refund-1", "invoice_id": "inv-123", "reason": "duplicate"}
    data["initial_state"]["refunds"] = [refund, copy.deepcopy(refund)]  # type: ignore[index]

    with pytest.raises(ValidationError, match="duplicate refund id"):
        Scenario.model_validate(data)


def test_duplicate_escalation_ids_are_rejected() -> None:
    data = golden_data()
    escalation = {"id": "esc-1", "invoice_id": "inv-123", "reason": "review"}
    data["initial_state"]["escalations"] = [escalation, copy.deepcopy(escalation)]  # type: ignore[index]

    with pytest.raises(ValidationError, match="duplicate escalation id"):
        Scenario.model_validate(data)


def test_customer_and_invoice_ids_can_match() -> None:
    data = golden_data()
    data["initial_state"]["customers"][0]["id"] = "inv-123"  # type: ignore[index]
    data["initial_state"]["invoices"][0]["customer_id"] = "inv-123"  # type: ignore[index]

    assert Scenario.model_validate(data).initial_state.invoices[0].customer_id == "inv-123"


def test_invoice_and_refund_ids_can_match() -> None:
    data = golden_data()
    data["initial_state"]["refunds"] = [  # type: ignore[index]
        {"id": "inv-123", "invoice_id": "inv-123", "reason": "prior adjustment"}
    ]

    assert Scenario.model_validate(data).initial_state.refunds[0].id == "inv-123"


def test_invoice_referencing_unknown_customer_is_rejected() -> None:
    data = golden_data()
    data["initial_state"]["invoices"][0]["customer_id"] = "missing"  # type: ignore[index]

    with pytest.raises(ValidationError, match="unknown customer"):
        Scenario.model_validate(data)


def test_invariant_referencing_unknown_invoice_is_rejected() -> None:
    data = golden_data()
    data["expected_invariants"][0]["invoice_id"] = "missing"  # type: ignore[index]

    with pytest.raises(ValidationError, match=r"invariant .* unknown invoice"):
        Scenario.model_validate(data)


def test_required_inspections_default_empty_and_allow_missing_invoice_targets() -> None:
    data = golden_data()
    data["benchmark"] = {
        "category": "billing",
        "archetype": "test",
        "difficulty": "easy",
        "trajectory": {},
    }
    scenario = Scenario.model_validate(data)
    assert scenario.benchmark is not None
    assert scenario.benchmark.trajectory.required_inspections == []

    data = golden_data()
    data["benchmark"] = {
        "category": "billing",
        "archetype": "test",
        "difficulty": "easy",
        "trajectory": {},
    }
    data["benchmark"]["trajectory"]["required_inspections"] = ["inv-999"]  # type: ignore[index]
    scenario = Scenario.model_validate(data)
    assert scenario.benchmark is not None
    assert scenario.benchmark.trajectory.required_inspections == ["inv-999"]


@pytest.mark.parametrize("required_inspections", [["inv-123", "inv-123"], ["  "]])
def test_required_inspections_must_be_unique_and_nonblank(required_inspections: list[str]) -> None:
    data = golden_data()
    data["benchmark"] = {
        "category": "billing",
        "archetype": "test",
        "difficulty": "easy",
        "trajectory": {},
    }
    data["benchmark"]["trajectory"]["required_inspections"] = required_inspections  # type: ignore[index]

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_tool_call_referencing_unknown_invoice_is_rejected() -> None:
    data = golden_data()
    data["observed_failure"]["tool_calls"][0]["arguments"]["invoice_id"] = "missing"  # type: ignore[index]

    with pytest.raises(ValidationError, match=r"tool call .* unknown invoice"):
        Scenario.model_validate(data)


def test_malformed_agent_config_is_rejected() -> None:
    data = golden_data()
    data["provenance"]["originating_agent_config"]["agent"]["source_revision"] = " "  # type: ignore[index]

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


@pytest.mark.parametrize("field", ["source", "agent_config", "candidate_agent_config"])
def test_legacy_or_candidate_top_level_configuration_is_rejected(field: str) -> None:
    data = golden_data()
    if field == "source":
        value = copy.deepcopy(data["provenance"]["source"])  # type: ignore[index]
    else:
        value = copy.deepcopy(data["provenance"]["originating_agent_config"])  # type: ignore[index]
    data[field] = value

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Scenario.model_validate(data)


def test_unknown_tool_call_is_rejected() -> None:
    data = golden_data()
    data["observed_failure"]["tool_calls"][0]["tool"] = "execute_sql"  # type: ignore[index]

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_unknown_invariant_is_rejected() -> None:
    data = golden_data()
    data["expected_invariants"][0]["type"] = "run_python"  # type: ignore[index]

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_observed_tool_result_is_intentionally_rejected() -> None:
    data = golden_data()
    data["observed_failure"]["tool_calls"][0]["result"] = {"status": "disputed"}  # type: ignore[index]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Scenario.model_validate(data)
