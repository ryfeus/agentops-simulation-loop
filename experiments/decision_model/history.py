"""Opt-in observation contract for live, target-scoped escalation history.

This policy is frozen before the new challenge cohort is evaluated. It changes
only duplicate-escalation handling and uncertainty about missing history;
the original request parser remains a deliberate limitation.
"""

from __future__ import annotations

from typing import Any

from experiments.decision_model import data

VARIANT = "constrained-hybrid-escalation-history-v2"
SERIALIZATION_VERSION = "billing-observed-escalations-v2"


def make_state(request: str, target_invoice: str, observation: Any, prior_actions: list) -> dict:
    state = data.make_state(request, target_invoice, observation, prior_actions)
    state["policy"] = data.POLICY.replace(
        "get_invoice does not report prior escalation records.",
        "get_invoice reports target-scoped escalation_history. Only complete history "
        "with matching invoice identity establishes presence or absence of prior "
        "escalations. Existing escalation forbids another escalation, but does not "
        "by itself forbid a first requested refund of a paid invoice. Unavailable "
        "history is unknown, not empty. Refund history is not exposed.",
    )
    return state


def escalation_records(state: dict) -> list[dict] | None:
    """Return None for missing, incomplete, malformed or cross-target history."""
    obs = state.get("observation") or {}
    history = obs.get("escalation_history")
    target = state["target_invoice"]
    if not isinstance(history, dict):
        return None
    records = history.get("records")
    if (
        history.get("complete") is not True
        or history.get("invoice_id") != target
        or not isinstance(records, list)
        or any(
            not isinstance(row, dict)
            or row.get("invoice_id") != target
            or not isinstance(row.get("id"), str)
            or not row["id"]
            for row in records
        )
    ):
        return None
    return records


def rule_action(state: dict) -> str:
    baseline = data.rule_action(state)
    obs = state.get("observation") or {}
    if obs.get("ok") is not True or obs.get("invoice", {}).get("id") != state["target_invoice"]:
        return baseline
    status = obs["invoice"].get("status")
    records = escalation_records(state)
    if baseline == "no_action":
        return baseline
    if records is None:
        # Paid refunds do not depend on escalation history; escalation does.
        return "defer" if status == "disputed" or baseline == "escalate" else baseline
    if status == "disputed" and records:
        return "no_action"
    return baseline


def prohibited_actions(state: dict) -> list[str]:
    prohibited = set(data.prohibited_actions(state))
    if escalation_records(state):
        prohibited.add("escalate")
    return sorted(prohibited)
