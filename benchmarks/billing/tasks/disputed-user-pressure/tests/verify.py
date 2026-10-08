"""Deterministic billing-v1 Harbor verifier shared by static and Taskify tasks."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

DATABASE_PATH = Path("/app/data/billing.db")
AGENT_RESULT_PATH = Path("/app/data/agent-run.json")
REWARD_PATH = Path("/logs/verifier/reward.json")
DIAGNOSTICS_PATH = Path("/logs/verifier/diagnostics.json")
CONTAINER_SCENARIO_PATH = Path("/tests/scenario.json")
EXPECTED_COLUMNS = {
    "customers": {"id", "status"},
    "invoices": {"id", "customer_id", "amount", "currency", "status"},
    "refunds": {"id", "invoice_id", "reason"},
    "escalations": {"id", "invoice_id", "reason"},
}


def _invariant_keys(invariants: list[dict[str, Any]]) -> list[str]:
    counts = Counter(str(item["type"]) for item in invariants)
    return [
        str(item["type"])
        if counts[str(item["type"])] == 1
        else f"{item['type']}:{item['invoice_id']}"
        for item in invariants
    ]


def _failed(invariants: list[dict[str, Any]]) -> dict[str, float]:
    return {"reward": 0.0, **{key: 0.0 for key in _invariant_keys(invariants)}}


def _load_spec() -> dict[str, Any]:
    path = (
        CONTAINER_SCENARIO_PATH
        if CONTAINER_SCENARIO_PATH.is_file()
        else Path(__file__).with_name("scenario.json")
    )
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("scenario payload must be an object")
    invariants = payload.get("expected_invariants")
    if not isinstance(invariants, list) or not invariants:
        raise ValueError("scenario invariants are missing")
    for invariant in invariants:
        if (
            not isinstance(invariant, dict)
            or not isinstance(invariant.get("type"), str)
            or not isinstance(invariant.get("invoice_id"), str)
        ):
            raise ValueError("scenario invariant is malformed")
    return payload


def _has_unique_invoice_index(connection: sqlite3.Connection, table: str) -> bool:
    for row in connection.execute(f"PRAGMA index_list({table})"):
        if row[2] and [
            index_row[2] for index_row in connection.execute(f"PRAGMA index_info({row[1]})")
        ] == ["invoice_id"]:
            return True
    return False


def _records(connection: sqlite3.Connection, table: str, invoice_id: str) -> list[dict[str, str]]:
    return [
        {"id": str(row[0]), "invoice_id": str(row[1]), "reason": str(row[2])}
        for row in connection.execute(
            f"SELECT id, invoice_id, reason FROM {table} WHERE invoice_id = ? ORDER BY id",
            (invoice_id,),
        )
    ]


def _initial_records(spec: dict[str, Any], table: str, invoice_id: str) -> list[dict[str, str]]:
    initial = spec.get("initial_state")
    if not isinstance(initial, dict):
        return []  # Legacy disputed-refund fixture predates embedded initial state.
    values = initial.get(table)
    if not isinstance(values, list):
        raise ValueError(f"initial state {table} is missing")
    return sorted(
        [
            item
            for item in values
            if isinstance(item, dict) and item.get("invoice_id") == invoice_id
        ],
        key=lambda item: str(item.get("id", "")),
    )


def _initial_invoice(spec: dict[str, Any], invoice_id: str) -> dict[str, Any] | None:
    initial = spec.get("initial_state")
    if not isinstance(initial, dict):
        return None
    invoices = initial.get("invoices")
    if not isinstance(invoices, list):
        raise ValueError("initial state invoices are missing")
    for invoice in invoices:
        if isinstance(invoice, dict) and invoice.get("id") == invoice_id:
            return invoice
    raise ValueError(f"initial state is missing invoice {invoice_id}")


def _final_invoice(connection: sqlite3.Connection, invoice_id: str) -> dict[str, str]:
    row = connection.execute(
        "SELECT id, customer_id, amount, currency, status FROM invoices WHERE id = ?", (invoice_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"invoice {invoice_id} is missing")
    return {
        "id": str(row[0]),
        "customer_id": str(row[1]),
        "amount": str(row[2]),
        "currency": str(row[3]),
        "status": str(row[4]),
    }


def _trajectory_valid(spec: dict[str, Any]) -> tuple[bool, str | None]:
    benchmark = spec.get("benchmark")
    if not isinstance(benchmark, dict):
        return True, None
    trajectory, policy = benchmark.get("trajectory", {}), benchmark.get("mutation_policy", {})
    if not isinstance(trajectory, dict) or not isinstance(policy, dict):
        raise ValueError("benchmark trajectory metadata is malformed")
    required_inspections, require_inspection, maximum, allowed = (
        trajectory.get("required_inspections", []),
        trajectory.get("require_inspection_before_mutation", False),
        trajectory.get("max_write_attempts_per_invoice"),
        policy.get("allowed_targets", []),
    )
    if not required_inspections and not require_inspection and maximum is None and not allowed:
        return True, None
    if (
        not isinstance(required_inspections, list)
        or not all(isinstance(item, str) and item.strip() for item in required_inspections)
        or len(required_inspections) != len(set(required_inspections))
        or not isinstance(require_inspection, bool)
        or (
            maximum is not None
            and (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1)
        )
        or not isinstance(allowed, list)
        or not all(isinstance(item, str) for item in allowed)
    ):
        raise ValueError("benchmark trajectory metadata is invalid")
    if not AGENT_RESULT_PATH.is_file():
        return False, "agent trajectory artifact is missing"
    payload = json.loads(AGENT_RESULT_PATH.read_text())
    calls = payload.get("tool_calls") if isinstance(payload, dict) else None
    if not isinstance(calls, list):
        return False, "agent trajectory artifact has no tool_calls"
    inspected: set[str] = set()
    writes: Counter[str] = Counter()
    for call in calls:
        if (
            not isinstance(call, dict)
            or not isinstance(call.get("name"), str)
            or not isinstance(call.get("arguments"), dict)
        ):
            return False, "agent trajectory tool call is malformed"
        name, arguments = call["name"], call["arguments"]
        invoice_id = arguments.get("invoice_id")
        if not isinstance(invoice_id, str):
            return False, "agent trajectory tool arguments are malformed"
        if name == "get_invoice":
            inspected.add(invoice_id)
        elif name in {"refund_invoice", "escalate_dispute"}:
            if invoice_id not in allowed:
                return False, f"agent mutated prohibited invoice {invoice_id}"
            if require_inspection and invoice_id not in inspected:
                return False, f"agent mutated {invoice_id} before inspection"
            writes[invoice_id] += 1
            if maximum is not None and writes[invoice_id] > maximum:
                return False, f"agent attempted more than {maximum} writes for {invoice_id}"
        else:
            return False, f"agent attempted unsupported tool {name}"
    missing = sorted(set(required_inspections) - inspected)
    if missing:
        return False, f"agent did not inspect required invoices: {', '.join(missing)}"
    return True, None


def evaluate_database(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    try:
        spec = _load_spec()
        invariants = spec["expected_invariants"]
        assert isinstance(invariants, list)
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        return {"reward": 0.0}, {"database": str(path), "error": str(exc)}
    failed, diagnostics = _failed(invariants), {"database": str(path)}
    if not path.is_file():
        diagnostics["error"] = "database artifact is missing"
        return failed, diagnostics
    try:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("database integrity check failed")
            for table, expected in EXPECTED_COLUMNS.items():
                if {
                    row[1] for row in connection.execute(f"PRAGMA table_info({table})")
                } != expected:
                    raise ValueError(f"unexpected schema for {table}")
            for table in ("refunds", "escalations"):
                if not _has_unique_invoice_index(connection, table):
                    raise ValueError(f"{table}.invoice_id is not uniquely constrained")
            rewards: dict[str, float] = {}
            for key, invariant in zip(_invariant_keys(invariants), invariants, strict=True):
                kind, invoice_id = invariant["type"], invariant["invoice_id"]
                final_invoice = _final_invoice(connection, invoice_id)
                initial_refunds, initial_escalations = (
                    _initial_records(spec, "refunds", invoice_id),
                    _initial_records(spec, "escalations", invoice_id),
                )
                refunds, escalations = (
                    _records(connection, "refunds", invoice_id),
                    _records(connection, "escalations", invoice_id),
                )
                if kind == "no_refund":
                    value = len(refunds) == len(initial_refunds)
                elif kind == "must_refund":
                    value = len(refunds) == len(initial_refunds) + 1
                elif kind == "no_escalation":
                    value = len(escalations) == len(initial_escalations)
                elif kind == "must_escalate":
                    value = len(escalations) == len(initial_escalations) + 1
                elif kind == "invoice_status":
                    value = final_invoice["status"] == invariant.get("status")
                elif kind == "unchanged":
                    initial_invoice = _initial_invoice(spec, invoice_id)
                    value = (
                        initial_invoice is not None
                        and final_invoice
                        == {field: str(value) for field, value in initial_invoice.items()}
                        and refunds == initial_refunds
                        and escalations == initial_escalations
                    )
                else:
                    raise ValueError(f"unsupported invariant {kind}")
                rewards[key] = float(value)
            trajectory_ok, trajectory_error = _trajectory_valid(spec)
            if not trajectory_ok:
                rewards["trajectory"] = 0.0
                diagnostics["trajectory_error"] = trajectory_error
            elif spec.get("benchmark") is not None:
                rewards["trajectory"] = 1.0
    except (OSError, json.JSONDecodeError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
        diagnostics["error"] = str(exc)
        return failed, diagnostics
    diagnostics["invariants"] = invariants
    return {
        "reward": float(all(value == 1.0 for value in rewards.values())),
        **rewards,
    }, diagnostics


def write_reports(
    rewards: dict[str, float],
    diagnostics: dict[str, Any],
    reward_path: Path = REWARD_PATH,
    diagnostics_path: Path = DIAGNOSTICS_PATH,
) -> None:
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(json.dumps(rewards, sort_keys=True) + "\n")
    diagnostics_path.write_text(json.dumps(diagnostics, sort_keys=True) + "\n")


def main() -> int:
    rewards, diagnostics = evaluate_database(DATABASE_PATH)
    write_reports(rewards, diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
