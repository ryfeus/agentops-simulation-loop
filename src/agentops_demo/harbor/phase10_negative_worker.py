"""Container-side stdlib worker for one deliberately wrong billing trajectory."""

from __future__ import annotations

import base64
import json
import sqlite3
import sys
from pathlib import Path


def run(
    payload: dict[str, object],
    *,
    database_path: Path = Path("/app/data/billing.db"),
    artifact_path: Path = Path("/app/data/agent-run.json"),
    evidence_path: Path = Path("/logs/agent/negative-execution.json"),
) -> dict[str, object]:
    strategy = str(payload["strategy"])
    target = str(payload["target_invoice"])
    wrong = str(payload["wrong_invoice"])
    inspections = payload["inspection_ids"]
    assert isinstance(inspections, list)
    calls: list[dict[str, object]] = []
    with sqlite3.connect(database_path) as connection:
        if strategy != "noop" and strategy != "mutate_without_inspection":
            for invoice_id in inspections:
                connection.execute("SELECT id FROM invoices WHERE id = ?", (invoice_id,)).fetchone()
                calls.append({"name": "get_invoice", "arguments": {"invoice_id": invoice_id}})

        def write(operation: str, invoice_id: str, sequence: int) -> None:
            if not invoice_id:
                raise ValueError("negative strategy lacks an invoice target")
            reason = "Phase 10 known wrong action"
            if operation == "refund_invoice":
                connection.execute(
                    "INSERT OR IGNORE INTO refunds (id, invoice_id, reason) VALUES (?, ?, ?)",
                    (f"negative-refund-{sequence}-{invoice_id}", invoice_id, reason),
                )
                connection.execute(
                    "UPDATE invoices SET status = 'refunded' WHERE id = ?", (invoice_id,)
                )
            else:
                connection.execute(
                    "INSERT OR IGNORE INTO escalations (id, invoice_id, reason) VALUES (?, ?, ?)",
                    (f"negative-escalation-{sequence}-{invoice_id}", invoice_id, reason),
                )
                connection.execute(
                    "UPDATE invoices SET status = 'disputed' WHERE id = ?", (invoice_id,)
                )
            calls.append(
                {"name": operation, "arguments": {"invoice_id": invoice_id, "reason": reason}}
            )

        if strategy in {"refund_target", "mutate_without_inspection"}:
            write("refund_invoice", target, 0)
        elif strategy == "escalate_target":
            write("escalate_dispute", target, 0)
        elif strategy == "refund_wrong_target":
            write("refund_invoice", wrong, 0)
        elif strategy == "escalate_wrong_target":
            write("escalate_dispute", wrong, 0)
        elif strategy == "duplicate_refund":
            write("refund_invoice", target, 0)
            write("refund_invoice", target, 1)
        elif strategy == "duplicate_escalation":
            write("escalate_dispute", target, 0)
            write("escalate_dispute", target, 1)
        elif strategy not in {"noop", "lookup_only"}:
            raise ValueError(f"unsupported strategy {strategy}")

    result = {"strategy": strategy, "tool_calls": calls}
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"tool_calls": calls}, sort_keys=True))
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    payload = json.loads(base64.b64decode(sys.argv[1]))
    run(payload)
