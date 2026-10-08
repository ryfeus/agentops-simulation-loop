"""Sandbox-side JSON bridge for the Phase 8B billing Harbor harness."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any

BILLING_DATABASE_PATH = Path(os.environ.get("BILLING_DATABASE_PATH", "/app/data/billing.db"))
AGENT_RUN_PATH = Path(os.environ.get("AGENT_RUN_PATH", "/app/data/agent-run.json"))
OPERATIONS = frozenset({"get_invoice", "refund_invoice", "escalate_dispute"})


def _paths() -> tuple[Path, Path]:
    return (
        Path(os.environ.get("BILLING_DATABASE_PATH", str(BILLING_DATABASE_PATH))),
        Path(os.environ.get("AGENT_RUN_PATH", str(AGENT_RUN_PATH))),
    )


def _record_call(path: Path, operation: str, arguments: dict[str, str]) -> None:
    """Atomically append one attempted logical tool call before execution."""

    payload: dict[str, Any] = {"tool_calls": []}
    if path.is_file():
        loaded = json.loads(path.read_text())
        if isinstance(loaded, dict) and isinstance(loaded.get("tool_calls"), list):
            payload = loaded
    payload["tool_calls"].append({"name": operation, "arguments": arguments})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _decode_payload(value: str) -> dict[str, str]:
    try:
        decoded = base64.b64decode(value, validate=True)
        payload = json.loads(decoded)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("payload must be base64-encoded JSON") from exc
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("payload must be a JSON object")
    if not all(isinstance(item, str) for item in payload.values()):
        raise ValueError("tool arguments must be strings")
    return payload


async def invoke(operation: str, arguments: dict[str, str]) -> dict[str, Any]:
    """Run one billing operation and return its transport-safe observation."""

    if operation not in OPERATIONS:
        raise ValueError(f"unsupported billing operation {operation!r}")
    invoice_id = arguments.get("invoice_id")
    if not invoice_id:
        raise ValueError("invoice_id is required")
    if operation != "get_invoice" and not arguments.get("reason"):
        raise ValueError("reason is required")

    database_path, agent_run_path = _paths()
    _record_call(agent_run_path, operation, arguments)

    from agentops_demo.billing.errors import BillingDomainError
    from agentops_demo.billing.service import BillingService
    from agentops_demo.billing.sqlite_repository import SQLiteBillingRepository

    service = BillingService(SQLiteBillingRepository(database_path), policy_mode="permissive")
    try:
        if operation == "get_invoice":
            invoice = await service.get_invoice(invoice_id)
            if invoice is None:
                return {
                    "ok": False,
                    "error_type": "InvoiceNotFoundError",
                    "error": f"invoice {invoice_id!r} does not exist",
                }
            return {"ok": True, "invoice": invoice.model_dump(mode="json")}
        if operation == "refund_invoice":
            result = await service.refund_invoice(invoice_id, arguments["reason"])
            return {"ok": True, "refund": result.model_dump(mode="json")}
        result = await service.escalate_dispute(invoice_id, arguments["reason"])
        return {"ok": True, "escalation": result.model_dump(mode="json")}
    except BillingDomainError as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=sorted(OPERATIONS))
    parser.add_argument("payload")
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(invoke(args.operation, _decode_payload(args.payload)))
    except Exception as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
