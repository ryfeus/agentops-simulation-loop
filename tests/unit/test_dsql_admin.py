from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from agentops_demo.billing.dsql_schema import SCHEMA_STATEMENTS
from agentops_demo.validation.scenario import load_scenario
from scripts.dsql_admin import SCENARIO_PATH, reset_state


class Transaction:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def start(self) -> None:
        self.events.append("start")

    async def commit(self) -> None:
        self.events.append("commit")

    async def rollback(self) -> None:
        self.events.append("rollback")


class Connection:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.executions: list[tuple[str, object]] = []

    def transaction(self) -> Transaction:
        return Transaction(self.events)

    async def execute(self, query: str) -> None:
        self.executions.append((query, None))

    async def executemany(self, query: str, values: object) -> None:
        self.executions.append((query, values))


def test_dsql_schema_preserves_contract_constraints() -> None:
    schema = "\n".join(SCHEMA_STATEMENTS)
    assert "NUMERIC(12, 2)" in schema
    assert "refunds_invoice_unique" in schema
    assert "escalations_invoice_unique" in schema
    assert "invoices_customer_fk" in schema


@pytest.mark.asyncio
async def test_reset_uses_canonical_scenario_and_decimal() -> None:
    connection = Connection()
    scenario = load_scenario(Path(SCENARIO_PATH))
    await reset_state(connection, scenario.initial_state)

    assert connection.events == ["start", "commit"]
    invoice_insert = next(
        values for query, values in connection.executions if "INSERT INTO invoices" in query
    )
    assert invoice_insert == [("inv-123", "customer-42", Decimal("89.00"), "USD", "disputed")]
    assert any(query == "DELETE FROM refunds" for query, _values in connection.executions)


@pytest.mark.asyncio
async def test_admin_connector_uses_verified_operator_profile(monkeypatch):
    import sys
    from types import SimpleNamespace

    from scripts import aws_context
    from scripts.dsql_admin import connect_admin

    monkeypatch.setenv("AWS_PROFILE", "billing-sandbox")
    events = []
    monkeypatch.setattr(aws_context, "verified_session", lambda: events.append("verified"))

    async def connector(**kwargs):
        events.append(kwargs)
        return "synthetic connection"

    monkeypatch.setitem(sys.modules, "aurora_dsql_asyncpg", SimpleNamespace(connect=connector))
    assert await connect_admin("synthetic.invalid", "us-west-2") == "synthetic connection"
    assert events[0] == "verified"
    assert events[1]["profile"] == "billing-sandbox"
