from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import pytest

from agentops_demo.billing.dsql_repository import (
    DSQLBillingRepository,
    run_write_transaction,
)
from agentops_demo.billing.dsql_settings import DSQLSettings
from agentops_demo.billing.errors import (
    InvoiceNotFoundError,
    RecordIdConflictError,
    RefundAlreadyExistsError,
)


class FakeTransaction:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def start(self) -> None:
        self.events.append("start")

    async def commit(self) -> None:
        self.events.append("commit")

    async def rollback(self) -> None:
        self.events.append("rollback")


class FakeConnection:
    def __init__(
        self,
        rows: list[Mapping[str, Any] | None],
        *,
        execute_error: Exception | None = None,
    ) -> None:
        self.rows = rows
        self.execute_error = execute_error
        self.events: list[str] = []
        self.queries: list[tuple[str, tuple[object, ...]]] = []

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self.events)

    async def fetchrow(self, query: str, *args: object) -> Mapping[str, Any] | None:
        self.queries.append((query, args))
        return self.rows.pop(0)

    async def execute(self, query: str, *args: object) -> str:
        self.queries.append((query, args))
        if self.execute_error is not None:
            raise self.execute_error
        return "OK"

    async def close(self) -> None:
        self.events.append("close")


class DatabaseError(Exception):
    def __init__(self, sqlstate: str, constraint_name: str = "") -> None:
        self.sqlstate = sqlstate
        self.constraint_name = constraint_name
        super().__init__(sqlstate)


def settings(max_retries: int = 0) -> DSQLSettings:
    return DSQLSettings(endpoint="endpoint", region="us-west-2", max_retries=max_retries)


@pytest.mark.asyncio
async def test_get_invoice_preserves_decimal() -> None:
    connection = FakeConnection(
        [
            {
                "id": "inv-123",
                "customer_id": "customer-42",
                "amount": Decimal("89.00"),
                "currency": "USD",
                "status": "disputed",
            }
        ]
    )

    async def connect(_settings: DSQLSettings) -> FakeConnection:
        return connection

    invoice = await DSQLBillingRepository(settings(), connection_factory=connect).get_invoice(
        "inv-123"
    )
    assert invoice is not None
    assert invoice.amount == Decimal("89.00")
    assert connection.events == ["close"]


@pytest.mark.asyncio
async def test_refund_transaction_sequence() -> None:
    connection = FakeConnection([{"id": "inv-123"}, None])

    async def connect(_settings: DSQLSettings) -> FakeConnection:
        return connection

    record = await DSQLBillingRepository(settings(), connection_factory=connect).refund_invoice(
        "inv-123", "duplicate charge"
    )
    assert record.id == "refund-inv-123"
    assert connection.events == ["start", "commit", "close"]
    assert any("INSERT INTO refunds" in query for query, _ in connection.queries)
    assert any("UPDATE invoices" in query for query, _ in connection.queries)


@pytest.mark.asyncio
async def test_missing_invoice_rolls_back() -> None:
    connection = FakeConnection([None])

    async def connect(_settings: DSQLSettings) -> FakeConnection:
        return connection

    with pytest.raises(InvoiceNotFoundError):
        await DSQLBillingRepository(settings(), connection_factory=connect).refund_invoice(
            "missing", "reason"
        )
    assert connection.events == ["start", "rollback", "close"]


@pytest.mark.asyncio
async def test_existing_refund_uses_domain_error() -> None:
    connection = FakeConnection([{"id": "inv-123"}, {"id": "refund-inv-123"}])

    async def connect(_settings: DSQLSettings) -> FakeConnection:
        return connection

    with pytest.raises(RefundAlreadyExistsError):
        await DSQLBillingRepository(settings(), connection_factory=connect).refund_invoice(
            "inv-123", "reason"
        )
    assert connection.events == ["start", "rollback", "close"]


@pytest.mark.asyncio
async def test_unique_constraint_mapping() -> None:
    connection = FakeConnection(
        [{"id": "inv-123"}, None],
        execute_error=DatabaseError("23505", "refunds_invoice_unique"),
    )

    async def connect(_settings: DSQLSettings) -> FakeConnection:
        return connection

    with pytest.raises(RefundAlreadyExistsError):
        await DSQLBillingRepository(settings(), connection_factory=connect).refund_invoice(
            "inv-123", "reason"
        )


@pytest.mark.asyncio
async def test_unknown_unique_constraint_is_record_conflict() -> None:
    connection = FakeConnection(
        [{"id": "inv-123"}, None], execute_error=DatabaseError("23505", "refunds_pkey")
    )

    async def connect(_settings: DSQLSettings) -> FakeConnection:
        return connection

    with pytest.raises(RecordIdConflictError):
        await DSQLBillingRepository(settings(), connection_factory=connect).refund_invoice(
            "inv-123", "reason"
        )


@pytest.mark.asyncio
async def test_occ_retry_then_success() -> None:
    attempts = 0
    delays: list[float] = []

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DatabaseError("40001")
        return "done"

    async def sleep(delay: float) -> None:
        delays.append(delay)

    result = await run_write_transaction(operation, max_retries=2, sleep=sleep, jitter=lambda: 0)
    assert result == "done"
    assert attempts == 2
    assert delays == [0.01]


@pytest.mark.asyncio
async def test_occ_retry_is_bounded() -> None:
    attempts = 0

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise DatabaseError("OC000")

    async def sleep(_delay: float) -> None:
        pass

    with pytest.raises(DatabaseError, match="OC000"):
        await run_write_transaction(operation, max_retries=2, sleep=sleep)
    assert attempts == 3


@pytest.mark.asyncio
async def test_non_occ_error_is_not_retried() -> None:
    attempts = 0

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise DatabaseError("42501")

    with pytest.raises(DatabaseError, match="42501"):
        await run_write_transaction(operation, max_retries=4)
    assert attempts == 1
