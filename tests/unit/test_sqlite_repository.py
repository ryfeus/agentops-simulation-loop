from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import aiosqlite
import pytest
from conftest import GOLDEN_SCENARIO

from agentops_demo.billing.errors import (
    EscalationAlreadyExistsError,
    InvoiceNotFoundError,
    RecordIdConflictError,
    RefundAlreadyExistsError,
)
from agentops_demo.billing.schema import SCHEMA_SQL
from agentops_demo.billing.sqlite_repository import (
    SQLiteBillingRepository,
    initialize_database,
)
from agentops_demo.contracts.billing import Customer, Invoice, RefundRecord
from agentops_demo.contracts.scenario import InitialState
from agentops_demo.validation.scenario import load_scenario


async def golden_repository(path: Path, **kwargs: object) -> SQLiteBillingRepository:
    scenario = load_scenario(GOLDEN_SCENARIO)
    await initialize_database(path, scenario.initial_state)
    return SQLiteBillingRepository(path, **kwargs)  # type: ignore[arg-type]


async def test_get_existing_invoice_preserves_decimal(tmp_path: Path) -> None:
    repository = await golden_repository(tmp_path / "billing.db")

    invoice = await repository.get_invoice("inv-123")

    assert invoice is not None
    assert invoice.amount == Decimal("89.00")
    assert str(invoice.amount) == "89.00"
    assert invoice.status == "disputed"


async def test_get_missing_invoice_returns_none(tmp_path: Path) -> None:
    repository = await golden_repository(tmp_path / "billing.db")

    assert await repository.get_invoice("missing") is None


async def test_create_refund_uses_deterministic_id_and_updates_status(tmp_path: Path) -> None:
    repository = await golden_repository(tmp_path / "billing.db")

    record = await repository.refund_invoice("inv-123", "duplicate charge")

    assert record.id == "refund-inv-123"
    assert await repository.mutation_counts("inv-123") == (1, 0)
    invoice = await repository.get_invoice("inv-123")
    assert invoice is not None
    assert invoice.status == "refunded"


async def test_create_escalation_uses_injected_id_and_updates_status(tmp_path: Path) -> None:
    repository = await golden_repository(
        tmp_path / "billing.db",
        id_factory=lambda kind, invoice_id: f"custom-{kind}-{invoice_id}",
    )

    record = await repository.escalate_dispute("inv-123", "specialist review")

    assert record.id == "custom-escalation-inv-123"
    assert await repository.mutation_counts("inv-123") == (0, 1)
    invoice = await repository.get_invoice("inv-123")
    assert invoice is not None
    assert invoice.status == "disputed"


async def test_duplicate_refund_is_rejected(tmp_path: Path) -> None:
    repository = await golden_repository(tmp_path / "billing.db")
    await repository.refund_invoice("inv-123", "first")

    with pytest.raises(RefundAlreadyExistsError) as exc_info:
        await repository.refund_invoice("inv-123", "second")

    assert exc_info.value.code == "refund_already_exists"
    assert await repository.mutation_counts("inv-123") == (1, 0)


async def test_duplicate_escalation_is_rejected(tmp_path: Path) -> None:
    repository = await golden_repository(tmp_path / "billing.db")
    await repository.escalate_dispute("inv-123", "first")

    with pytest.raises(EscalationAlreadyExistsError) as exc_info:
        await repository.escalate_dispute("inv-123", "second")

    assert exc_info.value.code == "escalation_already_exists"
    assert await repository.mutation_counts("inv-123") == (0, 1)


@pytest.mark.parametrize("table", ["refunds", "escalations"])
async def test_schema_rejects_duplicate_mutations_for_one_invoice(
    tmp_path: Path, table: str
) -> None:
    path = tmp_path / "billing.db"
    await golden_repository(path)

    async with aiosqlite.connect(path) as connection:
        await connection.execute(
            f"INSERT INTO {table} (id, invoice_id, reason) VALUES (?, ?, ?)",
            ("first", "inv-123", "first mutation"),
        )
        with pytest.raises(aiosqlite.IntegrityError, match="UNIQUE constraint failed"):
            await connection.execute(
                f"INSERT INTO {table} (id, invoice_id, reason) VALUES (?, ?, ?)",
                ("second", "inv-123", "bypasses repository guard"),
            )


async def test_initialize_database_upgrades_earlier_schema_cardinality(tmp_path: Path) -> None:
    path = tmp_path / "billing.db"
    async with aiosqlite.connect(path) as connection:
        await connection.executescript(SCHEMA_SQL)
        await connection.execute("INSERT INTO customers VALUES ('customer-old', 'active')")
        await connection.execute(
            "INSERT INTO invoices VALUES ('inv-old', 'customer-old', '1.00', 'USD', 'paid')"
        )
        await connection.execute("INSERT INTO refunds VALUES ('refund-1', 'inv-old', 'first')")
        await connection.execute("INSERT INTO refunds VALUES ('refund-2', 'inv-old', 'second')")
        await connection.commit()

    scenario = load_scenario(GOLDEN_SCENARIO)
    await initialize_database(path, scenario.initial_state)

    async with aiosqlite.connect(path) as connection:
        refund_cursor = await connection.execute("PRAGMA index_list('refunds')")
        escalation_cursor = await connection.execute("PRAGMA index_list('escalations')")
        refund_indexes = {row[1] for row in await refund_cursor.fetchall()}
        escalation_indexes = {row[1] for row in await escalation_cursor.fetchall()}

    assert "refunds_invoice_id_unique" in refund_indexes
    assert "escalations_invoice_id_unique" in escalation_indexes


@pytest.mark.parametrize("operation", ["refund", "escalate"])
async def test_mutating_unknown_invoice_is_rejected(tmp_path: Path, operation: str) -> None:
    repository = await golden_repository(tmp_path / "billing.db")

    with pytest.raises(InvoiceNotFoundError):
        if operation == "refund":
            await repository.refund_invoice("missing", "reason")
        else:
            await repository.escalate_dispute("missing", "reason")


async def test_initialize_database_is_idempotent_and_replaces_state(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "billing.db"
    scenario = load_scenario(GOLDEN_SCENARIO)
    await initialize_database(path, scenario.initial_state)
    repository = SQLiteBillingRepository(path)
    await repository.refund_invoice("inv-123", "temporary")

    await initialize_database(path, scenario.initial_state)

    assert path.exists()
    assert await repository.mutation_counts("inv-123") == (0, 0)
    invoice = await repository.get_invoice("inv-123")
    assert invoice is not None
    assert invoice.status == "disputed"


async def test_record_id_collision_rolls_back_status_change(tmp_path: Path) -> None:
    path = tmp_path / "billing.db"
    initial_state = InitialState(
        customers=[Customer(id="customer-1", status="active")],
        invoices=[
            Invoice(
                id="inv-1",
                customer_id="customer-1",
                amount="10.00",
                currency="USD",
                status="paid",
            ),
            Invoice(
                id="inv-2",
                customer_id="customer-1",
                amount="20.00",
                currency="USD",
                status="paid",
            ),
        ],
        refunds=[RefundRecord(id="collision", invoice_id="inv-2", reason="existing")],
        escalations=[],
    )
    await initialize_database(path, initial_state)
    repository = SQLiteBillingRepository(path, id_factory=lambda kind, invoice_id: "collision")

    with pytest.raises(RecordIdConflictError):
        await repository.refund_invoice("inv-1", "new")

    assert await repository.mutation_counts("inv-1") == (0, 0)
    invoice = await repository.get_invoice("inv-1")
    assert invoice is not None
    assert invoice.status == "paid"
