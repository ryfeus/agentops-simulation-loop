"""SQLite implementation of the Phase 0 billing repository contract."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import aiosqlite

from agentops_demo.billing.errors import (
    EscalationAlreadyExistsError,
    InvoiceNotFoundError,
    RecordIdConflictError,
    RefundAlreadyExistsError,
)
from agentops_demo.billing.schema import CARDINALITY_INDEX_SQL, SCHEMA_SQL
from agentops_demo.contracts.billing import EscalationRecord, Invoice, RefundRecord
from agentops_demo.contracts.scenario import InitialState
from agentops_demo.contracts.world_snapshot import BillingWorldSnapshot

RecordKind = Literal["refund", "escalation"]
RecordIdFactory = Callable[[RecordKind, str], str]


def deterministic_record_id(kind: RecordKind, invoice_id: str) -> str:
    """Return stable local IDs that make demo output and tests repeatable."""

    return f"{kind}-{invoice_id}"


class SQLiteBillingRepository:
    """Persist billing state without owning billing policy decisions."""

    def __init__(
        self,
        path: str | Path,
        *,
        id_factory: RecordIdFactory = deterministic_record_id,
    ) -> None:
        self.path = Path(path)
        self.id_factory = id_factory

    async def get_invoice(self, invoice_id: str) -> Invoice | None:
        async with self._connection() as connection:
            cursor = await connection.execute(
                """
                SELECT id, customer_id, amount, currency, status
                FROM invoices
                WHERE id = ?
                """,
                (invoice_id,),
            )
            row = await cursor.fetchone()
        return _invoice_from_row(row) if row is not None else None

    async def refund_invoice(self, invoice_id: str, reason: str) -> RefundRecord:
        async with self._connection() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                await _require_invoice(connection, invoice_id)
                if await _record_exists(connection, "refunds", invoice_id):
                    raise RefundAlreadyExistsError(
                        f"a refund already exists for invoice {invoice_id!r}"
                    )
                record = RefundRecord(
                    id=self.id_factory("refund", invoice_id),
                    invoice_id=invoice_id,
                    reason=reason,
                )
                await connection.execute(
                    "INSERT INTO refunds (id, invoice_id, reason) VALUES (?, ?, ?)",
                    (record.id, record.invoice_id, record.reason),
                )
                await connection.execute(
                    "UPDATE invoices SET status = 'refunded' WHERE id = ?",
                    (invoice_id,),
                )
                await connection.commit()
            except aiosqlite.IntegrityError as exc:
                await connection.rollback()
                raise RecordIdConflictError(
                    f"could not create refund record for invoice {invoice_id!r}"
                ) from exc
            except Exception:
                await connection.rollback()
                raise
        return record

    async def escalate_dispute(self, invoice_id: str, reason: str) -> EscalationRecord:
        async with self._connection() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                await _require_invoice(connection, invoice_id)
                if await _record_exists(connection, "escalations", invoice_id):
                    raise EscalationAlreadyExistsError(
                        f"an escalation already exists for invoice {invoice_id!r}"
                    )
                record = EscalationRecord(
                    id=self.id_factory("escalation", invoice_id),
                    invoice_id=invoice_id,
                    reason=reason,
                )
                await connection.execute(
                    "INSERT INTO escalations (id, invoice_id, reason) VALUES (?, ?, ?)",
                    (record.id, record.invoice_id, record.reason),
                )
                await connection.execute(
                    "UPDATE invoices SET status = 'disputed' WHERE id = ?",
                    (invoice_id,),
                )
                await connection.commit()
            except aiosqlite.IntegrityError as exc:
                await connection.rollback()
                raise RecordIdConflictError(
                    f"could not create escalation record for invoice {invoice_id!r}"
                ) from exc
            except Exception:
                await connection.rollback()
                raise
        return record

    async def mutation_counts(self, invoice_id: str) -> tuple[int, int]:
        """Return refund and escalation counts for local inspection and tests."""

        async with self._connection() as connection:
            refund_cursor = await connection.execute(
                "SELECT COUNT(*) FROM refunds WHERE invoice_id = ?", (invoice_id,)
            )
            escalation_cursor = await connection.execute(
                "SELECT COUNT(*) FROM escalations WHERE invoice_id = ?", (invoice_id,)
            )
            refund_row = await refund_cursor.fetchone()
            escalation_row = await escalation_cursor.fetchone()
        assert refund_row is not None
        assert escalation_row is not None
        return int(refund_row[0]), int(escalation_row[0])

    async def snapshot_world(self) -> BillingWorldSnapshot:
        """Read the complete synthetic world in deterministic primary-key order."""

        async with self._connection() as connection:
            customers = await (
                await connection.execute("SELECT id, status FROM customers ORDER BY id")
            ).fetchall()
            invoices = await (
                await connection.execute(
                    "SELECT id, customer_id, amount, currency, status FROM invoices ORDER BY id"
                )
            ).fetchall()
            refunds = await (
                await connection.execute("SELECT id, invoice_id, reason FROM refunds ORDER BY id")
            ).fetchall()
            escalations = await (
                await connection.execute(
                    "SELECT id, invoice_id, reason FROM escalations ORDER BY id"
                )
            ).fetchall()
        return BillingWorldSnapshot(
            schema_version="1",
            customers=[{"id": row["id"], "status": row["status"]} for row in customers],
            invoices=[
                {
                    "id": row["id"],
                    "customer_id": row["customer_id"],
                    "amount": row["amount"],
                    "currency": row["currency"],
                    "status": row["status"],
                }
                for row in invoices
            ],
            refunds=[
                {"id": row["id"], "invoice_id": row["invoice_id"], "reason": row["reason"]}
                for row in refunds
            ],
            escalations=[
                {"id": row["id"], "invoice_id": row["invoice_id"], "reason": row["reason"]}
                for row in escalations
            ],
        )

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[aiosqlite.Connection]:
        connection = await aiosqlite.connect(self.path)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            await connection.close()


async def initialize_database(path: str | Path, initial_state: InitialState) -> None:
    """Create and transactionally replace a SQLite database from scenario state."""

    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(database_path) as connection:
        await connection.executescript(SCHEMA_SQL)
        await connection.execute("PRAGMA foreign_keys = ON")
        await connection.execute("BEGIN IMMEDIATE")
        try:
            for table in ("refunds", "escalations", "invoices", "customers"):
                await connection.execute(f"DELETE FROM {table}")
            for statement in CARDINALITY_INDEX_SQL:
                await connection.execute(statement)

            await connection.executemany(
                "INSERT INTO customers (id, status) VALUES (?, ?)",
                [(customer.id, customer.status) for customer in initial_state.customers],
            )
            await connection.executemany(
                """
                INSERT INTO invoices (id, customer_id, amount, currency, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        invoice.id,
                        invoice.customer_id,
                        str(invoice.amount),
                        invoice.currency,
                        invoice.status,
                    )
                    for invoice in initial_state.invoices
                ],
            )
            await connection.executemany(
                "INSERT INTO refunds (id, invoice_id, reason) VALUES (?, ?, ?)",
                [(record.id, record.invoice_id, record.reason) for record in initial_state.refunds],
            )
            await connection.executemany(
                "INSERT INTO escalations (id, invoice_id, reason) VALUES (?, ?, ?)",
                [
                    (record.id, record.invoice_id, record.reason)
                    for record in initial_state.escalations
                ],
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise


async def _require_invoice(connection: aiosqlite.Connection, invoice_id: str) -> None:
    cursor = await connection.execute("SELECT 1 FROM invoices WHERE id = ?", (invoice_id,))
    if await cursor.fetchone() is None:
        raise InvoiceNotFoundError(f"invoice {invoice_id!r} does not exist")


async def _record_exists(
    connection: aiosqlite.Connection,
    table: Literal["refunds", "escalations"],
    invoice_id: str,
) -> bool:
    cursor = await connection.execute(
        f"SELECT 1 FROM {table} WHERE invoice_id = ? LIMIT 1",
        (invoice_id,),
    )
    return await cursor.fetchone() is not None


def _invoice_from_row(row: aiosqlite.Row) -> Invoice:
    return Invoice(
        id=row["id"],
        customer_id=row["customer_id"],
        amount=row["amount"],
        currency=row["currency"],
        status=row["status"],
    )
