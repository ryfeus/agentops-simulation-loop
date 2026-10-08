"""Aurora DSQL implementation of the stable billing repository contract."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol, cast

from agentops_demo.billing.dsql_settings import DSQLSettings
from agentops_demo.billing.errors import (
    EscalationAlreadyExistsError,
    InvoiceNotFoundError,
    RecordIdConflictError,
    RefundAlreadyExistsError,
)
from agentops_demo.billing.sqlite_repository import RecordIdFactory, deterministic_record_id
from agentops_demo.contracts.billing import EscalationRecord, Invoice, RefundRecord
from agentops_demo.contracts.world_snapshot import BillingWorldSnapshot

OCC_SQLSTATES = frozenset({"40001", "OC000", "OC001"})
Sleep = Callable[[float], Awaitable[None]]
Jitter = Callable[[], float]


class Transaction(Protocol):
    async def start(self) -> None: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...


class Connection(Protocol):
    def transaction(self) -> Transaction: ...
    async def fetchrow(self, query: str, *args: object) -> Mapping[str, Any] | None: ...
    async def fetch(self, query: str, *args: object) -> list[Mapping[str, Any]]: ...
    async def execute(self, query: str, *args: object) -> str: ...
    async def close(self) -> None: ...


ConnectionFactory = Callable[[DSQLSettings], Awaitable[Connection]]


async def _connect(settings: DSQLSettings) -> Connection:
    try:
        import aurora_dsql_asyncpg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Aurora DSQL support is not installed; run with the 'dsql' project extra"
        ) from exc
    return cast(
        Connection,
        await aurora_dsql_asyncpg.connect(
            host=settings.endpoint,
            region=settings.region,
            user=settings.user,
            database=settings.database,
        ),
    )


def _sqlstate(error: BaseException) -> str | None:
    return cast(str | None, getattr(error, "sqlstate", getattr(error, "sql_state", None)))


async def run_write_transaction(
    operation: Callable[[], Awaitable[Any]],
    *,
    max_retries: int,
    sleep: Sleep = asyncio.sleep,
    jitter: Jitter = random.random,
) -> Any:
    """Retry a complete write transaction only for documented DSQL OCC conflicts."""

    retries = 0
    while True:
        try:
            return await operation()
        except Exception as exc:
            if _sqlstate(exc) not in OCC_SQLSTATES or retries >= max_retries:
                raise
            base_delay = min(0.01 * (2**retries), 0.25)
            await sleep(base_delay * (1 + 0.25 * jitter()))
            retries += 1


class DSQLBillingRepository:
    """Persist billing state in Aurora DSQL using short IAM-authenticated connections."""

    def __init__(
        self,
        settings: DSQLSettings,
        *,
        connection_factory: ConnectionFactory = _connect,
        id_factory: RecordIdFactory = deterministic_record_id,
        sleep: Sleep = asyncio.sleep,
        jitter: Jitter = random.random,
    ) -> None:
        self.settings = settings
        self.connection_factory = connection_factory
        self.id_factory = id_factory
        self.sleep = sleep
        self.jitter = jitter

    async def get_invoice(self, invoice_id: str) -> Invoice | None:
        connection = await self.connection_factory(self.settings)
        try:
            row = await connection.fetchrow(
                """
                SELECT id, customer_id, amount, currency, status
                FROM invoices
                WHERE id = $1
                """,
                invoice_id,
            )
        finally:
            await connection.close()
        return _invoice_from_row(row) if row is not None else None

    async def snapshot_world(self) -> BillingWorldSnapshot:
        """Capture the demo world through one read-only DSQL connection."""

        connection = await self.connection_factory(self.settings)
        try:
            transaction = connection.transaction()
            await transaction.start()
            try:
                customers = await connection.fetch("SELECT id, status FROM customers ORDER BY id")
                invoices = await connection.fetch(
                    "SELECT id, customer_id, amount, currency, status FROM invoices ORDER BY id"
                )
                refunds = await connection.fetch(
                    "SELECT id, invoice_id, reason FROM refunds ORDER BY id"
                )
                escalations = await connection.fetch(
                    "SELECT id, invoice_id, reason FROM escalations ORDER BY id"
                )
                await transaction.commit()
            except Exception:
                await transaction.rollback()
                raise
        finally:
            await connection.close()
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

    async def refund_invoice(self, invoice_id: str, reason: str) -> RefundRecord:
        record = RefundRecord(
            id=self.id_factory("refund", invoice_id), invoice_id=invoice_id, reason=reason
        )

        async def operation() -> RefundRecord:
            await self._write_mutation("refund", record)
            return record

        return cast(
            RefundRecord,
            await run_write_transaction(
                operation,
                max_retries=self.settings.max_retries,
                sleep=self.sleep,
                jitter=self.jitter,
            ),
        )

    async def escalate_dispute(self, invoice_id: str, reason: str) -> EscalationRecord:
        record = EscalationRecord(
            id=self.id_factory("escalation", invoice_id), invoice_id=invoice_id, reason=reason
        )

        async def operation() -> EscalationRecord:
            await self._write_mutation("escalation", record)
            return record

        return cast(
            EscalationRecord,
            await run_write_transaction(
                operation,
                max_retries=self.settings.max_retries,
                sleep=self.sleep,
                jitter=self.jitter,
            ),
        )

    async def _write_mutation(self, kind: str, record: RefundRecord | EscalationRecord) -> None:
        connection = await self.connection_factory(self.settings)
        transaction = connection.transaction()
        await transaction.start()
        try:
            invoice = await connection.fetchrow(
                "SELECT id FROM invoices WHERE id = $1", record.invoice_id
            )
            if invoice is None:
                raise InvoiceNotFoundError(f"invoice {record.invoice_id!r} does not exist")
            table = "refunds" if kind == "refund" else "escalations"
            existing = await connection.fetchrow(
                f"SELECT id FROM {table} WHERE invoice_id = $1 LIMIT 1", record.invoice_id
            )
            if existing is not None:
                if kind == "refund":
                    raise RefundAlreadyExistsError(
                        f"a refund already exists for invoice {record.invoice_id!r}"
                    )
                raise EscalationAlreadyExistsError(
                    f"an escalation already exists for invoice {record.invoice_id!r}"
                )
            await connection.execute(
                f"INSERT INTO {table} (id, invoice_id, reason) VALUES ($1, $2, $3)",
                record.id,
                record.invoice_id,
                record.reason,
            )
            status = "refunded" if kind == "refund" else "disputed"
            await connection.execute(
                "UPDATE invoices SET status = $1 WHERE id = $2", status, record.invoice_id
            )
            await transaction.commit()
        except Exception as exc:
            await transaction.rollback()
            mapped = _map_write_error(exc, kind, record.invoice_id)
            if mapped is exc:
                raise
            raise mapped from exc
        finally:
            await connection.close()


def _map_write_error(error: Exception, kind: str, invoice_id: str) -> Exception:
    if _sqlstate(error) not in {"23505", "23000"}:
        return error
    constraint = str(getattr(error, "constraint_name", ""))
    if constraint == "refunds_invoice_unique":
        return RefundAlreadyExistsError(f"a refund already exists for invoice {invoice_id!r}")
    if constraint == "escalations_invoice_unique":
        return EscalationAlreadyExistsError(
            f"an escalation already exists for invoice {invoice_id!r}"
        )
    return RecordIdConflictError(f"could not create {kind} record for invoice {invoice_id!r}")


def _invoice_from_row(row: Mapping[str, Any]) -> Invoice:
    return Invoice(
        id=row["id"],
        customer_id=row["customer_id"],
        amount=row["amount"],
        currency=row["currency"],
        status=row["status"],
    )
