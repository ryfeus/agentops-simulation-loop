"""Domain-oriented billing tool and repository contracts.

Business invariant: a disputed invoice must not be automatically refunded. It
must be escalated for specialist review. Phase 0 records this rule but does not
implement tool behavior.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BeforeValidator, Field

from agentops_demo.contracts._base import ContractModel

InvoiceStatus = Literal["open", "paid", "refunded", "disputed"]
CustomerStatus = Literal["active", "inactive"]
BillingToolName = Literal["get_invoice", "refund_invoice", "escalate_dispute"]


def _parse_money(value: object) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError("monetary values must be strings, integers, or Decimal values")
    try:
        amount = Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("invalid monetary value") from exc
    if not amount.is_finite():
        raise ValueError("monetary values must be finite")
    return amount


Money = Annotated[Decimal, BeforeValidator(_parse_money), Field(gt=0)]
EntityId = Annotated[str, Field(min_length=1)]
CurrencyCode = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
Reason = Annotated[str, Field(min_length=1)]


class Customer(ContractModel):
    """Synthetic billing customer."""

    id: EntityId
    status: CustomerStatus


class Invoice(ContractModel):
    """Invoice returned by the billing domain."""

    id: EntityId
    customer_id: EntityId
    amount: Money
    currency: CurrencyCode
    status: InvoiceStatus


class RefundRecord(ContractModel):
    """Persisted refund state used by deterministic scenarios."""

    id: EntityId
    invoice_id: EntityId
    reason: Reason


class EscalationRecord(ContractModel):
    """Persisted dispute escalation used by deterministic scenarios."""

    id: EntityId
    invoice_id: EntityId
    reason: Reason


class GetInvoiceRequest(ContractModel):
    invoice_id: EntityId


class RefundInvoiceRequest(ContractModel):
    invoice_id: EntityId
    reason: Reason


class RefundInvoiceResult(ContractModel):
    success: bool
    invoice_id: EntityId
    reason: str | None = None


class EscalateDisputeRequest(ContractModel):
    invoice_id: EntityId
    reason: Reason


class EscalateDisputeResult(ContractModel):
    success: bool
    invoice_id: EntityId
    escalation_id: str | None = None


@runtime_checkable
class BillingRepository(Protocol):
    """Backend-agnostic storage boundary required by the future MCP service."""

    async def get_invoice(self, invoice_id: str) -> Invoice | None: ...

    async def refund_invoice(self, invoice_id: str, reason: str) -> RefundRecord: ...

    async def escalate_dispute(self, invoice_id: str, reason: str) -> EscalationRecord: ...


@runtime_checkable
class BillingWorldInspector(Protocol):
    """Internal read-only boundary used for pre-invocation trace evidence."""

    async def snapshot_world(self): ...
