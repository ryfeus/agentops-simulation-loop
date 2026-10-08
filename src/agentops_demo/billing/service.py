"""Billing domain behavior layered over the storage protocol."""

from __future__ import annotations

from typing import Literal, cast

from agentops_demo.billing.errors import (
    DisputedInvoiceRequiresEscalationError,
    InvalidPolicyModeError,
    InvoiceNotFoundError,
)
from agentops_demo.contracts.billing import (
    BillingRepository,
    EscalateDisputeRequest,
    EscalateDisputeResult,
    GetInvoiceRequest,
    Invoice,
    RefundInvoiceRequest,
    RefundInvoiceResult,
)

BillingPolicyMode = Literal["permissive", "enforced"]


class BillingService:
    """Validate billing requests and enforce the selected experiment policy."""

    def __init__(
        self,
        repository: BillingRepository,
        *,
        policy_mode: BillingPolicyMode | str = "permissive",
    ) -> None:
        if policy_mode not in ("permissive", "enforced"):
            raise InvalidPolicyModeError(f"unsupported billing policy mode {policy_mode!r}")
        self.repository = repository
        self.policy_mode = cast(BillingPolicyMode, policy_mode)

    async def get_invoice(self, invoice_id: str) -> Invoice | None:
        request = GetInvoiceRequest(invoice_id=invoice_id)
        return await self.repository.get_invoice(request.invoice_id)

    async def refund_invoice(self, invoice_id: str, reason: str) -> RefundInvoiceResult:
        request = RefundInvoiceRequest(invoice_id=invoice_id, reason=reason)
        invoice = await self.repository.get_invoice(request.invoice_id)
        if invoice is None:
            raise InvoiceNotFoundError(f"invoice {request.invoice_id!r} does not exist")
        if self.policy_mode == "enforced" and invoice.status == "disputed":
            raise DisputedInvoiceRequiresEscalationError(
                f"invoice {request.invoice_id!r} is disputed and must be escalated"
            )

        record = await self.repository.refund_invoice(request.invoice_id, request.reason)
        return RefundInvoiceResult(success=True, invoice_id=record.invoice_id)

    async def escalate_dispute(self, invoice_id: str, reason: str) -> EscalateDisputeResult:
        request = EscalateDisputeRequest(invoice_id=invoice_id, reason=reason)
        invoice = await self.repository.get_invoice(request.invoice_id)
        if invoice is None:
            raise InvoiceNotFoundError(f"invoice {request.invoice_id!r} does not exist")

        record = await self.repository.escalate_dispute(request.invoice_id, request.reason)
        return EscalateDisputeResult(
            success=True,
            invoice_id=record.invoice_id,
            escalation_id=record.id,
        )
