"""Small, stable domain error surface for billing operations."""


class BillingDomainError(Exception):
    """Base error whose string form remains useful across MCP transport."""

    code = "billing_error"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f"{self.code}: {message}")


class InvoiceNotFoundError(BillingDomainError):
    code = "invoice_not_found"


class RefundAlreadyExistsError(BillingDomainError):
    code = "refund_already_exists"


class EscalationAlreadyExistsError(BillingDomainError):
    code = "escalation_already_exists"


class DisputedInvoiceRequiresEscalationError(BillingDomainError):
    code = "disputed_invoice_requires_escalation"


class RecordIdConflictError(BillingDomainError):
    code = "record_id_conflict"


class InvalidPolicyModeError(BillingDomainError):
    code = "invalid_policy_mode"
