from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from agentops_demo.contracts.billing import (
    BillingRepository,
    EscalateDisputeRequest,
    EscalateDisputeResult,
    GetInvoiceRequest,
    Invoice,
    RefundInvoiceRequest,
    RefundInvoiceResult,
)


@pytest.mark.parametrize("status", ["open", "paid", "refunded", "disputed"])
def test_all_invoice_statuses_are_valid(status: str) -> None:
    invoice = Invoice(
        id="inv-1",
        customer_id="customer-1",
        amount="10.00",
        currency="USD",
        status=status,
    )

    assert invoice.status == status


def test_invalid_invoice_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Invoice(
            id="inv-1",
            customer_id="customer-1",
            amount="10.00",
            currency="USD",
            status="cancelled",
        )


def test_money_remains_precise_and_serializes_as_string() -> None:
    invoice = Invoice(
        id="inv-1",
        customer_id="customer-1",
        amount="89.10",
        currency="USD",
        status="open",
    )

    assert invoice.amount == Decimal("89.10")
    assert invoice.model_dump(mode="json")["amount"] == "89.10"


@pytest.mark.parametrize("amount", ["not-money", "NaN", "Infinity", "-1.00", "0", True])
def test_invalid_money_is_rejected(amount: str | bool) -> None:
    with pytest.raises(ValidationError):
        Invoice(
            id="inv-1",
            customer_id="customer-1",
            amount=amount,
            currency="USD",
            status="open",
        )


def test_float_money_is_rejected_to_avoid_precision_loss() -> None:
    with pytest.raises(ValidationError):
        Invoice(
            id="inv-1",
            customer_id="customer-1",
            amount=89.1,
            currency="USD",
            status="open",
        )


@pytest.mark.parametrize("currency", ["usd", "US", "USDD", "12A"])
def test_invalid_currency_is_rejected(currency: str) -> None:
    with pytest.raises(ValidationError):
        Invoice(
            id="inv-1",
            customer_id="customer-1",
            amount="10.00",
            currency=currency,
            status="open",
        )


def test_request_and_result_contracts_parse() -> None:
    assert GetInvoiceRequest(invoice_id="inv-1").invoice_id == "inv-1"
    assert RefundInvoiceRequest(invoice_id="inv-1", reason="duplicate").reason == "duplicate"
    assert EscalateDisputeRequest(invoice_id="inv-1", reason="policy").reason == "policy"
    assert RefundInvoiceResult(success=True, invoice_id="inv-1").success
    assert (
        EscalateDisputeResult(success=True, invoice_id="inv-1", escalation_id="esc-1").escalation_id
        == "esc-1"
    )


def test_blank_request_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RefundInvoiceRequest(invoice_id=" ", reason=" ")


def test_extra_contract_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        GetInvoiceRequest(invoice_id="inv-1", sql="select 1")


def test_repository_surface_contains_exactly_domain_operations() -> None:
    public_methods = {
        name
        for name, value in BillingRepository.__dict__.items()
        if callable(value) and not name.startswith("_")
    }

    assert public_methods == {"get_invoice", "refund_invoice", "escalate_dispute"}
