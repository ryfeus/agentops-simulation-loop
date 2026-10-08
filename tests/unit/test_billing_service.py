from __future__ import annotations

from pathlib import Path

import pytest
from conftest import GOLDEN_SCENARIO
from pydantic import ValidationError

from agentops_demo.billing.errors import (
    DisputedInvoiceRequiresEscalationError,
    EscalationAlreadyExistsError,
    InvalidPolicyModeError,
    InvoiceNotFoundError,
    RefundAlreadyExistsError,
)
from agentops_demo.billing.service import BillingService
from agentops_demo.billing.sqlite_repository import (
    SQLiteBillingRepository,
    initialize_database,
)
from agentops_demo.validation.scenario import load_scenario


async def service_for(path: Path, policy_mode: str = "permissive") -> BillingService:
    scenario = load_scenario(GOLDEN_SCENARIO)
    await initialize_database(path, scenario.initial_state)
    return BillingService(SQLiteBillingRepository(path), policy_mode=policy_mode)


async def test_get_invoice_uses_phase_zero_contract(tmp_path: Path) -> None:
    service = await service_for(tmp_path / "billing.db")

    invoice = await service.get_invoice("inv-123")

    assert invoice is not None
    assert invoice.id == "inv-123"


async def test_blank_direct_request_is_rejected(tmp_path: Path) -> None:
    service = await service_for(tmp_path / "billing.db")

    with pytest.raises(ValidationError):
        await service.refund_invoice("inv-123", " ")


async def test_permissive_mode_refunds_disputed_invoice(tmp_path: Path) -> None:
    path = tmp_path / "billing.db"
    service = await service_for(path)

    result = await service.refund_invoice("inv-123", "duplicate charge")

    assert result.success
    assert result.invoice_id == "inv-123"
    assert await SQLiteBillingRepository(path).mutation_counts("inv-123") == (1, 0)


async def test_enforced_mode_rejects_disputed_refund_without_write(tmp_path: Path) -> None:
    path = tmp_path / "billing.db"
    service = await service_for(path, "enforced")

    with pytest.raises(DisputedInvoiceRequiresEscalationError) as exc_info:
        await service.refund_invoice("inv-123", "duplicate charge")

    assert exc_info.value.code == "disputed_invoice_requires_escalation"
    assert await SQLiteBillingRepository(path).mutation_counts("inv-123") == (0, 0)


async def test_escalate_dispute_returns_phase_zero_result(tmp_path: Path) -> None:
    path = tmp_path / "billing.db"
    service = await service_for(path)

    result = await service.escalate_dispute("inv-123", "specialist review")

    assert result.success
    assert result.escalation_id == "escalation-inv-123"
    assert await SQLiteBillingRepository(path).mutation_counts("inv-123") == (0, 1)


@pytest.mark.parametrize("operation", ["refund", "escalate"])
async def test_unknown_invoice_error_is_explicit(
    tmp_path: Path,
    operation: str,
) -> None:
    service = await service_for(tmp_path / "billing.db")

    with pytest.raises(InvoiceNotFoundError) as exc_info:
        if operation == "refund":
            await service.refund_invoice("missing", "reason")
        else:
            await service.escalate_dispute("missing", "reason")

    assert "invoice_not_found" in str(exc_info.value)


async def test_duplicate_errors_cross_service_boundary(tmp_path: Path) -> None:
    service = await service_for(tmp_path / "billing.db")
    await service.refund_invoice("inv-123", "first refund")
    await service.escalate_dispute("inv-123", "first escalation")

    with pytest.raises(RefundAlreadyExistsError):
        await service.refund_invoice("inv-123", "second refund")
    with pytest.raises(EscalationAlreadyExistsError):
        await service.escalate_dispute("inv-123", "second escalation")


def test_invalid_policy_mode_is_rejected() -> None:
    with pytest.raises(InvalidPolicyModeError) as exc_info:
        BillingService(repository=object(), policy_mode="unknown")  # type: ignore[arg-type]

    assert exc_info.value.code == "invalid_policy_mode"
