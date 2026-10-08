"""Internal, pre-invocation billing-world evidence contract."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from agentops_demo.contracts._base import ContractModel
from agentops_demo.contracts.billing import Customer, EscalationRecord, Invoice, RefundRecord
from agentops_demo.contracts.scenario import InitialState


class BillingWorldSnapshot(ContractModel):
    """The complete, tiny synthetic billing world seen before an invocation."""

    schema_version: Literal["1"]
    customers: list[Customer]
    invoices: list[Invoice]
    refunds: list[RefundRecord]
    escalations: list[EscalationRecord]

    def initial_state(self) -> InitialState:
        return InitialState(
            customers=self.customers,
            invoices=self.invoices,
            refunds=self.refunds,
            escalations=self.escalations,
        )


def canonical_snapshot_json(snapshot: BillingWorldSnapshot) -> str:
    """Return the stable compact representation hashed and attached to traces."""

    return json.dumps(
        snapshot.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def snapshot_sha256(snapshot: BillingWorldSnapshot) -> str:
    return hashlib.sha256(canonical_snapshot_json(snapshot).encode("utf-8")).hexdigest()
