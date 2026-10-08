"""Normalized, runtime-independent scenario contract."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from agentops_demo.contracts._base import ContractModel
from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.contracts.billing import (
    Customer,
    EscalateDisputeRequest,
    EscalationRecord,
    GetInvoiceRequest,
    Invoice,
    InvoiceStatus,
    RefundInvoiceRequest,
    RefundRecord,
)


class EvaluationEvidence(ContractModel):
    """Optional evaluator evidence associated with scenario discovery or curation."""

    evaluator: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reason: str | None = None


class ScenarioSource(ContractModel):
    """Provider-neutral origin of a normalized scenario."""

    kind: Literal["synthetic", "trace"]
    trace_ref: str | None = None
    evaluation: EvaluationEvidence | None = None

    @model_validator(mode="after")
    def validate_trace_reference(self) -> ScenarioSource:
        if self.kind == "synthetic" and self.trace_ref is not None:
            raise ValueError("synthetic scenario sources cannot contain a trace reference")
        if self.kind == "trace" and not self.trace_ref:
            raise ValueError("trace scenario sources require a non-empty trace reference")
        return self


class ScenarioProvenance(ContractModel):
    """Origin metadata, including the configuration that produced a failure if known."""

    source: ScenarioSource
    originating_agent_config: AgentConfig | None = None


class InitialState(ContractModel):
    """Deterministic billing state loaded before a replay."""

    customers: list[Customer]
    invoices: list[Invoice]
    refunds: list[RefundRecord]
    escalations: list[EscalationRecord]

    @model_validator(mode="after")
    def validate_entity_references(self) -> InitialState:
        collections = {
            "customer": self.customers,
            "invoice": self.invoices,
            "refund": self.refunds,
            "escalation": self.escalations,
        }
        for kind, entities in collections.items():
            seen: set[str] = set()
            for entity in entities:
                if entity.id in seen:
                    raise ValueError(f"duplicate {kind} id {entity.id!r}")
                seen.add(entity.id)

        customer_ids = {customer.id for customer in self.customers}
        for invoice in self.invoices:
            if invoice.customer_id not in customer_ids:
                raise ValueError(
                    f"invoice {invoice.id!r} references unknown customer {invoice.customer_id!r}"
                )

        invoice_ids = {invoice.id for invoice in self.invoices}
        for refund in self.refunds:
            if refund.invoice_id not in invoice_ids:
                raise ValueError(
                    f"refund {refund.id!r} references unknown invoice {refund.invoice_id!r}"
                )
        for escalation in self.escalations:
            if escalation.invoice_id not in invoice_ids:
                raise ValueError(
                    f"escalation {escalation.id!r} references unknown invoice "
                    f"{escalation.invoice_id!r}"
                )
        return self


class GetInvoiceToolCall(ContractModel):
    tool: Literal["get_invoice"]
    arguments: GetInvoiceRequest


class RefundInvoiceToolCall(ContractModel):
    tool: Literal["refund_invoice"]
    arguments: RefundInvoiceRequest


class EscalateDisputeToolCall(ContractModel):
    tool: Literal["escalate_dispute"]
    arguments: EscalateDisputeRequest


ObservedToolCall = Annotated[
    GetInvoiceToolCall | RefundInvoiceToolCall | EscalateDisputeToolCall,
    Field(discriminator="tool"),
]


class ObservedFailure(ContractModel):
    """The tool trajectory that demonstrated the policy failure."""

    tool_calls: list[ObservedToolCall] = Field(min_length=1)


class MutationPolicy(ContractModel):
    """Structured task intent used by deterministic benchmark evaluators."""

    allowed_targets: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_targets(self) -> MutationPolicy:
        if len(self.allowed_targets) != len(set(self.allowed_targets)):
            raise ValueError("mutation policy has duplicate allowed targets")
        return self


class TrajectoryExpectation(ContractModel):
    """Optional structural requirements in addition to final-state invariants."""

    required_inspections: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    require_inspection_before_mutation: bool = False
    max_write_attempts_per_invoice: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_required_inspections(self) -> TrajectoryExpectation:
        if any(not invoice_id.strip() for invoice_id in self.required_inspections):
            raise ValueError("required inspections must not contain blank invoice IDs")
        if len(self.required_inspections) != len(set(self.required_inspections)):
            raise ValueError("required inspections must be unique")
        return self


class SyntheticGenerationMetadata(ContractModel):
    """Stable lineage for a deterministically generated benchmark scenario."""

    generator: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    prototype_id: str = Field(min_length=1)
    prototype_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    variant_index: int = Field(ge=0)
    dimensions: dict[str, str]


class BenchmarkMetadata(ContractModel):
    """Labels and intent needed to run a concrete static benchmark task."""

    category: Literal["billing"]
    archetype: str = Field(min_length=1, pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
    difficulty: Literal["easy", "paraphrase", "adversarial", "compound"]
    mutation_policy: MutationPolicy = Field(default_factory=MutationPolicy)
    trajectory: TrajectoryExpectation = Field(default_factory=TrajectoryExpectation)
    generation: SyntheticGenerationMetadata | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class NoRefundInvariant(ContractModel):
    """Require no newly-created refund for the referenced invoice."""

    type: Literal["no_refund"]
    invoice_id: str = Field(min_length=1)


class MustEscalateInvariant(ContractModel):
    """Require exactly one newly-created escalation for the referenced invoice."""

    type: Literal["must_escalate"]
    invoice_id: str = Field(min_length=1)


class MustRefundInvariant(ContractModel):
    """Require exactly one newly-created refund for the referenced invoice."""

    type: Literal["must_refund"]
    invoice_id: str = Field(min_length=1)


class NoEscalationInvariant(ContractModel):
    """Require no newly-created escalation for the referenced invoice."""

    type: Literal["no_escalation"]
    invoice_id: str = Field(min_length=1)


class InvoiceStatusInvariant(ContractModel):
    """Require a concrete final invoice status."""

    type: Literal["invoice_status"]
    invoice_id: str = Field(min_length=1)
    status: InvoiceStatus


class UnchangedInvariant(ContractModel):
    """Require the invoice row and its mutation records to remain identical."""

    type: Literal["unchanged"]
    invoice_id: str = Field(min_length=1)


ExpectedInvariant = Annotated[
    NoRefundInvariant
    | MustEscalateInvariant
    | MustRefundInvariant
    | NoEscalationInvariant
    | InvoiceStatusInvariant
    | UnchangedInvariant,
    Field(discriminator="type"),
]


class Scenario(ContractModel):
    """Canonical intermediate form between traces and benchmark tasks."""

    schema_version: Literal["1"]
    id: str = Field(min_length=1, pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
    instruction: str = Field(min_length=1)
    provenance: ScenarioProvenance
    initial_state: InitialState
    observed_failure: ObservedFailure | None = None
    expected_invariants: list[ExpectedInvariant] = Field(min_length=1)
    benchmark: BenchmarkMetadata | None = None

    @model_validator(mode="after")
    def validate_invoice_references(self) -> Scenario:
        invoice_ids = {invoice.id for invoice in self.initial_state.invoices}

        if self.provenance.source.kind == "trace" and self.observed_failure is None:
            raise ValueError("trace scenario sources require observed failure evidence")

        if self.observed_failure is not None:
            for call in self.observed_failure.tool_calls:
                invoice_id = call.arguments.invoice_id
                if invoice_id not in invoice_ids:
                    raise ValueError(
                        "observed tool call "
                        f"{call.tool!r} references unknown invoice {invoice_id!r}"
                    )

        for invariant in self.expected_invariants:
            if invariant.invoice_id not in invoice_ids:
                raise ValueError(
                    f"invariant {invariant.type!r} references unknown invoice "
                    f"{invariant.invoice_id!r}"
                )

        if self.benchmark is not None:
            unknown_targets = set(self.benchmark.mutation_policy.allowed_targets) - invoice_ids
            if unknown_targets:
                raise ValueError(
                    f"mutation policy references unknown invoice {sorted(unknown_targets)[0]!r}"
                )

        if self.provenance.originating_agent_config is not None:
            self.provenance.originating_agent_config.fingerprint()
        return self
