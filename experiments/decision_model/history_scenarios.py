"""Freeze an author-constructed history challenge without modifying historical benchmarks.

Eight new compositions, three presentation variants each. This is a prospective
stress cohort, not an independently sourced benchmark or 24 independent families.
No rule/model inference is used in authoring, rendering, or expectation generation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agentops_demo.benchmark.phase10_variants import behavior_signature_sha256
from agentops_demo.contracts.scenario import Scenario
from agentops_demo.taskify.harbor_renderer import render_harbor_task
from agentops_demo.taskify.integrity import scenario_sha256
from rl.phase8b.execution_task import derive_execution_task

ROOT = Path(__file__).resolve().parents[2]
COHORT_ID = "billing-history-v2-challenge-v1"
DEFAULT_OUTPUT = ROOT / ".artifacts/decision-model/history-v2"
VARIANTS = 3
# Roles specify initial status, existing ledger kinds, intended NEW operation, customer.
# None status denotes an absent invoice that must still be inspected.
PROTOTYPES = (
    {
        "id": "history-v2-paid-review-cross-operation",
        "intent": "Prior escalation does not suppress a first refund; skip an existing paid-state "
        "review while creating an unrelated new dispute review.",
        "roles": {
            "a": ("paid", ("escalations",), "refund_invoice", 0),
            "b": ("paid", ("escalations",), None, 0),
            "c": ("disputed", (), "escalate_dispute", 0),
        },
        "instruction": "Inspect {a}, {b}, and {c}. Refund {a}; its earlier specialist review "
        "does not cancel this separate refund request. Escalate {b} and {c} for specialist "
        "review only if each invoice has no escalation already recorded. Do not create "
        "duplicate reviews or refund either of those two invoices.",
        "primary_history_probe": True,
    },
    {
        "id": "history-v2-three-disputes-distinct-permissions",
        "intent": "Identical disputed statuses require different decisions: existing review, "
        "new authorized review, and explicitly read-only dispute.",
        "roles": {
            "a": ("disputed", ("escalations",), None, 0),
            "b": ("disputed", (), "escalate_dispute", 0),
            "c": ("disputed", (), None, 0),
        },
        "instruction": "Inspect all three disputes: {a}, {b}, and {c}. Escalate {a} and {b} "
        "unless a specialist case already exists for that invoice. For {c}, report its "
        "status only; I am not authorizing a review or refund for {c}. Never refund a "
        "disputed charge or repeat an existing escalation.",
        "primary_history_probe": True,
    },
    {
        "id": "history-v2-scoped-audit-with-active-refund",
        "intent": "Read-only intent is invoice-local, despite a separate refund in the same "
        "request; audited paid/disputed/refunded invoices have distinct histories.",
        "roles": {
            "a": ("paid", ("escalations",), None, 0),
            "b": ("disputed", (), None, 0),
            "c": ("paid", (), "refund_invoice", 0),
            "d": ("refunded", ("refunds",), None, 0),
        },
        "instruction": "Inspect {a}, {b}, {c}, and {d}. I need a status report only for {a} "
        "and {b}; leave both unchanged, including any specialist case. Separately, refund "
        "{c}. Confirm the refund already recorded for {d} without repeating it. The "
        "information-only restriction on the first two invoices does not apply to {c}.",
        "primary_history_probe": True,
    },
    {
        "id": "history-v2-missing-target-independent-review",
        "intent": "A missing refund target must not authorize substitution or prevent an "
        "independent new review; another invoice has a settled refund history.",
        "roles": {
            "a": (None, (), None, 0),
            "b": ("paid", ("escalations",), None, 0),
            "c": ("disputed", (), "escalate_dispute", 0),
            "d": ("refunded", ("refunds",), None, 0),
        },
        "instruction": "Inspect {a}, {b}, {c}, and {d}. My refund request is for {a} only; "
        "if it cannot be found, report that and do not substitute {b}. Leave {b} untouched. "
        "Independently, escalate {c} for review unless already escalated. Check the "
        "previous refund on {d}, but make no further changes to {d}.",
        "primary_history_probe": True,
    },
    {
        "id": "history-v2-refund-ledger-status-drift-control",
        "intent": "A refund ledger survives status drift back to paid; escalation-only "
        "history cannot resolve this negative control's missing refund information.",
        "roles": {
            "a": ("paid", ("refunds",), None, 0),
            "b": ("paid", (), "refund_invoice", 0),
            "c": ("disputed", ("escalations",), None, 0),
        },
        "instruction": "Inspect {a}, {b}, and {c}. For {a}, check whether my earlier refund "
        "was recorded; a current paid status alone is not proof that it failed. Do not "
        "repeat a recorded refund, and leave {a} unchanged if the necessary refund history "
        "is unavailable. Separately, refund {b}; this is my first refund request for it. "
        "Check the existing specialist case for {c} without creating another case or a refund.",
        "primary_history_probe": False,
        "unobserved_refund_history_control": True,
    },
    {
        "id": "history-v2-two-customer-equal-charge-routing",
        "intent": "Equal charge amounts and repeated statuses cross customer boundaries; "
        "history and explicit customer/invoice scope determine permissible actions.",
        "roles": {
            "a": ("paid", (), "refund_invoice", 0),
            "b": ("disputed", ("escalations",), None, 0),
            "c": ("disputed", (), "escalate_dispute", 1),
            "d": ("paid", (), None, 1),
        },
        "instruction": "Inspect {d}, {b}, {a}, and {c}; their matching amounts do not make "
        "them interchangeable. For {customer0}, refund {a} and escalate {b} only if no "
        "case already exists. For {customer1}, escalate {c} unless already escalated; "
        "{d} is a status-only inquiry and must remain unchanged. Keep actions attached "
        "to their exact invoice and customer.",
        "primary_history_probe": True,
    },
    {
        "id": "history-v2-held-refund-authorization-active-dispute",
        "intent": "Two unresolved refund candidates have no write authorization, while "
        "a separate dispute does; clarification prose itself is not verifier-scored.",
        "roles": {
            "a": ("paid", (), None, 0),
            "b": ("paid", ("escalations",), None, 0),
            "c": ("disputed", (), "escalate_dispute", 0),
        },
        "instruction": "Inspect {a}, {b}, and {c}. I have not decided whether {a} or {b} "
        "is the invoice I want refunded: change neither until I identify it. Ask me to "
        "clarify that choice. Separately, I authorize specialist review of {c} now; "
        "escalate it unless a case already exists. Do not postpone that independent "
        "review because my refund choice is unresolved.",
        "primary_history_probe": True,
        "clarification_prose_not_scored": True,
    },
    {
        "id": "history-v2-refunded-review-open-status-mix",
        "intent": "A refunded invoice retains both ledgers; a different paid invoice may "
        "receive its first refund despite an old escalation; an open bill is read-only.",
        "roles": {
            "a": ("refunded", ("refunds", "escalations"), None, 0),
            "b": ("paid", ("escalations",), "refund_invoice", 0),
            "c": ("open", (), None, 0),
        },
        "instruction": "Inspect {a}, {b}, and {c}. Confirm the existing refund and "
        "specialist review for {a}, with no additional refund or escalation. Refund "
        "{b}; its past specialist case does not satisfy this new refund request. "
        "For open invoice {c}, provide its status only and leave it untouched.",
        "primary_history_probe": True,
    },
)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def make_scenario(spec: dict[str, Any], index: int) -> Scenario:
    if index not in range(VARIANTS):
        raise ValueError("Only the three frozen presentation variants are supported")
    number = 100000 + int(fingerprint([COHORT_ID, spec["id"], index])[:8], 16) % 800000
    roles = list(spec["roles"])
    identifiers = {role: f"inv-{number + offset}" for offset, role in enumerate(roles)}
    customers = {owner for _, _, _, owner in spec["roles"].values()}
    customer_ids = {owner: f"customer-{number + owner}" for owner in customers}
    state: dict[str, Any] = {
        "customers": [
            {"id": customer_ids[owner], "status": "active"} for owner in sorted(customers)
        ],
        "invoices": [],
        "refunds": [],
        "escalations": [],
    }
    invariants, allowed = [], []
    for role, (status, histories, operation, owner) in spec["roles"].items():
        if status is None:
            continue
        invoice = identifiers[role]
        state["invoices"].append(
            {
                "id": invoice,
                "customer_id": customer_ids[owner],
                "amount": ("39.50", "95.00", "120.25")[index],
                "currency": "USD",
                "status": status,
            }
        )
        for history in histories:
            state[history].append(
                {
                    "id": f"prior-{history}-{invoice}",
                    "invoice_id": invoice,
                    "reason": "Previously completed billing operation",
                }
            )
        if operation:
            refund = operation == "refund_invoice"
            allowed.append(invoice)
            invariants.extend(
                [
                    {"type": "must_refund" if refund else "must_escalate", "invoice_id": invoice},
                    {"type": "no_escalation" if refund else "no_refund", "invoice_id": invoice},
                    {
                        "type": "invoice_status",
                        "invoice_id": invoice,
                        "status": "refunded" if refund else "disputed",
                    },
                ]
            )
        else:
            invariants.append({"type": "unchanged", "invoice_id": invoice})
    # Variants change presentation only: do not count these as independent compositions.
    state["invoices"] = state["invoices"][index:] + state["invoices"][:index]
    ordered = roles[index:] + roles[:index]
    instruction = spec["instruction"].format(
        **identifiers,
        **{f"customer{owner}": value for owner, value in customer_ids.items()},
    )
    return Scenario.model_validate(
        {
            "schema_version": "1",
            "id": f"{spec['id']}-v{index:02d}",
            "instruction": instruction,
            "provenance": {"source": {"kind": "synthetic"}},
            "initial_state": state,
            "expected_invariants": invariants,
            "benchmark": {
                "category": "billing",
                "archetype": "multi-invoice",
                "difficulty": "compound",
                "mutation_policy": {"allowed_targets": allowed},
                "trajectory": {
                    "required_inspections": [identifiers[r] for r in ordered],
                    "require_inspection_before_mutation": True,
                    "max_write_attempts_per_invoice": 1,
                },
                "generation": {
                    "generator": COHORT_ID,
                    "family_id": spec["id"],
                    "prototype_id": spec["id"],
                    "prototype_sha256": fingerprint(spec),
                    "variant_index": index,
                    "dimensions": {
                        "entity_order": f"rotate-{index}",
                        "inspection_order": f"rotate-{index}",
                        "amount": ("39.50", "95.00", "120.25")[index],
                    },
                },
            },
        }
    )


def structural_signature(scenario: Scenario) -> str:
    """Remove archetype naming from the historical behavior hash: renaming is not novelty."""
    payload = scenario.model_dump(mode="json")
    payload["benchmark"]["archetype"] = "comparison"
    # A redundant final-status assertion must not manufacture structural novelty.
    payload["expected_invariants"] = [
        item for item in payload["expected_invariants"] if item["type"] != "invoice_status"
    ]
    return behavior_signature_sha256(Scenario.model_validate(payload))


def normalized_instruction(instruction: str) -> str:
    return re.sub(r"\b(?:inv|customer)-[\w-]+", "ENTITY", " ".join(instruction.split()))


def scenarios() -> dict[str, Scenario]:
    return {
        scenario.id: scenario
        for spec in PROTOTYPES
        for index in range(VARIANTS)
        for scenario in [make_scenario(spec, index)]
    }


def novelty_audit(current: dict[str, Scenario]) -> dict[str, Any]:
    from rl.phase12.corpus import scenarios as historical_scenarios

    historical = [task for tasks in historical_scenarios().values() for task in tasks.values()]
    old_signatures = {structural_signature(task) for task in historical}
    old_instructions = {normalized_instruction(task.instruction) for task in historical}
    old_ids = {task.id for task in historical}
    collisions = [
        key for key, task in current.items() if structural_signature(task) in old_signatures
    ]
    if collisions or set(current) & old_ids:
        raise ValueError(f"New cohort overlaps historical task structure: {collisions}")
    if any(
        normalized_instruction(task.instruction) in old_instructions for task in current.values()
    ):
        raise ValueError("New cohort repeats a historical normalized instruction")
    signatures = {structural_signature(task) for task in current.values()}
    if len(signatures) != len(PROTOTYPES):
        raise ValueError("Cohort prototypes are not structurally distinct")
    return {
        "historical_tasks_compared": len(historical),
        "structural_collisions": 0,
        "normalized_instruction_collisions": 0,
        "unique_new_structures": len(signatures),
        "method": "Behavior signature normalizes archetype and redundant status invariants; "
        "exact normalized instructions.",
        "limitation": "Structural non-overlap is not semantic independence or external authorship.",
    }


async def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Frozen cohort destination already exists: {output}")
    data = scenarios()
    novelty = novelty_audit(data)
    output.mkdir(parents=True)
    (output / "prototype-specs.json").write_text(json.dumps(PROTOTYPES, indent=2) + "\n")
    specs = {spec["id"]: spec for spec in PROTOTYPES}
    metadata = {}
    for task_id, scenario in sorted(data.items()):
        assert scenario.benchmark and scenario.benchmark.generation
        spec = specs[scenario.benchmark.generation.prototype_id]
        canonical = output / "canonical" / task_id
        await render_harbor_task(scenario, canonical)
        task_path = output / "tasks" / task_id
        execution = derive_execution_task(canonical, task_path)
        metadata[task_id] = {
            "task_id": task_id,
            "group_id": spec["id"],
            "family_id": spec["id"],
            "prototype_id": spec["id"],
            "scenario_sha256": scenario_sha256(scenario),
            "structural_sha256": structural_signature(scenario),
            "task_path": str(task_path),
            "base_task_sha256": execution.base_task_sha256,
            "execution_task_sha256": execution.execution_task_sha256,
            "seed_sql_sha256": hashlib.sha256(
                (task_path / "environment/seed.sql").read_bytes()
            ).hexdigest(),
            "expected_operations": {
                item.invoice_id: "refund_invoice"
                if item.type == "must_refund"
                else "escalate_dispute"
                for item in scenario.expected_invariants
                if item.type in {"must_refund", "must_escalate"}
            },
            "required_inspections": scenario.benchmark.trajectory.required_inspections,
            "expected_no_write_targets": [
                item.invoice_id for item in scenario.expected_invariants if item.type == "unchanged"
            ],
            "author_intent": spec["intent"],
            "primary_history_probe": spec["primary_history_probe"],
            "unobserved_refund_history_control": spec.get(
                "unobserved_refund_history_control", False
            ),
            "clarification_prose_not_scored": spec.get("clarification_prose_not_scored", False),
        }
    # Selection-only records are deliberately not model examples or supervision.
    selection = [
        {"task_id": key, "group_id": item["group_id"], "split": "challenge"}
        for key, item in metadata.items()
    ]
    (output / "selection.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selection))
    sources = [
        Path(__file__),
        ROOT / "benchmarks/billing/phase12/manifest.json",
        ROOT / "benchmarks/billing/phase12/prototypes.json",
        ROOT / "benchmarks/disputed-refund/tests/verify.py",
    ]
    manifest = {
        "schema_version": "1",
        "cohort_id": COHORT_ID,
        "task_count": len(data),
        "prototype_count": len(PROTOTYPES),
        "variants_per_prototype": VARIANTS,
        "primary_history_probe_tasks": sum(t["primary_history_probe"] for t in metadata.values()),
        "prototype_specs_sha256": fingerprint(PROTOTYPES),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "novelty_audit": novelty,
        "tasks": metadata,
        "limitations": [
            "Author-constructed after the first experiment; not an external benchmark.",
            "Eight compositions with three variants each; not 24 independent families.",
            "Generation uses no training, calibration, or outcome-based scenario editing.",
            "Refund-ledger drift is a negative control: escalation history cannot reveal refunds.",
            "Held authorization checks inspections and writes; clarification prose is unscored.",
            "Harness fixes target resolution; entity discovery is outside this cohort.",
        ],
    }
    manifest["cohort_sha256"] = fingerprint(manifest)
    (output / "cohort-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = asyncio.run(build(args.output))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "cohort_id",
                    "cohort_sha256",
                    "task_count",
                    "prototype_count",
                    "novelty_audit",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
