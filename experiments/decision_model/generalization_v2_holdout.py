"""Freeze prospective invoice-local billing families without policy/model exposure.

Author-written operations and negative controls are the specification. Validation
constructs offline final-state fixtures, never teacher or policy trajectories.
This is a new compositional holdout in the same billing domain, not an external
benchmark or evidence of semantic independence from known billing primitives.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import tempfile
import types
from collections import Counter
from pathlib import Path
from typing import Any

from agentops_demo.billing.sqlite_repository import initialize_database
from agentops_demo.contracts.scenario import InitialState, Scenario
from agentops_demo.taskify.harbor_renderer import render_harbor_task
from agentops_demo.taskify.integrity import scenario_sha256
from experiments.decision_model.history_scenarios import (
    normalized_instruction,
    structural_signature,
)
from rl.phase8b.execution_task import derive_execution_task

ROOT = Path(__file__).resolve().parents[2]
COHORT_ID = "billing-generalization-v2"
DEFAULT_OUTPUT = ROOT / ".artifacts/decision-model-generalization-v2/corpus"
VARIANTS = 2
# (initial status, existing ledger kinds, required NEW operation, customer index)
PROTOTYPES = (
    {
        "id": "generalization-v2-two-households-cross-operation-history",
        "intent": "Both households need a first refund on a previously reviewed paid invoice "
        "and a first review on a different paid invoice, while existing dispute cases "
        "remain intact.",
        "roles": {
            "a": ("paid", ("escalations",), "refund_invoice", 0),
            "b": ("paid", (), "escalate_dispute", 0),
            "c": ("disputed", ("escalations",), None, 0),
            "d": ("paid", ("escalations",), "refund_invoice", 1),
            "e": ("paid", (), "escalate_dispute", 1),
            "f": ("disputed", ("escalations",), None, 1),
            "g": ("open", (), None, 1),
        },
        "instruction": "Inspect {a}, {b}, {c}, {d}, {e}, {f}, and {g}. For {customer0}, "
        "refund {a} and escalate {b} for specialist review. For {customer1}, refund {d} "
        "and escalate {e} for specialist review. The reviews previously attached to {a} "
        "and {d} must remain intact; neither is a recorded refund. Inspect the existing "
        "cases on {c} and {f} without opening another case or refunding those invoices. "
        "For {g}, report status only.",
    },
    {
        "id": "generalization-v2-two-withdrawals-preserve-separate-grants",
        "intent": "Separate withdrawals revoke one of two refund commands and one of two "
        "review commands; the other refund and review remain authorized.",
        "roles": {
            "a": ("paid", (), None, 0),
            "b": ("paid", (), "refund_invoice", 1),
            "c": ("disputed", (), None, 1),
            "d": ("disputed", (), "escalate_dispute", 0),
            "e": ("paid", ("escalations",), None, 0),
            "f": ("refunded", ("refunds",), None, 1),
        },
        "instruction": "Inspect {a}, {b}, {c}, {d}, {e}, and {f}. Refund {a} and {b}. "
        "Escalate {c} and {d} for specialist review. I withdraw permission to refund "
        "{a}; leave {a} unchanged. I also withdraw permission to escalate {c}; leave "
        "{c} unchanged. The refund of {b} and the review of {d} are still requested. "
        "For {e}, inspect the existing case without repeating it or issuing a refund. "
        "For {f}, confirm the recorded refund without another write.",
    },
    {
        "id": "generalization-v2-ledger-asymmetry-with-missing-reference",
        "intent": "Possible prior refund evidence does not transfer to a different paid "
        "invoice whose only prior operation is a review, and a separate first review remains due.",
        "roles": {
            "a": ("paid", ("refunds",), None, 0),
            "b": ("paid", ("escalations",), "refund_invoice", 0),
            "c": ("disputed", ("escalations",), None, 1),
            "d": ("paid", (), "escalate_dispute", 1),
            "e": ("open", (), None, 0),
            "f": (None, (), None, 1),
        },
        "instruction": "Inspect {a}, {b}, {c}, {d}, {e}, and {f}. Do not refund {a}: "
        "an earlier refund may already have been recorded. Explain any gap in the "
        "available refund history and leave {a} unchanged. Refund {b}; preserve its "
        "earlier specialist review. Escalate {d} for specialist review. Inspect the "
        "existing review on {c} without repeating it. For {e}, provide status only. "
        "Locate {f}; report it missing if necessary, without substituting another invoice.",
        "defer_roles": ("a",),
    },
    {
        "id": "generalization-v2-two-absences-and-three-customer-actions",
        "intent": "Two missing invoices are non-substitutable references while two actual "
        "refunds and a paid-invoice review proceed across three customers.",
        "roles": {
            "a": (None, (), None, 0),
            "b": (None, (), None, 1),
            "c": ("paid", (), "refund_invoice", 0),
            "d": ("paid", (), "refund_invoice", 1),
            "e": ("paid", (), "escalate_dispute", 2),
            "f": ("disputed", (), None, 2),
            "g": ("paid", ("escalations",), None, 3),
            "h": ("refunded", ("refunds",), None, 3),
        },
        "instruction": "Inspect {a}, {b}, {c}, {d}, {e}, {f}, {g}, and {h}. My earlier "
        "questions mentioned {a} and {b}; if those invoices cannot be found, report both "
        "absences and do not substitute another invoice. Independently, refund {c} "
        "for {customer0} and refund {d} for {customer1}. For {customer2}, escalate {e} "
        "for specialist review, but report status only for {f}. Inspect the existing "
        "review on {g} without repeating it or issuing a refund. Confirm the recorded "
        "refund on {h} without changing it.",
    },
    {
        "id": "generalization-v2-same-status-three-owner-action-grid",
        "intent": "Each payment status appears with different permissions; customer scope "
        "and existing history distinguish requested operations from status-only invoices.",
        "roles": {
            "a": ("paid", (), "refund_invoice", 0),
            "b": ("paid", (), "escalate_dispute", 1),
            "c": ("paid", (), None, 2),
            "d": ("disputed", (), "escalate_dispute", 0),
            "e": ("disputed", ("escalations",), None, 1),
            "f": ("disputed", (), None, 2),
        },
        "instruction": "Inspect {a}, {b}, {c}, {d}, {e}, and {f}. For {customer0}, "
        "refund {a}; separately, escalate {d} for specialist review. For {customer1}, "
        "escalate {b} for specialist review and inspect the existing case on {e} without "
        "repeating it. For {customer2}, {c} and {f} are status-only inquiries and must "
        "remain unchanged. Do not refund {b}, {d}, or {e}. Do not escalate {a}.",
    },
    {
        "id": "generalization-v2-unselected-cross-operation-pair",
        "intent": "An unresolved choice between a refund and a review grants neither; "
        "one independently named refund and two independently named reviews remain due.",
        "roles": {
            "a": ("paid", (), None, 0),
            "b": ("paid", (), None, 0),
            "c": ("paid", ("escalations",), "refund_invoice", 1),
            "d": ("paid", (), "escalate_dispute", 1),
            "e": ("disputed", (), "escalate_dispute", 2),
            "f": ("refunded", ("refunds",), None, 2),
            "g": (None, (), None, 2),
        },
        "instruction": "Inspect {a}, {b}, {c}, {d}, {e}, {f}, and {g}. I have not "
        "decided whether to refund {a} or request a specialist review for {b}; ask me "
        "to choose and leave both invoices unchanged. Independently, refund {c}. "
        "Escalate {d} and {e} for specialist review. Preserve the existing review on "
        "{c}. Confirm the already recorded refund on {f} without another write. "
        "If {g} is absent, report that absence without substituting any of the other invoices.",
        "defer_roles": ("a", "b"),
    },
)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def make_scenario(prototype: dict, variant: int) -> Scenario:
    if variant not in range(VARIANTS):
        raise ValueError("Variant outside frozen design")
    number = (
        100_000_000 + int(fingerprint([COHORT_ID, prototype["id"], variant])[:10], 16) % 800_000_000
    )
    ids = {r: f"inv-{number + i}" for i, r in enumerate(prototype["roles"])}
    owners = sorted({details[3] for details in prototype["roles"].values()})
    customers = {i: f"customer-{number + i}" for i in owners}
    state = {
        "customers": [{"id": c, "status": "active"} for c in customers.values()],
        "invoices": [],
        "refunds": [],
        "escalations": [],
    }
    invariants, allowed = [], []
    for r, (status, ledgers, operation, owner) in prototype["roles"].items():
        if status is None:
            continue
        invoice = ids[r]
        state["invoices"].append(
            {
                "id": invoice,
                "customer_id": customers[owner],
                "amount": ("88.65", "247.30")[variant],
                "currency": "USD",
                "status": status,
            }
        )
        for ledger in ledgers:
            state[ledger].append(
                {
                    "id": f"prior-{ledger}-{invoice}",
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
    ordered = list(ids.values())
    if variant:
        ordered.reverse()
        state["invoices"].reverse()
    return Scenario.model_validate(
        {
            "schema_version": "1",
            "id": f"{prototype['id']}-v{variant:02d}",
            "instruction": prototype["instruction"].format(
                **ids, **{f"customer{k}": v for k, v in customers.items()}
            ),
            "provenance": {"source": {"kind": "synthetic"}},
            "initial_state": state,
            "expected_invariants": invariants,
            "benchmark": {
                "category": "billing",
                "archetype": "multi-invoice",
                "difficulty": "compound",
                "mutation_policy": {"allowed_targets": allowed},
                "trajectory": {
                    "required_inspections": ordered,
                    "require_inspection_before_mutation": True,
                    "max_write_attempts_per_invoice": 1,
                },
                "generation": {
                    "generator": COHORT_ID,
                    "family_id": prototype["id"],
                    "prototype_id": prototype["id"],
                    "prototype_sha256": fingerprint(prototype),
                    "variant_index": variant,
                    "dimensions": {"presentation": f"reverse-{variant}"},
                },
            },
        }
    )


def scenarios() -> dict[str, Scenario]:
    return {
        task.id: task for p in PROTOTYPES for v in range(VARIANTS) for task in [make_scenario(p, v)]
    }


def novelty_audit(current: dict[str, Scenario]) -> dict:
    from experiments.decision_model.fresh_holdout import scenarios as fallback_fresh_scenarios
    from experiments.decision_model.generalization_holdout import scenarios as generalization_v1
    from experiments.decision_model.history_scenarios import scenarios as history_scenarios
    from experiments.decision_model.next_data import planned_tasks
    from rl.phase12.corpus import scenarios as phase12_scenarios

    sources = {
        f"phase12-{split}": list(tasks.values()) for split, tasks in phase12_scenarios().items()
    }
    sources["history-v2"] = list(history_scenarios().values())
    sources["fallback-fresh"] = list(fallback_fresh_scenarios().values())
    sources["generalization-v1"] = list(generalization_v1().values())
    sources["next-v1-all-splits"] = [entry[0] for entry in planned_tasks()]
    prior = {task.id: task for tasks in sources.values() for task in tasks}
    roots = {task.benchmark.generation.prototype_id for task in prior.values()}
    structures = {structural_signature(t) for t in prior.values()}
    instructions = {normalized_instruction(t.instruction) for t in prior.values()}
    for task in current.values():
        if (
            task.id in prior
            or task.benchmark.generation.prototype_id in roots
            or structural_signature(task) in structures
            or normalized_instruction(task.instruction) in instructions
        ):
            raise ValueError(f"Prior family, structure, or instruction overlap: {task.id}")
    if len({structural_signature(t) for t in current.values()}) != len(PROTOTYPES):
        raise ValueError("Fresh composition roots must have distinct behavioral structures")
    return {
        "source_task_counts": {k: len(v) for k, v in sources.items()},
        "unique_prior_tasks": len(prior),
        "unique_prior_roots": len(roots),
        "root_collisions": 0,
        "structural_collisions": 0,
        "normalized_instruction_collisions": 0,
        "unique_new_structures": len(PROTOTYPES),
        "method": "Compare all Phase12 splits, history-v2, next-v1, fallback-fresh and "
        "generalization-v1 roots, "
        "ID-normalized "
        "instructions and behavior signatures removing archetype/status redundancy.",
        "limitation": "Non-overlap under these hashes is not semantic independence.",
    }


def operations(task: Scenario) -> dict[str, str]:
    return {
        i.invoice_id: "refund_invoice" if i.type == "must_refund" else "escalate_dispute"
        for i in task.expected_invariants
        if i.type in {"must_refund", "must_escalate"}
    }


async def validate_expected_fixture(task: Scenario, directory: Path) -> dict:
    """Validate native verifier against authored endpoint data, with no agent/tool execution."""
    final_state = copy.deepcopy(task.initial_state.model_dump(mode="json"))
    calls = [
        {"name": "get_invoice", "arguments": {"invoice_id": invoice}}
        for invoice in task.benchmark.trajectory.required_inspections
    ]
    for invoice, operation in operations(task).items():
        ledger = "refunds" if operation == "refund_invoice" else "escalations"
        final_state[ledger].append(
            {
                "id": f"fixture-{invoice}",
                "invoice_id": invoice,
                "reason": "Offline author expectation fixture",
            }
        )
        for row in final_state["invoices"]:
            if row["id"] == invoice:
                row["status"] = "refunded" if ledger == "refunds" else "disputed"
        calls.append(
            {"name": operation, "arguments": {"invoice_id": invoice, "reason": "Offline fixture"}}
        )
    database = directory / "fixture.db"
    await initialize_database(database, InitialState.model_validate(final_state))
    write(directory / "scenario.json", task.model_dump(mode="json"))
    write(directory / "trajectory.json", {"tool_calls": calls})
    source = ROOT / "benchmarks/disputed-refund/tests/verify.py"
    verifier = types.ModuleType("fresh_holdout_fixture_verifier")
    verifier.__file__ = str(source)
    exec(compile(source.read_text(), str(source), "exec"), verifier.__dict__)
    verifier.CONTAINER_SCENARIO_PATH = directory / "scenario.json"
    verifier.AGENT_RESULT_PATH = directory / "trajectory.json"
    rewards, diagnostics = verifier.evaluate_database(database)
    if rewards.get("reward") != 1:
        raise ValueError(f"Authored expected-state fixture rejected: {task.id}: {diagnostics}")
    # Empty initial state must not pass when expected writes have not occurred.
    await initialize_database(database, task.initial_state)
    missing_rewards, _ = verifier.evaluate_database(database)
    if missing_rewards.get("reward") != 0:
        raise ValueError(f"Verifier accepted missing required writes: {task.id}")
    return {
        "task_id": task.id,
        "expected_endpoint_accepted": True,
        "missing_writes_rejected": True,
        "fixture_calls": len(calls),
        "fixture_only": True,
        "agent_or_policy_executed": False,
    }


async def freeze(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Frozen destination already exists: {output}")
    tasks = scenarios()
    novelty = novelty_audit(tasks)
    output.mkdir(parents=True)
    write(output / "prototype-specs.json", PROTOTYPES)
    metadata, validations = {}, []
    for task in tasks.values():
        if len(task.benchmark.trajectory.required_inspections) + len(operations(task)) > 16:
            raise ValueError("Authored minimal path exceeds the unchanged global tool budget")
        with tempfile.TemporaryDirectory(prefix="fresh-heldout-fixture-") as tmp:
            validations.append(await validate_expected_fixture(task, Path(tmp)))
        canonical = output / "canonical" / task.id
        await render_harbor_task(task, canonical)
        execution_path = output / "tasks" / task.id
        execution = derive_execution_task(canonical, execution_path)
        root = task.benchmark.generation.prototype_id
        prototype = next(p for p in PROTOTYPES if p["id"] == root)
        role_ids = {
            r: row.invoice_id
            for r, row in zip(
                [r for r, d in prototype["roles"].items() if d[0] is not None],
                [
                    i
                    for i in task.expected_invariants
                    if i.type in {"must_refund", "must_escalate", "unchanged"}
                ],
                strict=True,
            )
        }
        metadata[task.id] = {
            "task_id": task.id,
            "split": "test",
            "group_id": root,
            "family_id": root,
            "composition_root": root,
            "prototype_id": root,
            "source": "prospective-authored-unexposed-composition",
            "scenario_sha256": scenario_sha256(task),
            "structural_sha256": structural_signature(task),
            "required_inspections": task.benchmark.trajectory.required_inspections,
            "expected_operations": operations(task),
            "defer_targets": [role_ids[r] for r in prototype.get("defer_roles", ())],
            "author_intent": prototype["intent"],
            "task_path": str(execution_path.resolve()),
            "execution_task_sha256": execution.execution_task_sha256,
            "base_task_sha256": execution.base_task_sha256,
            "scenario_raw_sha256": digest(execution_path / "scenario.yaml"),
        }
    manifest = {
        "schema_version": "1",
        "cohort_id": COHORT_ID,
        "task_count": len(tasks),
        "prototype_count": len(PROTOTYPES),
        "variants_per_prototype": VARIANTS,
        "split_task_counts": {"test": len(tasks)},
        "split_root_counts": {"test": len(PROTOTYPES)},
        "tasks": metadata,
        "novelty_audit": novelty,
        "prototype_specs_sha256": fingerprint(PROTOTYPES),
        "generator_sha256": digest(Path(__file__)),
        "cohort_previously_exposed": False,
        "pre_freeze_policy_rollouts": 0,
        "pre_freeze_authorization_parser_checks": 0,
        "pre_freeze_teacher_collections": 0,
        "pre_freeze_model_inferences": 0,
        "expected_operation_counts": dict(
            Counter(op for t in tasks.values() for op in operations(t).values())
        ),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                Path(__file__),
                ROOT / "experiments/decision_model/history_scenarios.py",
                ROOT / "experiments/decision_model/next_data.py",
                ROOT / "experiments/decision_model/fresh_holdout.py",
                ROOT / "experiments/decision_model/generalization_holdout.py",
                ROOT / "benchmarks/disputed-refund/tests/verify.py",
                ROOT / "rl/phase12/corpus.py",
                ROOT / "benchmarks/billing/phase12/manifest.json",
                ROOT / "benchmarks/billing/phase12/prototypes.json",
                ROOT / "src/agentops_demo/benchmark/phase10_variants.py",
            ]
        },
        "limitations": [
            "Prospectively authored independently of ongoing boundary repairs; same billing "
            "primitives and renderer. The canonical evaluation freeze follows runtime/regression "
            "gates. No requests were changed based on policy/model performance.",
            "New structures and instructions, not an independent externally sourced benchmark.",
            "Two variants per root are presentation repetitions, not independent families.",
            "Holdout was not used in training, calibration, policy preflight or outcome selection.",
            "Natural request variations were authored without querying the authorization parser; "
            "unsupported grammar is allowed to fail in the fixed evaluated system.",
            "Fixture validation checks authored endpoint/verifier consistency; "
            "it is not a rollout.",
            "Harness supplies target IDs; unrestricted entity discovery is not tested.",
            "Clarification text and explanation quality are not scored by the native verifier.",
        ],
    }
    manifest["cohort_sha256"] = fingerprint(manifest)
    write(output / "cohort-manifest.json", manifest)
    write(output / "fixture-validation.json", validations)
    (output / "test-selection.jsonl").write_text(
        "".join(
            json.dumps({"task_id": k, "group_id": v["group_id"], "split": "test"}) + "\n"
            for k, v in metadata.items()
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = asyncio.run(freeze(args.output))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"tasks", "source_sha256", "limitations"}},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
