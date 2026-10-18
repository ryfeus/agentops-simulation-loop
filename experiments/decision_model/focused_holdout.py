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
COHORT_ID = "billing-focused-test-v1"
DEFAULT_OUTPUT = ROOT / ".artifacts/decision-model-focused/holdout"
VARIANTS = 2
# (initial status, existing ledger kinds, required NEW operation, customer index)
PROTOTYPES = (
    {
        "id": "focused-test-parallel-refunds-with-review-preservation",
        "intent": "Two independent first refunds, including an already reviewed paid invoice; "
        "a paid invoice receives a review, an existing disputed review is preserved, "
        "and open and disputed status-only invoices remain unchanged.",
        "roles": {
            "a": ("paid", (), "refund_invoice", 0),
            "b": ("paid", ("escalations",), "refund_invoice", 1),
            "c": ("paid", (), "escalate_dispute", 0),
            "d": ("disputed", ("escalations",), None, 1),
            "e": ("open", (), None, 0),
            "f": ("disputed", (), None, 1),
        },
        "instruction": "Inspect {a}, {b}, {c}, {d}, {e}, and {f}. Refund {a}. Refund {b}. "
        "Escalate {c} for specialist review. Preserve the existing review on {d}. "
        "Do not refund {d}. For {e} and {f}, report status only. Leave {e} and {f} unchanged.",
    },
    {
        "id": "focused-test-review-pair-with-three-preserved-states",
        "intent": "A paid and a disputed invoice need first reviews. A paid existing review, "
        "a paid status-only invoice, a refunded invoice and an open invoice stay unchanged.",
        "roles": {
            "a": ("paid", (), "escalate_dispute", 0),
            "b": ("disputed", (), "escalate_dispute", 1),
            "c": ("paid", ("escalations",), None, 1),
            "d": ("paid", (), None, 0),
            "e": ("refunded", ("refunds",), None, 0),
            "f": ("open", (), None, 1),
        },
        "instruction": "Inspect {f}, {c}, {a}, {e}, {b}, and {d}. Escalate {a} and {b} "
        "for specialist review. Check the existing review on {c} without creating another "
        "review. Do not refund {c}. For {d}, {e}, and {f}, report status only. "
        "Do not refund or escalate {d}, {e}, or {f}.",
    },
    {
        "id": "focused-test-withdrawal-and-cross-operation-refund",
        "intent": "A withdrawn refund remains blocked while another reviewed paid invoice "
        "receives its first refund and a dispute receives review. Existing reviews and "
        "two unrelated status-only invoices are preserved.",
        "roles": {
            "a": ("paid", (), None, 0),
            "b": ("paid", ("escalations",), "refund_invoice", 0),
            "c": ("disputed", (), "escalate_dispute", 1),
            "d": ("disputed", ("escalations",), None, 1),
            "e": ("paid", (), None, 1),
            "f": ("open", (), None, 0),
        },
        "instruction": "Inspect {a}, {b}, {c}, {d}, {e}, and {f}. Refund {a}. "
        "I withdraw permission to refund {a}. Leave {a} unchanged. Refund {b}. "
        "Preserve the existing review on {b}. Escalate {c} for specialist review. "
        "Preserve the existing review on {d}. Do not refund {d}. "
        "For {e} and {f}, report status only; leave {e} and {f} unchanged.",
    },
    {
        "id": "focused-test-absent-invoice-with-paid-write-pair",
        "intent": "An absent invoice is reported without substitution. Separate paid "
        "invoices receive a refund and review; an open invoice, existing disputed review "
        "and completed refund remain unchanged.",
        "roles": {
            "a": (None, (), None, 0),
            "b": ("paid", (), "refund_invoice", 1),
            "c": ("paid", (), "escalate_dispute", 0),
            "d": ("open", (), None, 0),
            "e": ("disputed", ("escalations",), None, 1),
            "f": ("refunded", ("refunds",), None, 1),
        },
        "instruction": "Inspect {a}, {b}, {c}, {d}, {e}, and {f}. For {a}, report "
        "whether the invoice exists. Do not refund or escalate {a}. Refund {b}. "
        "Escalate {c} for specialist review. For {d} and {f}, report status only; "
        "leave {d} and {f} unchanged. Preserve the existing review on {e}. Do not refund {e}.",
    },
    {
        "id": "focused-test-unresolved-choice-with-independent-work",
        "intent": "Two paid invoices await a customer choice and cannot be mutated. "
        "An independent dispute review and paid refund proceed. A paid existing review "
        "and an open status-only invoice stay unchanged.",
        "roles": {
            "a": ("paid", (), None, 0),
            "b": ("paid", (), None, 0),
            "c": ("disputed", (), "escalate_dispute", 1),
            "d": ("paid", ("escalations",), None, 1),
            "e": ("paid", (), "refund_invoice", 1),
            "f": ("open", (), None, 0),
        },
        "instruction": "Inspect {a}, {b}, {c}, {d}, {e}, and {f}. I have not decided "
        "whether {a} or {b} is the invoice I want refunded. Ask me to clarify that choice; "
        "change neither. Escalate {c} for specialist review. Refund {e}. Preserve the "
        "existing review on {d}. Do not refund {d}. For {f}, report status only; "
        "leave {f} unchanged.",
        "defer_roles": ("a", "b"),
    },
    {
        "id": "focused-test-three-active-operations-with-ledger-preservation",
        "intent": "One first refund and two first reviews are independently authorized. "
        "An open invoice, a refunded invoice with both ledgers, and a paid existing "
        "review remain unchanged.",
        "roles": {
            "a": ("paid", (), "refund_invoice", 1),
            "b": ("paid", (), "escalate_dispute", 0),
            "c": ("disputed", (), "escalate_dispute", 1),
            "d": ("open", (), None, 0),
            "e": ("refunded", ("refunds", "escalations"), None, 1),
            "f": ("paid", ("escalations",), None, 0),
        },
        "instruction": "Inspect {e}, {a}, {f}, {b}, {d}, and {c}. Refund {a}. "
        "Escalate {b} for specialist review. Escalate {c} for specialist review. "
        "For {d} and {e}, report status only; leave {d} and {e} unchanged. "
        "Inspect the existing review on {f} without repeating it. Do not refund {f}.",
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
    from agentops_demo.validation.scenario import load_scenario
    from experiments.decision_model.fresh_holdout import scenarios as fallback_fresh_scenarios
    from experiments.decision_model.generalization_holdout import scenarios as generalization_v1
    from experiments.decision_model.generalization_v2_holdout import scenarios as generalization_v2
    from experiments.decision_model.generalization_v3_holdout import scenarios as generalization_v3
    from experiments.decision_model.generalization_v4_holdout import scenarios as generalization_v4
    from experiments.decision_model.history_scenarios import scenarios as history_scenarios
    from rl.phase12.corpus import scenarios as phase12_scenarios

    sources = {
        f"phase12-{split}": list(tasks.values()) for split, tasks in phase12_scenarios().items()
    }
    sources["history-v2"] = list(history_scenarios().values())
    sources["fallback-fresh"] = list(fallback_fresh_scenarios().values())
    sources["generalization-v1"] = list(generalization_v1().values())
    sources["generalization-v2"] = list(generalization_v2().values())
    sources["generalization-v3"] = list(generalization_v3().values())
    sources["generalization-v4"] = list(generalization_v4().values())
    # Read frozen scenarios directly: next_data imports runtime/teacher helpers.
    manifest_path = ROOT / ".artifacts/decision-model/next-v1/corpus/cohort-manifest.json"
    next_manifest = json.loads(manifest_path.read_text())
    sources["next-v1-all-splits"] = []
    for meta in next_manifest["tasks"].values():
        scenario_path = Path(meta["task_path"]) / "scenario.yaml"
        scenario = load_scenario(scenario_path)
        if (
            scenario_sha256(scenario) != meta["scenario_sha256"]
            or digest(scenario_path) != meta["scenario_raw_sha256"]
        ):
            raise ValueError(f"Frozen prior scenario changed: {scenario_path}")
        sources["next-v1-all-splits"].append(scenario)
    focused_manifest = ROOT / ".artifacts/decision-model-focused/corpus/cohort-manifest.json"
    focused = json.loads(focused_manifest.read_text())
    if set(focused["split_task_counts"]) != {"train", "dev", "calibration"}:
        raise ValueError("Focused training corpus must exclude test")
    for meta in focused["tasks"].values():
        scenario_path = Path(meta["task_path"]) / "scenario.yaml"
        task = load_scenario(scenario_path)
        if (
            scenario_sha256(task) != meta["scenario_sha256"]
            or digest(scenario_path) != meta["scenario_raw_sha256"]
        ):
            raise ValueError(f"Frozen focused training scenario changed: {scenario_path}")
        sources.setdefault("focused-" + meta["split"], []).append(task)
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
        "generalization-v1, generalization-v2, generalization-v3 and generalization-v4 roots, "
        "plus focused train/dev/calibration, ID-normalized "
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
        if len(task.benchmark.trajectory.required_inspections) + len(operations(task)) > 11:
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
                ROOT / ".artifacts/decision-model-focused/corpus/cohort-manifest.json",
                ROOT / ".artifacts/decision-model/next-v1/corpus/cohort-manifest.json",
                ROOT / "experiments/decision_model/fresh_holdout.py",
                ROOT / "experiments/decision_model/generalization_holdout.py",
                ROOT / "experiments/decision_model/generalization_v2_holdout.py",
                ROOT / "experiments/decision_model/generalization_v3_holdout.py",
                ROOT / "experiments/decision_model/generalization_v4_holdout.py",
                ROOT / ".artifacts/decision-model-generalization-v3/corpus/cohort-manifest.json",
                ROOT / "benchmarks/disputed-refund/tests/verify.py",
                ROOT / "rl/phase12/corpus.py",
                ROOT / "benchmarks/billing/phase12/manifest.json",
                ROOT / "benchmarks/billing/phase12/prototypes.json",
                ROOT / "src/agentops_demo/benchmark/phase10_variants.py",
            ]
        },
        "limitations": [
            "Prospectively authored without parser, policy, teacher or model execution; "
            "same billing "
            "primitives and renderer. The canonical evaluation freeze follows runtime/regression "
            "gates. No requests were changed based on policy/model performance.",
            "New structures and instructions, not an independent externally sourced benchmark.",
            "Two variants per root are presentation repetitions, not independent families.",
            "Holdout was not used in training, calibration, policy preflight or outcome selection.",
            "Focused requests use simple supported vocabulary selected by static source "
            "inspection, "
            "without querying authorization, model, policy or service outcomes.",
            "Six-invoice test compositions are separate from three-invoice training, "
            "four-invoice development and five-invoice calibration compositions.",
            "Unavailable refund-history conditions and complex parser challenges "
            "are outside scope.",
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
