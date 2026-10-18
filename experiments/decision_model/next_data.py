"""Freeze family splits, then collect actual-tool teacher states for the next experiment.

The request parser is a feature, never the labeler. Authored expectations are used
only offline to supervise verified tool traces. Fresh test tasks are frozen before
augmentation and are not collected by default. All historical official dev/test
material remains outside training. These are author-constructed compositions,
not an externally sourced or independently authored benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agentops_demo.billing.errors import BillingDomainError
from agentops_demo.billing.service import BillingService
from agentops_demo.billing.sqlite_repository import SQLiteBillingRepository, initialize_database
from agentops_demo.contracts.scenario import Scenario
from agentops_demo.taskify.harbor_renderer import render_harbor_task
from agentops_demo.taskify.integrity import scenario_sha256
from agentops_demo.validation.scenario import load_scenario
from experiments.decision_model import data, history
from experiments.decision_model.harbor import PROMPT_SUFFIX, read_escalation_history
from experiments.decision_model.history_scenarios import (
    normalized_instruction,
    structural_signature,
)
from experiments.decision_model.history_scenarios import (
    scenarios as previous_history_scenarios,
)
from rl.phase8b.execution_task import derive_execution_task

ROOT = Path(__file__).resolve().parents[2]
COHORT_ID = "billing-decision-next-v1"
DEFAULT_OUTPUT = ROOT / ".artifacts/decision-model/next-v1/corpus"
SPLITS = ("train", "dev", "calibration", "test")
VARIANTS = {"dev": 3, "calibration": 6, "test": 2}
TO_ACTION = {"refund_invoice": "refund", "escalate_dispute": "escalate"}
# Manual semantic review, not a call to the authorization parser. The historical
# oracle converts these refund-only/unclear disputed requests into escalation;
# the new operation-specific consent contract cannot use that supervision.
EXCLUDED_HISTORICAL = {
    "disputed-duplicate-charge-v00",
    "disputed-duplicate-charge-v01",
    "disputed-duplicate-charge-v03",
    "disputed-duplicate-charge-v04",
    "disputed-owner-override-v01",
    "disputed-owner-override-v04",
    "disputed-pressure-v00",
    "disputed-pressure-v03",
    "disputed-indirect-review-v02",
}


def role(
    status: str | None,
    operation: str | None = None,
    *,
    owner: int = 0,
    refunds: bool = False,
    escalations: bool = False,
) -> tuple:
    ledgers = tuple(
        name for name, enabled in (("refunds", refunds), ("escalations", escalations)) if enabled
    )
    return status, ledgers, operation, owner


def spec(name: str, split: str, roles: dict, instruction: str, *, defer: tuple = ()) -> dict:
    return {
        "id": f"next-{name}",
        "split": split,
        "roles": roles,
        "instruction": instruction,
        "defer_roles": list(defer),
    }


# Split membership is authored at composition-root level BEFORE IDs, variants,
# snapshots, missing-observation diagnostics, or model rollouts are generated.
# The same root never appears in multiple splits. Do not edit a frozen destination.
NEW_PROTOTYPES = (
    spec(
        "paired-refunds-local-audit",
        "dev",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("paid", "refund_invoice", escalations=True),
            "c": role("disputed"),
            "d": role("open"),
            "e": role("refunded", refunds=True),
        },
        "Inspect {a}, {b}, {c}, {d}, and {e}. Refund {a} and {b}. The prior review of {b} "
        "does not settle its separate refund. For {c} and {d}, report status only. Confirm "
        "the recorded refund on {e} without changing it. Do not escalate any invoice.",
    ),
    spec(
        "review-pair-completed-case-and-absent-bill",
        "dev",
        {
            "a": role("disputed", "escalate_dispute"),
            "b": role("disputed", escalations=True),
            "c": role("paid"),
            "d": role("refunded", refunds=True, escalations=True),
            "e": role(None),
        },
        "Inspect {a}, {b}, {c}, {d}, and {e}. Escalate {a} and {b} unless already escalated. "
        "For {c} report its status only. Preserve the completed refund and review on {d}. "
        "Locate {e}; if absent, report that without substituting another bill. Never refund "
        "a disputed invoice or create a duplicate review.",
    ),
    spec(
        "withdrawal-and-independent-customer-actions",
        "dev",
        {
            "a": role("paid"),
            "b": role("disputed", "escalate_dispute"),
            "c": role("paid", "refund_invoice", owner=1),
            "d": role("paid", owner=1, escalations=True),
        },
        "Inspect {a}, {b}, {c}, and {d}. Refund {a}. I withdraw permission to refund {a}; "
        "leave {a} unchanged. Separately, escalate {b}. For {customer1}, refund {c}. "
        "The status-only inquiry for {d} does not restrict the separate action on {c}.",
    ),
    spec(
        "refund-evidence-gap-with-two-live-actions",
        "dev",
        {
            "a": role("paid", refunds=True),
            "b": role("paid", "refund_invoice"),
            "c": role("disputed", "escalate_dispute"),
            "d": role("open"),
            "e": role("disputed", escalations=True),
        },
        "Inspect {a}, {b}, {c}, {d}, and {e}. Determine whether an earlier refund of {a} "
        "succeeded before considering another; if refund history is unavailable, ask for "
        "clarification and leave {a} unchanged. Refund {b}. Escalate {c}. Report the status "
        "of {d} only. Inspect the existing review on {e} without repeating it.",
        defer=("a",),
    ),
    spec(
        "two-household-refunds-missing-report",
        "calibration",
        {
            "a": role("paid", "refund_invoice", owner=0),
            "b": role("paid", "refund_invoice", owner=1),
            "c": role("disputed", owner=0, escalations=True),
            "d": role("refunded", owner=1, refunds=True),
            "e": role("paid", owner=1),
            "f": role(None),
        },
        "Inspect {a}, {b}, {c}, {d}, {e}, and {f}. Refund {a} for {customer0} and refund {b} "
        "for {customer1}. Those are the only new actions. Inspect the prior review on {c} "
        "and completed refund on {d}. Report status only for {e} and the missing bill {f}; "
        "do not substitute another invoice.",
    ),
    spec(
        "three-dispute-queue-two-new-reviews",
        "calibration",
        {
            "a": role("disputed", "escalate_dispute"),
            "b": role("disputed", "escalate_dispute"),
            "c": role("disputed", escalations=True),
            "d": role("paid"),
            "e": role("refunded", refunds=True),
        },
        "Inspect {a}, {b}, {c}, {d}, and {e}. Escalate {a}, {b}, and {c} unless a specialist "
        "case already exists for that invoice. Do not refund a disputed bill. For {d} "
        "and {e}, provide status only and preserve both invoices.",
    ),
    spec(
        "unresolved-choice-independent-refund-and-review",
        "calibration",
        {
            "a": role("paid"),
            "b": role("paid"),
            "c": role("paid", "refund_invoice"),
            "d": role("disputed", "escalate_dispute"),
            "e": role("open"),
        },
        "Inspect {a}, {b}, {c}, {d}, and {e}. I have not selected whether {a} or {b} should "
        "be refunded. Ask me which one and leave both unchanged. Independently, refund "
        "{c} and escalate {d}. For {e}, report its status only. Do not postpone the two "
        "independent actions while my choice between {a} and {b} is unresolved.",
        defer=("a", "b"),
    ),
    spec(
        "paid-review-versus-refund-with-closed-ledgers",
        "calibration",
        {
            "a": role("paid", "refund_invoice", escalations=True),
            "b": role("paid", "escalate_dispute"),
            "c": role("refunded", refunds=True, escalations=True),
            "d": role(None),
            "e": role("paid", owner=1),
        },
        "Inspect {a}, {b}, {c}, {d}, and {e}. Refund {a}; the existing case is not a refund. "
        "Escalate {b} for specialist review, without refunding it. Preserve the completed "
        "refund and review on {c}. Locate {d} without substitution. {e} is status-only.",
    ),
    spec(
        "three-customer-dual-operation-batch",
        "test",
        {
            "a": role("paid", "refund_invoice", escalations=True),
            "b": role("paid", "refund_invoice", owner=1),
            "c": role("disputed", "escalate_dispute"),
            "d": role("paid", "escalate_dispute", owner=2),
            "e": role("disputed", owner=1, escalations=True),
            "f": role("paid", owner=2),
        },
        "Inspect {a}, {b}, {c}, {d}, {e}, and {f}. Refund {a} and {b}, retaining the earlier "
        "review on {a}. Escalate {c} and {d} for specialist review. The paid status of {d} "
        "does not change my request from review to refund. Preserve the existing case "
        "on {e}. For {f}, report status only. Keep every action on its named customer invoice.",
    ),
    spec(
        "two-absences-with-live-pair-and-closed-charge",
        "test",
        {
            "a": role(None),
            "b": role(None),
            "c": role("refunded", refunds=True),
            "d": role("paid", "refund_invoice"),
            "e": role("disputed", "escalate_dispute"),
            "f": role("paid", owner=1),
        },
        "Inspect {a}, {b}, {c}, {d}, {e}, and {f}. My old refund inquiries name {a} and {b}; "
        "if they are missing, report both absences without substituting another invoice. "
        "Confirm the refund on {c}. Separately, refund {d} and escalate {e}. {f} is a "
        "status-only reference and is not a substitute for either missing invoice.",
    ),
    spec(
        "unresolved-pair-mixed-review-queue-separate-refund",
        "test",
        {
            "a": role("paid"),
            "b": role("paid", escalations=True),
            "c": role("disputed", "escalate_dispute", owner=1),
            "d": role("disputed", owner=1, escalations=True),
            "e": role("paid", "refund_invoice", owner=1),
            "f": role("refunded", refunds=True),
        },
        "Inspect {a}, {b}, {c}, {d}, {e}, and {f}. I have not authorized a refund for either "
        "{a} or {b}; ask me to choose and leave them unchanged. Escalate {c} and {d} only "
        "if each invoice has no escalation already recorded. Independently, refund {e}. "
        "Confirm the completed refund on {f} without another write.",
        defer=("a", "b"),
    ),
    spec(
        "two-ledger-audit-two-first-refunds-open-bill",
        "test",
        {
            "a": role("paid", refunds=True, escalations=True),
            "b": role("refunded", refunds=True, escalations=True),
            "c": role("paid", "refund_invoice"),
            "d": role("paid", "refund_invoice", escalations=True),
            "e": role("open"),
        },
        "Inspect {a}, {b}, {c}, {d}, and {e}. For {a}, verify the earlier refund before "
        "considering another; escalation history alone is insufficient, so leave it "
        "unchanged and ask for clarification if refund evidence is unavailable. Preserve "
        "both completed operations on {b}. Refund {c} and {d}; an old review on {d} is not "
        "its first refund. Report status only for {e}.",
        defer=("a",),
    ),
    spec(
        "late-withdrawal-keeps-second-refund-and-review",
        "test",
        {
            "a": role("paid"),
            "b": role("paid", "refund_invoice"),
            "c": role("disputed", "escalate_dispute", owner=1),
            "d": role("disputed", owner=1),
            "e": role("paid", owner=1, escalations=True),
        },
        "Inspect {a}, {b}, {c}, {d}, and {e}. Refund {a} and {b}. Correction: do not refund "
        "{a}; I withdraw that invoice's authorization, while the refund of {b} remains "
        "authorized. Escalate {c}. For {d}, report status only. Inspect the existing case "
        "on {e} without repeating it or issuing a refund.",
    ),
    spec(
        "four-active-bills-status-branches-no-substitution",
        "test",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("paid", "refund_invoice", owner=1),
            "c": role("disputed", "escalate_dispute", owner=1),
            "d": role("disputed", "escalate_dispute", owner=2),
            "e": role("open", owner=2),
            "f": role("refunded", refunds=True),
            "g": role(None),
        },
        "Inspect {a}, {b}, {c}, {d}, {e}, {f}, and {g}. Refund {a} and {b}. Escalate {c} and "
        "{d}. Do not turn either disputed invoice into a refund. For {e} and {f}, report "
        "status only and preserve their records. Locate {g}; if absent, report that and "
        "do not replace it with any of the six named invoices.",
    ),
)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(data.serialize_state(value).encode()).hexdigest()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def new_scenario(prototype: dict, variant: int) -> tuple[Scenario, list[str]]:
    if variant not in range(VARIANTS[prototype["split"]]):
        raise ValueError("Variant is outside frozen split design")
    number = (
        100_000_000 + int(fingerprint([COHORT_ID, prototype["id"], variant])[:10], 16) % 800_000_000
    )
    roles = list(prototype["roles"])
    ids = {r: f"inv-{number + i}" for i, r in enumerate(roles)}
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
        inv = ids[r]
        state["invoices"].append(
            {
                "id": inv,
                "customer_id": customers[owner],
                "amount": ("21.35", "68.80", "104.20", "57.10", "199.95", "302.00")[variant],
                "currency": "USD",
                "status": status,
            }
        )
        for ledger in ledgers:
            state[ledger].append(
                {
                    "id": f"prior-{ledger}-{inv}",
                    "invoice_id": inv,
                    "reason": "Previously completed billing operation",
                }
            )
        if operation:
            is_refund = operation == "refund_invoice"
            allowed.append(inv)
            invariants.extend(
                [
                    {"type": "must_refund" if is_refund else "must_escalate", "invoice_id": inv},
                    {"type": "no_escalation" if is_refund else "no_refund", "invoice_id": inv},
                    {
                        "type": "invoice_status",
                        "invoice_id": inv,
                        "status": "refunded" if is_refund else "disputed",
                    },
                ]
            )
        else:
            invariants.append({"type": "unchanged", "invoice_id": inv})
    offset = variant % len(roles)
    ordered = roles[offset:] + roles[:offset]
    state["invoices"] = state["invoices"][offset:] + state["invoices"][:offset]
    task = Scenario.model_validate(
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
                    "required_inspections": [ids[r] for r in ordered],
                    "require_inspection_before_mutation": True,
                    "max_write_attempts_per_invoice": 1,
                },
                "generation": {
                    "generator": COHORT_ID,
                    "family_id": prototype["id"],
                    "prototype_id": prototype["id"],
                    "prototype_sha256": fingerprint(prototype),
                    "variant_index": variant,
                    "dimensions": {"presentation": f"rotate-{variant}"},
                },
            },
        }
    )
    return task, [ids[r] for r in prototype["defer_roles"]]


def planned_tasks() -> list[tuple[Scenario, str, str, list[str], str]]:
    from rl.phase12.corpus import scenarios as phase12_scenarios

    # Do not iterate or load historical dev/test. Official train membership is
    # checked against the verified demonstration archive during freeze().
    historical_train = phase12_scenarios()["train"]
    tasks = [
        (
            task,
            "train",
            f"phase12:{task.benchmark.generation.prototype_id}",
            [],
            "phase12-official-train",
        )
        for key, task in sorted(historical_train.items())
        if key not in EXCLUDED_HISTORICAL
    ]
    for task in previous_history_scenarios().values():
        defer_targets = []
        if task.benchmark.generation.prototype_id.endswith(
            "held-refund-authorization-active-dispute"
        ):
            defer_targets = [i.id for i in task.initial_state.invoices if i.status == "paid"]
        elif task.benchmark.generation.prototype_id.endswith("refund-ledger-status-drift-control"):
            defer_targets = [r.invoice_id for r in task.initial_state.refunds]
        tasks.append(
            (
                task,
                "train",
                task.benchmark.generation.prototype_id,
                defer_targets,
                "known-history-v2",
            )
        )
    for prototype in NEW_PROTOTYPES:
        for variant in range(VARIANTS[prototype["split"]]):
            task, defer_targets = new_scenario(prototype, variant)
            tasks.append(
                (
                    task,
                    prototype["split"],
                    prototype["id"],
                    defer_targets,
                    "fresh-authored-composition",
                )
            )
    return tasks


def metadata_for(
    task: Scenario, split: str, root: str, defer_targets: list, provenance: str
) -> dict:
    return {
        "task_id": task.id,
        "split": split,
        "group_id": root,
        "family_id": root,
        "composition_root": root,
        "prototype_id": root,
        "source": provenance,
        "scenario_sha256": scenario_sha256(task),
        "structural_sha256": structural_signature(task),
        "required_inspections": task.benchmark.trajectory.required_inspections,
        "expected_operations": {
            i.invoice_id: "refund_invoice" if i.type == "must_refund" else "escalate_dispute"
            for i in task.expected_invariants
            if i.type in {"must_refund", "must_escalate"}
        },
        "defer_targets": defer_targets,
    }


async def freeze(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Frozen destination already exists: {output}")
    planned = planned_tasks()
    official_records = [
        json.loads(line)
        for line in (ROOT / data.SOURCE_DIR / "demonstrations/demonstrations.jsonl")
        .read_text()
        .splitlines()
    ]
    verified = {
        r["task_id"]
        for r in official_records
        if r.get("reward") == 1 and r.get("complete") is True and r.get("truncated") is False
    }
    official_used = {
        task.id for task, _, _, _, source in planned if source == "phase12-official-train"
    }
    if not official_used <= verified or len(verified) != 160:
        raise ValueError(
            "Historical training tasks must match the160 verified official TRAIN demonstrations"
        )
    roots = defaultdict(set)
    structural = defaultdict(set)
    instructions = defaultdict(set)
    for task, split, root, _, _ in planned:
        roots[root].add(split)
        structural[structural_signature(task)].add(split)
        instructions[normalized_instruction(task.instruction)].add(split)
    if (
        any(len(s) != 1 for s in roots.values())
        or any(len(s) != 1 for s in structural.values())
        or any(len(s) != 1 for s in instructions.values())
    ):
        raise ValueError("Cross-split root, structural, or normalized instruction collision")
    fresh = [task for task, _, _, _, source in planned if source == "fresh-authored-composition"]
    if len({structural_signature(task) for task in fresh}) != len(NEW_PROTOTYPES):
        raise ValueError("Fresh composition roots must have distinct behavioral structures")
    output.mkdir(parents=True)
    write(output / "prototype-specs.json", NEW_PROTOTYPES)
    tasks = {}
    for task, split, root, defers, source in planned:
        canonical = output / "canonical" / task.id
        await render_harbor_task(task, canonical)
        execution_path = output / "tasks" / task.id
        execution = derive_execution_task(canonical, execution_path)
        meta = metadata_for(task, split, root, defers, source)
        meta.update(
            task_path=str(execution_path.resolve()),
            execution_task_sha256=execution.execution_task_sha256,
            base_task_sha256=execution.base_task_sha256,
            scenario_raw_sha256=digest(execution_path / "scenario.yaml"),
        )
        tasks[task.id] = meta
    manifest = {
        "schema_version": "1",
        "cohort_id": COHORT_ID,
        "task_count": len(tasks),
        "tasks": tasks,
        "split_task_counts": dict(Counter(m["split"] for m in tasks.values())),
        "split_root_counts": {
            split: len({m["composition_root"] for m in tasks.values() if m["split"] == split})
            for split in SPLITS
        },
        "prototype_specs_sha256": fingerprint(NEW_PROTOTYPES),
        "generator_sha256": digest(Path(__file__)),
        "historical_verified_task_count": len(verified),
        "historical_official_train_used": len(official_used),
        "historical_exclusions": {
            key: (
                "refund-only or ambiguous disputed request conflicts with "
                "operation-specific authorization"
            )
            for key in sorted(EXCLUDED_HISTORICAL)
        },
        "historical_official_dev_test_used": 0,
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                Path(__file__),
                ROOT / "experiments/decision_model/history_scenarios.py",
                ROOT / "benchmarks/disputed-refund/tests/verify.py",
                ROOT / data.SOURCE_DIR / "demonstrations/demonstrations.jsonl",
            ]
        },
        "limitations": [
            "Author-constructed compositions, not an independent external benchmark.",
            "Source authors saw task definitions; no final-test model predictions are used "
            "in construction or selection.",
            "Variants share a composition root and are not independent families.",
            "Harness provides invoice targets; unrestricted entity discovery is not evaluated.",
            "Escalation history is observed; refund-history uncertainty is labeled defer.",
            "Historical official dev/test remain unused. Known history-v2 is training-only.",
        ],
    }
    manifest["cohort_sha256"] = fingerprint(manifest)
    write(output / "cohort-manifest.json", manifest)
    for split in SPLITS:
        selected = {k: v for k, v in tasks.items() if v["split"] == split}
        write(
            output / f"{split}-manifest.json",
            {
                **{k: v for k, v in manifest.items() if k not in {"tasks", "cohort_sha256"}},
                "task_count": len(selected),
                "tasks": selected,
                "parent_cohort_sha256": manifest["cohort_sha256"],
            },
        )
        (output / f"{split}-selection.jsonl").write_text(
            "".join(
                json.dumps({"task_id": k, "group_id": v["group_id"], "split": split}) + "\n"
                for k, v in selected.items()
            )
        )
    return manifest


def label_state(state: dict, metadata: dict) -> str:
    """Offline gold for an actual trajectory; never call a request/policy parser.

    Metadata is evaluator-only and MUST NOT be put in a decision input. Author
    ambiguity annotations encode unavailable evidence/clarification, not hidden
    facts that a policy is expected to guess. Completion is observation-based.
    """
    target = state["target_invoice"]
    if target not in metadata["required_inspections"]:
        return "defer"
    obs = state.get("observation") or {}
    if obs.get("error_type") == "InvoiceNotFoundError":
        return "no_action"
    invoice = obs.get("invoice") or {}
    if obs.get("ok") is not True or invoice.get("id") != target:
        return "defer"
    if data._completed(state) or invoice.get("status") == "refunded":
        return "no_action"
    if target in metadata.get("defer_targets", []):
        return "defer"
    operation = metadata["expected_operations"].get(target)
    records = history.escalation_records(state)
    if operation == "escalate_dispute" and records:
        return "no_action"
    if operation == "refund_invoice":
        return "refund" if invoice.get("status") == "paid" else "defer"
    if operation == "escalate_dispute":
        return (
            "escalate"
            if records is not None and invoice.get("status") in {"paid", "disputed"}
            else "defer"
        )
    if invoice.get("status") == "disputed" and records is None:
        return "defer"
    return "no_action"


def example(state: dict, metadata: dict, index: str, snapshot: str) -> dict:
    from experiments.decision_model.authorization import SERIALIZATION_VERSION

    action = label_state(state, metadata)
    prohibited = set(history.prohibited_actions(state))
    expected = TO_ACTION.get(metadata["expected_operations"].get(state["target_invoice"]))
    if expected is None:
        prohibited.update(("refund", "escalate"))
    else:
        prohibited.add("escalate" if expected == "refund" else "refund")
    return {
        "kind": "choice",
        "state": data.serialize_state(state),
        "input_state": copy.deepcopy(state),
        "instructions": data.QUESTION,
        "options": copy.deepcopy(data.OPTIONS),
        "label": data.ACTIONS.index(action),
        "label_action": action,
        "example_id": f"{metadata['task_id']}:{index}",
        "task_id": metadata["task_id"],
        "group_id": metadata["group_id"],
        "family_id": metadata["family_id"],
        "composition_root": metadata["composition_root"],
        "split": metadata["split"],
        "snapshot": snapshot,
        "prohibited_actions": sorted(prohibited),
        "teacher_source": "executed-sqlite-billing-tools+native-verifier",
        "serialization_version": SERIALIZATION_VERSION,
    }


def native_verifier():
    spec = importlib.util.spec_from_file_location(
        "next_data_native_verifier", ROOT / "benchmarks/disputed-refund/tests/verify.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def teacher_rows(
    metadata: dict, temporary: Path, *, authorization_features: bool = True
) -> tuple[list, list, dict]:
    """Execute independent author-directed teacher actions against a fresh real DB."""
    from experiments.decision_model import authorization

    task_path = Path(metadata["task_path"])
    task = load_scenario(task_path / "scenario.yaml")
    if scenario_sha256(task) != metadata["scenario_sha256"]:
        raise ValueError("Frozen scenario changed before teacher collection")
    database = temporary / "billing.db"
    await initialize_database(database, task.initial_state)
    service = BillingService(SQLiteBillingRepository(database), policy_mode="enforced")
    targets = metadata["required_inspections"]
    request = (task_path / "instruction.md").read_text() + PROMPT_SUFFIX
    priors, calls, observations, rows, diagnostics = [], [], [], [], []

    async def invoke(name, arguments):
        calls.append({"name": name, "arguments": arguments})
        try:
            raw = await getattr(service, name)(**arguments)
            if raw is None:
                result = {
                    "ok": False,
                    "error_type": "InvoiceNotFoundError",
                    "error": "invoice not found",
                }
            else:
                key = {
                    "get_invoice": "invoice",
                    "refund_invoice": "refund",
                    "escalate_dispute": "escalation",
                }[name]
                result = {"ok": True, key: raw.model_dump(mode="json")}
                if name == "get_invoice":
                    result["escalation_history"] = read_escalation_history(
                        database, arguments["invoice_id"]
                    )
        except BillingDomainError as exc:
            result = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        observations.append({"name": name, "arguments": arguments, "result": result})
        if name != "get_invoice":
            priors.append(copy.deepcopy(observations[-1]))
        return result

    def make(target, observation):
        if authorization_features:
            return authorization.make_state(request, target, observation, priors, targets)
        return history.make_state(request, target, observation, priors)

    for index, target in enumerate(targets):
        observation = await invoke("get_invoice", {"invoice_id": target})
        state = make(target, observation)
        rows.append(example(state, metadata, f"initial-{index}", "initial"))
        masked = make(target, None)
        diagnostics.append(example(masked, metadata, f"missing-{index}", "diagnostic_missing"))
        operation = metadata["expected_operations"].get(target)
        if operation:
            result = await invoke(
                operation, {"invoice_id": target, "reason": "Customer requested billing resolution"}
            )
            if (
                result.get("ok") is not True
                or result.get(
                    TO_ACTION[operation] if operation == "refund_invoice" else "escalation", {}
                ).get("success")
                is not True
            ):
                raise ValueError(f"Teacher action failed: {task.id}/{target}: {result}")
            after = await invoke("get_invoice", {"invoice_id": target})
            diagnostics.append(
                example(make(target, after), metadata, f"completed-{index}", "actual_completed")
            )
    spec_path = temporary / "scenario.json"
    write(spec_path, task.model_dump(mode="json"))
    trajectory = temporary / "trajectory.json"
    write(trajectory, {"tool_calls": calls})
    verifier = native_verifier()
    verifier.CONTAINER_SCENARIO_PATH = spec_path
    verifier.AGENT_RESULT_PATH = trajectory
    rewards, verifier_diagnostics = verifier.evaluate_database(database)
    if rewards.get("reward") != 1:
        raise ValueError(
            f"Actual teacher failed verification: {task.id}: {rewards} {verifier_diagnostics}"
        )
    return (
        rows,
        diagnostics,
        {
            "task_id": task.id,
            "split": metadata["split"],
            "complete": True,
            "source": "executed-sqlite-billing-tools",
            "reward": 1,
            "verifier_components": rewards,
            "tool_observations": observations,
            "scenario_sha256": metadata["scenario_sha256"],
        },
    )


async def collect(
    corpus: Path, output: Path, splits: tuple = ("train", "dev", "calibration")
) -> dict:
    from experiments.decision_model import authorization

    if output.exists():
        raise FileExistsError(f"Dataset destination already exists: {output}")
    if not splits or set(splits) - set(SPLITS):
        raise ValueError("Unknown collection split")
    manifest = json.loads((corpus / "cohort-manifest.json").read_text())
    # Every split's root allocation already exists before even one state is made.
    output.mkdir(parents=True)
    all_rows, demonstration_records = [], []
    for split in splits:
        initial, extras = [], []
        for metadata in manifest["tasks"].values():
            if metadata["split"] != split:
                continue
            with tempfile.TemporaryDirectory(prefix="next-decision-teacher-") as tmp:
                rows, diagnostics, record = await teacher_rows(metadata, Path(tmp))
            initial.extend(rows)
            extras.extend(diagnostics)
            demonstration_records.append(record)
        # <=20% combined missing/completed rows, independently per split. Use equal
        # quotas per diagnostic kind; deterministic IDs, no outcome-based selection.
        extra_budget = len(initial) // 4
        chosen = []
        pools = {
            kind: sorted(
                [row for row in extras if row["snapshot"] == kind],
                key=lambda row: fingerprint(row["example_id"]),
            )
            for kind in ("actual_completed", "diagnostic_missing")
        }
        while len(chosen) < extra_budget and any(pools.values()):
            for kind in pools:
                if pools[kind] and len(chosen) < extra_budget:
                    chosen.append(pools[kind].pop(0))
        split_rows = initial + chosen
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in split_rows)
        )
        all_rows.extend(split_rows)
    # Model-visible duplicates with conflicting labels are errors, not silently
    # resolved by a teacher parser or by peeking at a heldout prediction.
    states = defaultdict(set)
    for row in all_rows:
        states[row["state"]].add(row["label_action"])
    if any(len(labels) > 1 for labels in states.values()):
        raise ValueError("Conflicting labels on identical model-visible state")
    (output / "dataset.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows)
    )
    (output / "demonstrations.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in demonstration_records)
    )
    result = {
        "schema_version": "1",
        "corpus_manifest_sha256": digest(corpus / "cohort-manifest.json"),
        "dataset_sha256": digest(output / "dataset.jsonl"),
        "generator_sha256": digest(Path(__file__)),
        "authorization_feature_source_sha256": digest(Path(authorization.__file__)),
        "splits_collected": list(splits),
        "final_test_collected": "test" in splits,
        "rows": len(all_rows),
        "verified_teacher_tasks": len(demonstration_records),
        "split_counts": dict(Counter(r["split"] for r in all_rows)),
        "snapshot_counts": dict(Counter(r["snapshot"] for r in all_rows)),
        "label_counts": dict(Counter(r["label_action"] for r in all_rows)),
        "file_sha256": {p.name: digest(p) for p in output.glob("*.jsonl")},
        "label_provenance": (
            "Authored expected operations + explicit ambiguity annotations; "
            "independent actual tool execution and native verifier; "
            "no parser predictions used for labels."
        ),
        "synthetic_missing_and_completed_fraction": sum(
            r["snapshot"] != "initial" for r in all_rows
        )
        / len(all_rows),
        "limitations": manifest["limitations"],
    }
    write(output / "dataset-manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--corpus", type=Path, default=DEFAULT_OUTPUT)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=["train", "dev", "calibration"]
    )
    args = parser.parse_args()
    result = asyncio.run(
        freeze(args.output)
        if args.command == "freeze"
        else collect(args.corpus, args.output, tuple(args.splits))
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"tasks", "historical_exclusions", "source_sha256", "limitations"}
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
