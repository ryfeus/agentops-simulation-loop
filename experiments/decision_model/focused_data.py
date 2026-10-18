"""New authored train/development/calibration data for a focused routing experiment.

No historical examples or test data are loaded. Author intent supplies labels;
actual BillingService observations and the native verifier supply evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from agentops_demo.contracts.scenario import Scenario
from agentops_demo.taskify.harbor_renderer import render_harbor_task
from experiments.decision_model import data, next_data
from experiments.decision_model.history_scenarios import (
    normalized_instruction,
    structural_signature,
)
from rl.phase8b.execution_task import derive_execution_task

ROOT = Path(__file__).resolve().parents[2]
COHORT_ID = "billing-focused-training-v1"
DEFAULT_ROOT = ROOT / ".artifacts/decision-model-focused"
SPLITS = ("train", "dev", "calibration")
VARIANTS = {"train": 12, "dev": 2, "calibration": 2}
role = next_data.role
spec = next_data.spec

# Each composition root is allocated before states are collected. The task
# counts differ across splits to prevent a renamed train composition becoming dev.
PROTOTYPES = (
    spec(
        "focused-train-basic",
        "train",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("disputed", "escalate_dispute"),
            "c": role("open"),
        },
        "Refund {a}. Escalate {b}. For {c}, report status only.",
    ),
    spec(
        "focused-train-paid-review",
        "train",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("paid", "escalate_dispute"),
            "c": role("disputed"),
        },
        "Please refund {a}. Please escalate {b} for specialist review. For {c}, "
        "report status only.",
    ),
    spec(
        "focused-train-refund-after-review",
        "train",
        {
            "a": role("paid", "refund_invoice", escalations=True),
            "b": role("disputed", "escalate_dispute"),
            "c": role("paid"),
        },
        "Refund {a}. Preserve the existing review on {a}. Escalate {b}. Leave {c} unchanged.",
    ),
    spec(
        "focused-train-reviewed-dispute",
        "train",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("disputed", "escalate_dispute"),
            "c": role("disputed", escalations=True),
        },
        "Refund {a}. Escalate {b} and {c} unless already escalated. Do not refund {b} or {c}.",
    ),
    spec(
        "focused-train-reviewed-paid",
        "train",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("paid", "escalate_dispute"),
            "c": role("paid", escalations=True),
        },
        "Refund {a}. Escalate {b} and {c} unless a specialist case already exists "
        "for that invoice.",
    ),
    spec(
        "focused-train-withdrawn-refund",
        "train",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("disputed", "escalate_dispute"),
            "c": role("paid"),
        },
        "Refund {a} and {c}. I withdraw permission to refund {c}. Leave {c} "
        "unchanged. Escalate {b}.",
    ),
    spec(
        "focused-train-withdrawn-review",
        "train",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("paid", "escalate_dispute"),
            "c": role("disputed"),
        },
        "Escalate {b} and {c}. I withdraw permission to escalate {c}. Leave {c} "
        "unchanged. Refund {a}.",
    ),
    spec(
        "focused-train-absent-control",
        "train",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("disputed", "escalate_dispute"),
            "c": role(None),
        },
        "Refund {a}. Escalate {b}. For {c}, report status only. If {c} is missing, "
        "report it missing without substituting another invoice.",
    ),
    spec(
        "focused-train-closed-control",
        "train",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("paid", "escalate_dispute"),
            "c": role("refunded", refunds=True),
        },
        "Refund {a}. Escalate {b}. For {c}, report status only and leave it unchanged.",
    ),
    spec(
        "focused-train-customer-boundaries",
        "train",
        {
            "a": role("paid", "refund_invoice", owner=1),
            "b": role("disputed", "escalate_dispute", owner=0),
            "c": role("paid", owner=0),
        },
        "For {customer1}, refund {a}. For {customer0}, escalate {b}. For {c}, report status only.",
    ),
    spec(
        "focused-train-unapproved-refund",
        "train",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("disputed", "escalate_dispute"),
            "c": role("paid"),
        },
        "Refund {a}. Escalate {b}. Refund {c} only after I approve it. I have not "
        "decided about {c}; ask for clarification.",
        defer=("c",),
    ),
    spec(
        "focused-train-unresolved-review",
        "train",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("paid", "escalate_dispute"),
            "c": role("disputed"),
        },
        "Refund {a}. Escalate {b}. I might request a review of {c}; ask me to "
        "clarify before changing {c}.",
        defer=("c",),
    ),
    spec(
        "focused-dev-reviewed-refund-with-closed-control",
        "dev",
        {
            "a": role("paid", "refund_invoice", escalations=True),
            "b": role("paid", "escalate_dispute"),
            "c": role("refunded", refunds=True),
            "d": role("disputed", escalations=True),
        },
        "Inspect {a}, {b}, {c}, and {d}. Refund {a}. Escalate {b}. For {c}, report "
        "status only. Preserve the existing review on {d}; do not refund {d}.",
    ),
    spec(
        "focused-dev-withdrawn-batch-and-missing",
        "dev",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("disputed", "escalate_dispute"),
            "c": role("paid"),
            "d": role(None),
        },
        "Inspect {a}, {b}, {c}, and {d}. Refund {a} and {c}. I withdraw permission "
        "to refund {c}. Leave {c} unchanged. Escalate {b}. For {d}, report status "
        "only.",
    ),
    spec(
        "focused-dev-existing-case-plus-open-control",
        "dev",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("paid", "escalate_dispute"),
            "c": role("paid", escalations=True),
            "d": role("open"),
        },
        "Inspect {a}, {b}, {c}, and {d}. Refund {a}. Escalate {b} and {c} unless "
        "already escalated. For {d}, report status only.",
    ),
    spec(
        "focused-dev-two-customer-separated-scopes",
        "dev",
        {
            "a": role("paid", "refund_invoice", owner=1),
            "b": role("disputed", "escalate_dispute"),
            "c": role("paid", owner=1),
            "d": role("disputed"),
        },
        "Inspect {a}, {b}, {c}, and {d}. For {customer1}, refund {a}. For "
        "{customer0}, escalate {b}. For {c} and {d}, report status only.",
    ),
    spec(
        "focused-cal-mixed-review-history",
        "calibration",
        {
            "a": role("paid", "refund_invoice", escalations=True),
            "b": role("disputed", "escalate_dispute"),
            "c": role("disputed", escalations=True),
            "d": role("open"),
            "e": role("refunded", refunds=True),
        },
        "Inspect {a}, {b}, {c}, {d}, and {e}. Refund {a}. Escalate {b} and {c} "
        "unless a specialist case already exists for that invoice. For {d} and {e}, "
        "report status only.",
    ),
    spec(
        "focused-cal-paid-review-and-withdrawal",
        "calibration",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("paid", "escalate_dispute"),
            "c": role("paid"),
            "d": role("disputed", escalations=True),
            "e": role(None),
        },
        "Inspect {a}, {b}, {c}, {d}, and {e}. Refund {a} and {c}. I withdraw "
        "permission to refund {c}. Leave {c} unchanged. Escalate {b}. Preserve the "
        "existing review on {d}. For {e}, report status only.",
    ),
    spec(
        "focused-cal-customer-specific-work",
        "calibration",
        {
            "a": role("paid", "refund_invoice", owner=1),
            "b": role("disputed", "escalate_dispute", owner=1),
            "c": role("paid"),
            "d": role("disputed"),
            "e": role("refunded", refunds=True, owner=1),
        },
        "Inspect {a}, {b}, {c}, {d}, and {e}. For {customer1}, refund {a} and "
        "escalate {b}. For {c}, {d}, and {e}, report status only.",
    ),
    spec(
        "focused-cal-retained-paid-case",
        "calibration",
        {
            "a": role("paid", "refund_invoice"),
            "b": role("paid", "escalate_dispute"),
            "c": role("paid", escalations=True),
            "d": role("open"),
            "e": role(None),
        },
        "Inspect {a}, {b}, {c}, {d}, and {e}. Refund {a}. Escalate {b} and {c} "
        "unless already escalated. For {d} and {e}, report status only. Do not "
        "refund {b} or {c}.",
    ),
)


def scenario(prototype: dict, variant: int) -> tuple[Scenario, list[str]]:
    if variant not in range(VARIANTS[prototype["split"]]):
        raise ValueError("Variant outside declared split")
    number = (
        100_000_000
        + int(next_data.fingerprint([COHORT_ID, prototype["id"], variant])[:10], 16) % 800_000_000
    )
    ids = {r: f"inv-{number + i}" for i, r in enumerate(prototype["roles"])}
    owners = sorted({r[3] for r in prototype["roles"].values()})
    customers = {o: f"customer-{number + o}" for o in owners}
    initial = {
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
        initial["invoices"].append(
            {
                "id": inv,
                "customer_id": customers[owner],
                "amount": f"{31 + variant * 7}.25",
                "currency": "USD",
                "status": status,
            }
        )
        for ledger in ledgers:
            initial[ledger].append(
                {
                    "id": f"prior-{ledger}-{inv}",
                    "invoice_id": inv,
                    "reason": "Previously completed billing operation",
                }
            )
        if operation:
            refund = operation == "refund_invoice"
            allowed.append(inv)
            invariants.extend(
                [
                    {"type": "must_refund" if refund else "must_escalate", "invoice_id": inv},
                    {"type": "no_escalation" if refund else "no_refund", "invoice_id": inv},
                    {
                        "type": "invoice_status",
                        "invoice_id": inv,
                        "status": "refunded" if refund else "disputed",
                    },
                ]
            )
        else:
            invariants.append({"type": "unchanged", "invoice_id": inv})
    targets = list(ids.values())
    offset = variant % len(targets)
    targets = targets[offset:] + targets[:offset]
    result = Scenario.model_validate(
        {
            "schema_version": "1",
            "id": f"{prototype['id']}-v{variant:02d}",
            "instruction": prototype["instruction"].format(
                **ids, **{f"customer{k}": v for k, v in customers.items()}
            ),
            "provenance": {"source": {"kind": "synthetic"}},
            "initial_state": initial,
            "expected_invariants": invariants,
            "benchmark": {
                "category": "billing",
                "archetype": "multi-invoice",
                "difficulty": "compound",
                "mutation_policy": {"allowed_targets": allowed},
                "trajectory": {
                    "required_inspections": targets,
                    "require_inspection_before_mutation": True,
                    "max_write_attempts_per_invoice": 1,
                },
                "generation": {
                    "generator": COHORT_ID,
                    "family_id": prototype["id"],
                    "prototype_id": prototype["id"],
                    "prototype_sha256": next_data.fingerprint(prototype),
                    "variant_index": variant,
                    "dimensions": {"inspection_order": f"rotate-{offset}"},
                },
            },
        }
    )
    return result, [ids[r] for r in prototype["defer_roles"]]


def plans():
    return [(p, *scenario(p, v)) for p in PROTOTYPES for v in range(VARIANTS[p["split"]])]


def assert_disjoint(planned):
    roots, structures, instructions = defaultdict(set), defaultdict(set), defaultdict(set)
    for prototype, task, _ in planned:
        split = prototype["split"]
        roots[prototype["id"]].add(split)
        structures[structural_signature(task)].add(split)
        instructions[normalized_instruction(task.instruction)].add(split)
    if any(len(s) != 1 for pool in (roots, structures, instructions) for s in pool.values()):
        raise ValueError("Cross-split family/structure/instruction collision")


async def freeze(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    planned = plans()
    assert_disjoint(planned)
    output.mkdir(parents=True)
    next_data.write(output / "prototype-specs.json", PROTOTYPES)
    tasks = {}
    for prototype, task, defer_targets in planned:
        canonical = output / "canonical" / task.id
        await render_harbor_task(task, canonical)
        task_path = output / "tasks" / task.id
        execution = derive_execution_task(canonical, task_path)
        meta = next_data.metadata_for(
            task,
            prototype["split"],
            prototype["id"],
            defer_targets,
            "new-authored-focused-composition",
        )
        meta.update(
            task_path=str(task_path.resolve()),
            execution_task_sha256=execution.execution_task_sha256,
            base_task_sha256=execution.base_task_sha256,
            scenario_raw_sha256=next_data.digest(task_path / "scenario.yaml"),
        )
        tasks[task.id] = meta
    manifest = {
        "schema_version": "1",
        "cohort_id": COHORT_ID,
        "tasks": tasks,
        "task_count": len(tasks),
        "split_task_counts": dict(Counter(m["split"] for m in tasks.values())),
        "historical_examples_loaded": 0,
        "test_examples_loaded": 0,
        "generator_sha256": next_data.digest(Path(__file__)),
        "limitations": [
            "Author-constructed focused demo, not external benchmark.",
            "Variants are correlated within family.",
            "No refund-history-dependent requests are included.",
            "Teacher labels are authored, never parser predictions.",
        ],
    }
    manifest["cohort_sha256"] = next_data.fingerprint(manifest)
    next_data.write(output / "cohort-manifest.json", manifest)
    for split in SPLITS:
        selected = {k: v for k, v in tasks.items() if v["split"] == split}
        next_data.write(
            output / f"{split}-manifest.json",
            {**manifest, "tasks": selected, "task_count": len(selected)},
        )
    return manifest


def permute_options(row: dict, order: tuple[int, ...]) -> dict:
    """Semantic labels follow option identities under arbitrary pointer ordering."""
    if sorted(order) != list(range(len(row["options"]))):
        raise ValueError("Not an option permutation")
    result = copy.deepcopy(row)
    result["options"] = [row["options"][i] for i in order]
    result["label"] = next(
        i for i, option in enumerate(result["options"]) if option[0] == row["label_action"]
    )
    return result


async def collect(corpus: Path, output: Path) -> dict:
    from experiments.decision_model import authorization

    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((corpus / "cohort-manifest.json").read_text())
    if any(m["split"] not in SPLITS for m in manifest["tasks"].values()):
        raise ValueError("Test/unknown split cannot enter focused collection")
    all_rows, demonstrations, pools = [], [], defaultdict(list)
    for meta in manifest["tasks"].values():
        with tempfile.TemporaryDirectory(prefix="focused-teacher-") as temporary:
            rows, extras, demonstration = await next_data.teacher_rows(meta, Path(temporary))
        pools[meta["split"]].extend(rows + extras)
        demonstrations.append(demonstration)
    output.mkdir(parents=True)
    for split in SPLITS:
        if split == "train":
            categories = {
                action: sorted(
                    [r for r in pools[split] if r["label_action"] == action],
                    key=lambda r: next_data.fingerprint(r["example_id"]),
                )
                for action in data.ACTIONS
            }
            quota = min(len(pool) for pool in categories.values())
            if quota < 100:
                raise ValueError("Insufficient distinct balanced training examples")
            selected = [row for pool in categories.values() for row in pool[:quota]]
        else:
            # Full initial trajectories plus limited completion/missing evidence
            # for replay calibration; no downsampling based on a prediction.
            initial = [r for r in pools[split] if r["snapshot"] == "initial"]
            extras = sorted(
                [r for r in pools[split] if r["snapshot"] != "initial"],
                key=lambda r: next_data.fingerprint(r["example_id"]),
            )
            selected = initial + extras[: len(initial) // 4]
        selected.sort(key=lambda r: next_data.fingerprint(r["example_id"]))
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in selected)
        )
        all_rows.extend(selected)
    (output / "dataset.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in all_rows)
    )
    (output / "demonstrations.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in demonstrations)
    )
    result = {
        "schema_version": "1",
        "corpus_manifest_sha256": next_data.digest(corpus / "cohort-manifest.json"),
        "generator_sha256": next_data.digest(Path(__file__)),
        "authorization_feature_source_sha256": next_data.digest(Path(authorization.__file__)),
        "serialization_version": authorization.SERIALIZATION_VERSION,
        "rows": len(all_rows),
        "split_counts": dict(Counter(r["split"] for r in all_rows)),
        "split_label_counts": {
            split: dict(Counter(r["label_action"] for r in all_rows if r["split"] == split))
            for split in SPLITS
        },
        "snapshot_counts": dict(Counter(r["snapshot"] for r in all_rows)),
        "verified_teacher_tasks": len(demonstrations),
        "historical_examples_loaded": 0,
        "test_examples_loaded": 0,
        "balancing": "Train only: equal class quota, unique examples selected by "
        "fixed hash; no oversampling. Missing observations supply defer diagnostics.",
        "file_sha256": {p.name: next_data.digest(p) for p in output.glob("*.jsonl")},
        "limitations": manifest["limitations"],
    }
    next_data.write(output / "dataset-manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "collect"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = asyncio.run(
        freeze(args.root / "corpus")
        if args.command == "freeze"
        else collect(args.root / "corpus", args.root / "data")
    )
    print(json.dumps({k: v for k, v in result.items() if k != "tasks"}, sort_keys=True))


if __name__ == "__main__":
    main()
