"""Deterministic, typed billing Scenario variants and frozen corpus lineage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from string import Formatter
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from agentops_demo.contracts._base import ContractModel
from agentops_demo.contracts.scenario import Scenario
from agentops_demo.taskify.integrity import scenario_sha256
from agentops_demo.validation.scenario import dump_scenario_yaml, load_scenario

ROOT = Path(__file__).resolve().parents[3]
PHASE10 = ROOT / "benchmarks" / "billing" / "phase10"
GENERATOR_VERSION = "billing-phase10-v1"
GLOBAL_SEED = 20260925
AMOUNTS = ("9.99", "19.00", "49.00", "85.50", "120.00", "249.95")
ARCHETYPES = (
    "normal-refund",
    "disputed-refund",
    "read-only",
    "missing-invoice",
    "multi-invoice",
    "already-refunded",
    "already-escalated",
    "duplicate-action",
    "wrong-target",
)
NEGATIVE_STRATEGIES = (
    "lookup_only",
    "refund_target",
    "escalate_target",
    "refund_wrong_target",
    "escalate_wrong_target",
    "mutate_without_inspection",
    "duplicate_refund",
    "duplicate_escalation",
    "noop",
)
DIMENSIONS = {
    "target_amount",
    "distractor_count",
    "distractor_status",
    "distractor_owner",
    "target_invoice_style",
    "entity_order",
}
PLACEHOLDERS = {"target_invoice", "other_invoice", "customer_id", "target_amount"}


class FamilySpec(ContractModel):
    schema_version: Literal["1"]
    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    prototype: str
    archetype: Literal[
        "normal-refund",
        "disputed-refund",
        "read-only",
        "missing-invoice",
        "multi-invoice",
        "already-refunded",
        "already-escalated",
        "duplicate-action",
        "wrong-target",
    ]
    difficulty: Literal["easy", "paraphrase", "adversarial", "compound"]
    variant_count: int = Field(ge=1)
    instruction_templates: list[str] = Field(min_length=2)
    dimensions: dict[str, list[str]]
    constraints: dict[str, list[str]]
    negative_strategy: str
    inspection_scope: Literal["target", "both"] = "target"
    diversity_exception: str | None = None

    @model_validator(mode="after")
    def validate_family(self) -> FamilySpec:
        if self.negative_strategy not in NEGATIVE_STRATEGIES:
            raise ValueError(f"unknown negative strategy {self.negative_strategy}")
        if set(self.dimensions) != DIMENSIONS:
            raise ValueError(f"dimensions must be exactly {sorted(DIMENSIONS)}")
        if any(not pool for pool in self.dimensions.values()):
            raise ValueError("dimension pools must be nonempty")
        if not set(self.dimensions["target_amount"]).issubset(AMOUNTS):
            raise ValueError("target_amount must use the reviewed amount pool")
        if not set(self.dimensions["distractor_count"]).issubset({"0", "1", "2"}):
            raise ValueError("invalid distractor count")
        if not set(self.dimensions["distractor_status"]).issubset({"paid", "open", "disputed"}):
            raise ValueError("invalid distractor status")
        if not set(self.dimensions["distractor_owner"]).issubset({"same", "other"}):
            raise ValueError("invalid distractor owner")
        if not set(self.dimensions["target_invoice_style"]).issubset({"numeric", "similar_ids"}):
            raise ValueError("invalid invoice ID style")
        if not set(self.dimensions["entity_order"]).issubset({"original", "reverse"}):
            raise ValueError("invalid entity order")
        if set(self.constraints) != {"target_status"} or not self.constraints["target_status"]:
            raise ValueError("constraints.target_status is required")
        for template in self.instruction_templates:
            names = {name for _, name, _, _ in Formatter().parse(template) if name is not None}
            if "target_invoice" not in names or names - PLACEHOLDERS or not template.strip():
                raise ValueError(f"invalid instruction template placeholders: {template!r}")
        if len(set(self.instruction_templates)) != len(self.instruction_templates):
            raise ValueError("duplicate instruction template")
        return self


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def load_families(root: Path = PHASE10) -> list[FamilySpec]:
    files = sorted((root / "families").glob("*.yaml"))
    if not files:
        raise ValueError("Phase 10 has no family specs")
    families = [FamilySpec.model_validate(_read_yaml(path)) for path in files]
    ids = [family.id for family in families]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate family ID")
    if any(path.stem != family.id for path, family in zip(files, families, strict=True)):
        raise ValueError("family filename must match its ID")
    return families


def _seed(family_id: str, index: int) -> int:
    payload = [GENERATOR_VERSION, GLOBAL_SEED, family_id, index]
    return int.from_bytes(hashlib.sha256(_canonical(payload)).digest()[:8], "big")


def _target_id(scenario: Scenario) -> str:
    assert scenario.benchmark is not None
    allowed = scenario.benchmark.mutation_policy.allowed_targets
    if allowed:
        return allowed[0]
    known = {invoice.id for invoice in scenario.initial_state.invoices}
    missing = [
        value for value in scenario.benchmark.trajectory.required_inspections if value not in known
    ]
    if missing:
        if scenario.benchmark.archetype != "missing-invoice" or len(missing) != 1:
            raise ValueError("only missing-invoice prototypes may inspect one absent invoice")
        return missing[0]
    return scenario.initial_state.invoices[0].id


def _validate_prototype(family: FamilySpec, prototype: Scenario) -> None:
    if prototype.benchmark is None or prototype.benchmark.archetype != family.archetype:
        raise ValueError(f"{family.id}: prototype archetype mismatch")
    if prototype.provenance.source.kind != "synthetic" or prototype.observed_failure is not None:
        raise ValueError(f"{family.id}: prototype must be synthetic without observed failure")
    target = _target_id(prototype)
    by_id = {invoice.id: invoice for invoice in prototype.initial_state.invoices}
    actual = by_id[target].status if target in by_id else "missing"
    if actual not in family.constraints["target_status"]:
        raise ValueError(f"{family.id}: prototype target status {actual} violates constraints")


def _dimensions(family: FamilySpec, index: int) -> dict[str, str]:
    seed = _seed(family.id, index)
    values = {}
    for offset, key in enumerate(sorted(DIMENSIONS)):
        pool = family.dimensions[key]
        shift = _seed(family.id, 0) if key == "target_amount" else seed >> (offset * 8)
        values[key] = pool[(index + (shift & 0xFF)) % len(pool)]
    values["instruction_template"] = f"template-{index % len(family.instruction_templates)}"
    return values


def make_variant(family: FamilySpec, prototype: Scenario, index: int) -> Scenario:
    """Rewrite model fields structurally, then validate the entire Scenario."""
    _validate_prototype(family, prototype)
    if not 0 <= index < family.variant_count:
        raise ValueError("variant index outside family")
    values = _dimensions(family, index)
    seed = _seed(family.id, index)
    base = 1000 + seed % 8000
    target_old = _target_id(prototype)
    old_ids = [invoice.id for invoice in prototype.initial_state.invoices]
    all_ids = list(dict.fromkeys([target_old, *old_ids]))
    invoice_map = {old: f"inv-{base + number * 10}" for number, old in enumerate(all_ids)}
    if values["target_invoice_style"] == "similar_ids":
        invoice_map = {old: f"inv-{base + number}" for number, old in enumerate(all_ids)}
    customer_map = {
        customer.id: f"customer-{100 + (base + number) % 800}"
        for number, customer in enumerate(prototype.initial_state.customers)
    }
    payload = prototype.model_dump(mode="json")
    payload["id"] = f"{family.id}-v{index:02d}"
    state = payload["initial_state"]
    for customer in state["customers"]:
        customer["id"] = customer_map[customer["id"]]
    for number, invoice in enumerate(state["invoices"]):
        original = invoice["id"]
        invoice["id"] = invoice_map[original]
        invoice["customer_id"] = customer_map[invoice["customer_id"]]
        invoice["amount"] = (
            values["target_amount"]
            if original == target_old or (target_old not in old_ids and number == 0)
            else AMOUNTS[(index + number + 2) % len(AMOUNTS)]
        )
    for number, record in enumerate(state["refunds"]):
        record["id"] = f"refund-{base + number}"
        record["invoice_id"] = invoice_map[record["invoice_id"]]
    for number, record in enumerate(state["escalations"]):
        record["id"] = f"escalation-{base + number}"
        record["invoice_id"] = invoice_map[record["invoice_id"]]
    for invariant in payload["expected_invariants"]:
        invariant["invoice_id"] = invoice_map[invariant["invoice_id"]]
    benchmark = payload["benchmark"]
    benchmark["difficulty"] = family.difficulty
    benchmark["mutation_policy"]["allowed_targets"] = [
        invoice_map[value] for value in benchmark["mutation_policy"]["allowed_targets"]
    ]
    benchmark["trajectory"]["required_inspections"] = [
        invoice_map[value] for value in benchmark["trajectory"]["required_inspections"]
    ]
    # New distractors are protected by an explicit final-state invariant.
    distractor_count = int(values["distractor_count"])
    distractor_customer = state["customers"][0]["id"]
    if distractor_count and values["distractor_owner"] == "other":
        distractor_customer = f"customer-{900 + base}"
        state["customers"].append({"id": distractor_customer, "status": "active"})
        allowed = set(benchmark["mutation_policy"]["allowed_targets"])
        for invoice in state["invoices"]:
            if invoice["id"] != invoice_map[target_old] and invoice["id"] not in allowed:
                invoice["customer_id"] = distractor_customer
    for number in range(distractor_count):
        invoice_id = f"inv-{base + 100 + number}"
        state["invoices"].append(
            {
                "id": invoice_id,
                "customer_id": distractor_customer,
                "amount": AMOUNTS[(index + number + 1) % len(AMOUNTS)],
                "currency": "USD",
                "status": values["distractor_status"] if number == 0 else "paid",
            }
        )
        payload["expected_invariants"].append({"type": "unchanged", "invoice_id": invoice_id})
    if values["entity_order"] == "reverse":
        state["invoices"].reverse()
    target_new = invoice_map[target_old]
    other = next((invoice_map[old] for old in old_ids if old != target_old), None)
    if other is None and distractor_count:
        other = f"inv-{base + 100}"
    template_index = index % len(family.instruction_templates)
    template = family.instruction_templates[template_index]
    if "{other_invoice}" in template and other is None:
        raise ValueError(f"{family.id}: instruction requires an absent other invoice")
    if family.inspection_scope == "both":
        if other is None:
            raise ValueError(f"{family.id}: both inspections require a second invoice")
        required = benchmark["trajectory"]["required_inspections"]
        if other not in required:
            required.append(other)
    payload["instruction"] = template.format(
        target_invoice=target_new,
        other_invoice=other,
        customer_id=state["customers"][0]["id"],
        target_amount=values["target_amount"],
    )
    benchmark["generation"] = {
        "generator": GENERATOR_VERSION,
        "family_id": family.id,
        "prototype_id": prototype.id,
        "prototype_sha256": scenario_sha256(prototype),
        "variant_index": index,
        "dimensions": values,
    }
    result = Scenario.model_validate(payload)
    validate_generated(result, family, prototype)
    return result


def validate_generated(scenario: Scenario, family: FamilySpec, prototype: Scenario) -> None:
    benchmark = scenario.benchmark
    if benchmark is None or benchmark.generation is None:
        raise ValueError(f"{scenario.id}: generation metadata missing")
    generation = benchmark.generation
    if (
        scenario.id != f"{family.id}-v{generation.variant_index:02d}"
        or generation.family_id != family.id
        or generation.prototype_id != prototype.id
        or generation.prototype_sha256 != scenario_sha256(prototype)
        or benchmark.archetype != family.archetype
        or benchmark.difficulty != family.difficulty
        or scenario.provenance.source.kind != "synthetic"
        or scenario.observed_failure is not None
    ):
        raise ValueError(f"{scenario.id}: generated scenario lineage is inconsistent")
    known = {invoice.id for invoice in scenario.initial_state.invoices}
    inspections = set(benchmark.trajectory.required_inspections)
    unknown = inspections - known
    if unknown and (family.archetype != "missing-invoice" or len(unknown) != 1):
        raise ValueError(f"{scenario.id}: invalid required inspections {sorted(unknown)}")
    if not set(benchmark.mutation_policy.allowed_targets) <= known:
        raise ValueError(f"{scenario.id}: mutation policy has unknown targets")
    if not {item.invoice_id for item in scenario.expected_invariants} <= known:
        raise ValueError(f"{scenario.id}: invariant has unknown invoice")


def _normalized_instruction(instruction: str) -> str:
    value = re.sub(r"inv-\d+", "inv-id", instruction.lower())
    value = re.sub(r"customer-\d+", "customer-id", value)
    value = re.sub(r"\b\d+\.\d{2}\b", "amount", value)
    return " ".join(value.split())


def behavior_signature_sha256(scenario: Scenario) -> str:
    """Hash task behavior while ignoring cosmetic identifiers and presentation."""
    benchmark = scenario.benchmark
    if benchmark is None:
        raise ValueError(f"{scenario.id}: benchmark metadata is required")
    state = scenario.initial_state
    allowed = set(benchmark.mutation_policy.allowed_targets)
    required = set(benchmark.trajectory.required_inspections)
    invoices = {invoice.id: invoice for invoice in state.invoices}
    refund_counts = Counter(record.invoice_id for record in state.refunds)
    escalation_counts = Counter(record.invoice_id for record in state.escalations)
    invariants: dict[str, list[tuple[str, str | None]]] = {}
    for invariant in scenario.expected_invariants:
        invariants.setdefault(invariant.invoice_id, []).append(
            (invariant.type, getattr(invariant, "status", None))
        )
    customer_groups = []
    for customer in state.customers:
        rows = [
            (
                invoice.status,
                invoice.currency,
                refund_counts[invoice.id],
                escalation_counts[invoice.id],
                invoice.id in allowed,
                invoice.id in required,
                sorted(invariants.get(invoice.id, [])),
            )
            for invoice in state.invoices
            if invoice.customer_id == customer.id
        ]
        customer_groups.append((customer.status, sorted(rows)))
    payload = {
        "archetype": benchmark.archetype,
        "customer_groups": sorted(customer_groups),
        "absent_required_inspections": len(required - invoices.keys()),
        "allowed_target_count": len(allowed),
        "required_inspection_count": len(required),
        "require_inspection_before_mutation": (
            benchmark.trajectory.require_inspection_before_mutation
        ),
        "max_write_attempts_per_invoice": benchmark.trajectory.max_write_attempts_per_invoice,
    }
    return _sha(payload)


def _split(
    root: Path, families: list[FamilySpec], expected: dict[str, int]
) -> tuple[dict[str, list[str]], str]:
    config = _read_yaml(root / "split.yaml")
    if config.get("schema_version") != "1" or not isinstance(config.get("families"), dict):
        raise ValueError("invalid split config")
    splits = config["families"]
    if set(splits) != {"train", "dev", "holdout"}:
        raise ValueError("split must define train, dev, and holdout")
    values = [value for members in splits.values() for value in members]
    known = {family.id for family in families}
    if set(values) != known or len(values) != len(set(values)):
        raise ValueError("split has missing, duplicate, or unknown families")
    by_id = {family.id: family for family in families}
    prototype_assignments: dict[str, dict[str, list[str]]] = {}
    for split_name, members in splits.items():
        for family_id in members:
            prototype_assignments.setdefault(by_id[family_id].prototype, {}).setdefault(
                split_name, []
            ).append(family_id)
    for prototype, assignments in sorted(prototype_assignments.items()):
        if len(assignments) > 1:
            detail = "; ".join(
                f"{name}: {', '.join(sorted(members))}"
                for name, members in sorted(assignments.items())
            )
            raise ValueError(f"prototype {prototype} crosses splits: {detail}")
    for name, members in splits.items():
        if sum(by_id[member].variant_count for member in members) != expected[name]:
            raise ValueError(f"{name} split has wrong task count")
    archetypes = {family.id: family.archetype for family in families}
    if len(families) == 40 and {archetypes[value] for value in splits["train"]} != set(ARCHETYPES):
        raise ValueError("train must cover every archetype")
    for name in ("dev", "holdout"):
        if len(families) == 40 and len({archetypes[value] for value in splits[name]}) < 3:
            raise ValueError(f"{name} lacks behavioral diversity")
    return {name: sorted(members) for name, members in splits.items()}, _sha(config)


def build_corpus(root: Path, output: Path) -> dict[str, Any]:
    config = _read_yaml(root / "config.yaml")
    if (
        config.get("schema_version") != "1"
        or config.get("generator_version") != GENERATOR_VERSION
        or config.get("global_seed") != GLOBAL_SEED
    ):
        raise ValueError("Phase 10 generator configuration mismatch")
    expected = config["split_counts"]
    if root.resolve() == PHASE10.resolve() and (
        config.get("family_count") != 40
        or config.get("task_count") != 200
        or expected != {"train": 160, "dev": 20, "holdout": 20}
    ):
        raise ValueError("production Phase 10 requires 40 families and a 160/20/20 split")
    families = load_families(root)
    if (
        len(families) != config["family_count"]
        or sum(f.variant_count for f in families) != config["task_count"]
    ):
        raise ValueError("Phase 10 family or task count mismatch")
    if len(families) == 40 and any(family.variant_count != 5 for family in families):
        raise ValueError("production families require exactly five variants")
    splits, split_config_sha = _split(root, families, expected)
    output.mkdir(parents=True, exist_ok=False)
    scenarios = output / "scenarios"
    scenarios.mkdir()
    family_specs_sha = _sha(
        [family.model_dump(mode="json") for family in sorted(families, key=lambda f: f.id)]
    )
    tasks: dict[str, dict[str, Any]] = {}
    scenario_hashes: set[str] = set()
    dimension_hashes: set[str] = set()
    instruction_splits: dict[str, str] = {}
    behavior_uses: dict[str, list[tuple[str, str, str, str]]] = {}
    counts_archetype: Counter[str] = Counter()
    counts_difficulty: Counter[str] = Counter()
    counts_prototype: Counter[str] = Counter()
    distractor_counts: Counter[str] = Counter()
    template_count = 0
    exceptions: list[str] = []
    for family in sorted(families, key=lambda item: item.id):
        prototype_path = (
            ROOT / "benchmarks" / "billing" / "scenarios" / family.prototype / "scenario.yaml"
        )
        if not prototype_path.is_file():
            raise ValueError(f"{family.id}: unknown prototype {family.prototype}")
        prototype = load_scenario(prototype_path)
        _validate_prototype(family, prototype)
        family_templates: set[str] = set()
        family_states: set[str] = set()
        family_hashes: set[str] = set()
        split = next(name for name, members in splits.items() if family.id in members)
        for index in range(family.variant_count):
            scenario = make_variant(family, prototype, index)
            task_id = scenario.id
            digest = scenario_sha256(scenario)
            dimensions = scenario.benchmark.generation.dimensions
            dimension_digest = _sha([family.id, dimensions])
            if digest in scenario_hashes or dimension_digest in dimension_hashes:
                raise ValueError(f"{task_id}: duplicate Scenario or dimension signature")
            normalized = _normalized_instruction(scenario.instruction)
            previous_split = instruction_splits.get(normalized)
            if previous_split is not None and previous_split != split:
                raise ValueError(f"{task_id}: normalized instruction leaks across splits")
            instruction_splits[normalized] = split
            behavior_signature = behavior_signature_sha256(scenario)
            behavior_uses.setdefault(behavior_signature, []).append(
                (task_id, family.id, prototype.id, split)
            )
            scenario_hashes.add(digest)
            dimension_hashes.add(dimension_digest)
            family_hashes.add(digest)
            family_templates.add(dimensions["instruction_template"])
            family_states.add(
                _sha(
                    {
                        key: value
                        for key, value in dimensions.items()
                        if key != "instruction_template"
                    }
                )
            )
            task_dir = scenarios / task_id
            task_dir.mkdir()
            (task_dir / "scenario.yaml").write_text(dump_scenario_yaml(scenario), encoding="utf-8")
            tasks[task_id] = {
                "family_id": family.id,
                "prototype_id": prototype.id,
                "prototype_sha256": scenario_sha256(prototype),
                "variant_index": index,
                "archetype": family.archetype,
                "difficulty": family.difficulty,
                "scenario_sha256": digest,
                "dimension_signature_sha256": dimension_digest,
                "behavior_signature_sha256": behavior_signature,
            }
            counts_archetype[family.archetype] += 1
            counts_difficulty[family.difficulty] += 1
            counts_prototype[prototype.id] += 1
            distractor_counts[dimensions["distractor_count"]] += 1
        if (
            len(family_hashes) != family.variant_count
            or len(family_templates) < 2
            or len(family_states) < 2
        ):
            if family.diversity_exception is None:
                raise ValueError(f"{family.id}: family diversity gate failed")
            exceptions.append(family.id)
        template_count += len(family_templates)
    cross_split_signatures = {
        signature: uses
        for signature, uses in behavior_uses.items()
        if len({use[3] for use in uses}) > 1
    }
    if cross_split_signatures:
        signature, uses = sorted(cross_split_signatures.items())[0]
        detail = "; ".join(
            f"task={task}, family={family}, prototype={prototype}, split={split}"
            for task, family, prototype, split in sorted(uses)
        )
        raise ValueError(f"behavior signature {signature} crosses splits: {detail}")
    corpus_sha = _sha([(task_id, tasks[task_id]["scenario_sha256"]) for task_id in sorted(tasks)])
    manifest = {
        "schema_version": "1",
        "generator_version": GENERATOR_VERSION,
        "global_seed": GLOBAL_SEED,
        "task_count": len(tasks),
        "family_count": len(families),
        "corpus_sha256": corpus_sha,
        "family_specs_sha256": family_specs_sha,
        "tasks": {task_id: tasks[task_id] for task_id in sorted(tasks)},
        "counts": {
            "by_archetype": dict(sorted(counts_archetype.items())),
            "by_difficulty": dict(sorted(counts_difficulty.items())),
        },
    }
    split_tasks = {
        task_id: next(
            name for name, members in splits.items() if tasks[task_id]["family_id"] in members
        )
        for task_id in sorted(tasks)
    }
    split_manifest = {
        "schema_version": "1",
        "corpus_sha256": corpus_sha,
        "split_config_sha256": split_config_sha,
        "counts": {
            name: sum(value == name for value in split_tasks.values())
            for name in ("train", "dev", "holdout")
        },
        "families": splits,
        "tasks": split_tasks,
    }
    stats = {
        "task_count": len(tasks),
        "family_count": len(families),
        "counts_by_archetype": dict(sorted(counts_archetype.items())),
        "counts_by_difficulty": dict(sorted(counts_difficulty.items())),
        "counts_by_prototype": dict(sorted(counts_prototype.items())),
        "counts_by_split": split_manifest["counts"],
        "prototype_counts_by_split": {
            name: len(
                {
                    tasks[task_id]["prototype_id"]
                    for task_id, value in split_tasks.items()
                    if value == name
                }
            )
            for name in ("train", "dev", "holdout")
        },
        "cross_split_prototype_collisions": 0,
        "behavior_signature_count": len(behavior_uses),
        "behavior_signature_collisions": {
            "within_split": sum(len(uses) > 1 for uses in behavior_uses.values()),
            "cross_split": 0,
        },
        "instruction_template_count": template_count,
        "distractor_distribution": dict(sorted(distractor_counts.items())),
        "multi_invoice_count": counts_archetype["multi-invoice"],
        "read_only_count": counts_archetype["read-only"],
        "wrong_target_count": counts_archetype["wrong-target"],
        "diversity_exceptions": exceptions,
    }
    for name, value in (
        ("corpus-manifest.json", manifest),
        ("split-manifest.json", split_manifest),
        ("corpus-stats.json", stats),
    ):
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return manifest


def generate(root: Path = PHASE10) -> dict[str, Any]:
    output = root / "generated"
    output.parent.mkdir(parents=True, exist_ok=True)
    staged_parent = Path(tempfile.mkdtemp(prefix=".phase10-stage-", dir=output.parent))
    staged = staged_parent / "generated"
    backup = staged_parent / "previous"
    try:
        manifest = build_corpus(root, staged)
        had_previous = output.exists()
        if had_previous:
            os.replace(output, backup)
        try:
            os.replace(staged, output)
        except OSError:
            if had_previous:
                os.replace(backup, output)
            raise
        return manifest
    finally:
        shutil.rmtree(staged_parent, ignore_errors=True)


def check(root: Path = PHASE10) -> dict[str, Any]:
    output = root / "generated"
    with tempfile.TemporaryDirectory(prefix="phase10-check-") as temporary:
        expected_root = Path(temporary) / "generated"
        expected = build_corpus(root, expected_root)
        expected_files = {
            path.relative_to(expected_root): path.read_bytes()
            for path in expected_root.rglob("*")
            if path.is_file()
        }
    actual_files = (
        {
            path.relative_to(output): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file()
        }
        if output.is_dir()
        else {}
    )
    missing = sorted(str(path) for path in expected_files.keys() - actual_files.keys())
    unexpected = sorted(str(path) for path in actual_files.keys() - expected_files.keys())
    changed = sorted(
        str(path)
        for path in expected_files.keys() & actual_files.keys()
        if expected_files[path] != actual_files[path]
    )
    return {
        "status": "CURRENT" if not (missing or unexpected or changed) else "STALE",
        "task_count": expected["task_count"],
        "corpus_sha256": expected["corpus_sha256"],
        "missing": missing,
        "unexpected": unexpected,
        "changed": changed,
    }


def load_phase10_corpus(
    split: Literal["train", "dev", "holdout"], root: Path = PHASE10
) -> dict[str, Any]:
    if check(root)["status"] != "CURRENT":
        raise ValueError("Phase 10 corpus snapshot is stale")
    manifest_path = root / "generated" / "corpus-manifest.json"
    split_path = root / "generated" / "split-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignment = json.loads(split_path.read_text(encoding="utf-8"))
    if manifest["corpus_sha256"] != assignment["corpus_sha256"]:
        raise ValueError("Phase 10 corpus and split lineage disagree")
    if set(manifest["tasks"]) != set(assignment["tasks"]):
        raise ValueError("Phase 10 split task IDs do not match corpus")
    if sum(assignment["counts"].values()) != manifest["task_count"]:
        raise ValueError("Phase 10 split counts do not match corpus")
    task_ids = sorted(task_id for task_id, name in assignment["tasks"].items() if name == split)
    return {
        "task_ids": task_ids,
        "family_ids": assignment["families"][split],
        "prototype_ids": sorted(
            {manifest["tasks"][task_id]["prototype_id"] for task_id in task_ids}
        ),
        "behavior_signatures": {
            task_id: manifest["tasks"][task_id]["behavior_signature_sha256"] for task_id in task_ids
        },
        "corpus_sha256": manifest["corpus_sha256"],
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "family_specs_sha256": manifest["family_specs_sha256"],
        "generator_version": manifest["generator_version"],
    }
