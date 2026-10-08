"""Frozen training sources and independently authored evaluation compositions."""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

from agentops_demo.benchmark.phase10_variants import (
    ARCHETYPES,
    PHASE10,
    _normalized_instruction,
    behavior_signature_sha256,
    load_phase10_corpus,
)
from agentops_demo.contracts.scenario import Scenario
from agentops_demo.taskify.harbor_renderer import render_harbor_task
from agentops_demo.taskify.integrity import scenario_sha256
from agentops_demo.validation.scenario import load_scenario
from rl.phase8c.execution_suite import derive_execution_suite
from rl.phase12.common import REPO, fingerprint, read, write
from rl.phase12.config import load

SPECS = REPO / "benchmarks/billing/phase12/prototypes.json"
FROZEN = REPO / "benchmarks/billing/phase12/manifest.json"
ACTION_ARCHETYPES = (
    "normal-refund",
    "disputed-refund",
    "multi-invoice",
    "duplicate-action",
    "wrong-target",
)


def make_scenario(spec: dict[str, Any], index: int) -> Scenario:
    ident = f"{spec['id']}-v{index:02}"
    number = int(fingerprint([spec["id"], index])[:8], 16)
    target, second = f"inv-{number}-A", f"inv-{number}-B"
    customer = f"customer-{number}"
    archetype = spec["archetype"]
    state: dict[str, Any] = {
        "customers": [{"id": customer, "status": "active"}],
        "invoices": [],
        "refunds": [],
        "escalations": [],
    }

    def invoice(key: str, status: str, prior: str | None = None) -> None:
        state["invoices"].append(
            {
                "id": key,
                "customer_id": customer,
                "amount": ["19.00", "49.50", "85.00", "120.00", "249.95"][index],
                "currency": "USD",
                "status": status,
            }
        )
        if prior:
            state[prior].append(
                {"id": f"prior-{key}", "invoice_id": key, "reason": "Previous request"}
            )

    invariants: list[dict[str, str]] = []
    allowed: list[str] = []
    inspections = [target]
    if archetype != "missing-invoice":
        status = {
            "disputed-refund": "disputed",
            "already-refunded": "refunded",
            "already-escalated": "disputed",
        }.get(archetype, "paid")
        prior = {"already-refunded": "refunds", "already-escalated": "escalations"}.get(archetype)
        invoice(target, status, prior)
        if archetype in ACTION_ARCHETYPES:
            escalate = archetype == "disputed-refund"
            invariants.extend(
                [
                    {"type": "must_escalate" if escalate else "must_refund", "invoice_id": target},
                    {"type": "no_refund" if escalate else "no_escalation", "invoice_id": target},
                ]
            )
            allowed.append(target)
        else:
            invariants.append({"type": "unchanged", "invoice_id": target})
    if archetype == "multi-invoice":
        invoice(second, "disputed")
        inspections.append(second)
        allowed.append(second)
        invariants.extend(
            [
                {"type": "must_escalate", "invoice_id": second},
                {"type": "no_refund", "invoice_id": second},
            ]
        )
    companions = []
    for n, descriptor in enumerate(spec["companions"]):
        key = f"inv-{number}-C{n}"
        invoice(key, descriptor["status"], descriptor.get("prior"))
        companions.append(key)
        inspections.append(key)
        invariants.append({"type": "unchanged", "invoice_id": key})
    if index % 2:
        state["invoices"].reverse()
    instruction = spec["instructions"][index].format(
        target=target, second=second, companions=", ".join(companions)
    )
    return Scenario.model_validate(
        {
            "schema_version": "1",
            "id": ident,
            "instruction": instruction,
            "provenance": {"source": {"kind": "synthetic"}},
            "initial_state": state,
            "expected_invariants": invariants,
            "benchmark": {
                "category": "billing",
                "archetype": archetype,
                "difficulty": "compound",
                "mutation_policy": {"allowed_targets": allowed},
                "trajectory": {
                    "required_inspections": inspections,
                    "require_inspection_before_mutation": True,
                    "max_write_attempts_per_invoice": 1,
                },
                "generation": {
                    "generator": "billing-phase12-v1",
                    "family_id": spec["id"],
                    "prototype_id": spec["id"],
                    "prototype_sha256": fingerprint(spec),
                    "variant_index": index,
                    "dimensions": {
                        "composition": spec["composition"],
                        "entity_order": "reverse" if index % 2 else "original",
                    },
                },
            },
        }
    )


def scenarios() -> dict[str, dict[str, Scenario]]:
    train = load_phase10_corpus("train")
    if train["corpus_sha256"] != load()["phase10_corpus_sha256"]:
        raise ValueError("Phase 10 source corpus changed")
    result = {
        "train": {
            key: load_scenario(PHASE10 / "generated/scenarios" / key / "scenario.yaml")
            for key in train["task_ids"]
        },
        "dev": {},
        "test": {},
    }
    specs = read(SPECS)
    if len(specs) != 36 or len({s["id"] for s in specs}) != 36:
        raise ValueError("36 independent evaluation prototypes required")
    for spec in specs:
        if spec["split"] not in ("dev", "test") or len(spec["instructions"]) != 5:
            raise ValueError("invalid evaluation prototype")
        for index in range(5):
            scenario = make_scenario(spec, index)
            result[spec["split"]][scenario.id] = scenario
    return result


def metadata(scenario: Scenario) -> dict[str, Any]:
    benchmark = scenario.benchmark
    assert benchmark and benchmark.generation
    return {
        "family_id": benchmark.generation.family_id,
        "prototype_id": benchmark.generation.prototype_id,
        "archetype": benchmark.archetype,
        "difficulty": benchmark.difficulty,
        "scenario_sha256": scenario_sha256(scenario),
        "behavior_signature_sha256": behavior_signature_sha256(scenario),
        "instruction_sha256": fingerprint(_normalized_instruction(scenario.instruction)),
        "expected_write_targets": benchmark.mutation_policy.allowed_targets,
        "required_tool_calls": len(benchmark.trajectory.required_inspections)
        + sum(i.type in {"must_refund", "must_escalate"} for i in scenario.expected_invariants),
        "expected_operations": {
            i.invoice_id: "refund_invoice" if i.type == "must_refund" else "escalate_dispute"
            for i in scenario.expected_invariants
            if i.type in {"must_refund", "must_escalate"}
        },
    }


def manifest(data: dict[str, dict[str, Scenario]]) -> dict[str, Any]:
    roles = {
        role: {key: metadata(s) for key, s in sorted(tasks.items())} for role, tasks in data.items()
    }
    if {role: len(tasks) for role, tasks in roles.items()} != {"train": 160, "dev": 90, "test": 90}:
        raise ValueError("Phase 12 requires 160/90/90 tasks")
    for role in ("dev", "test"):
        if Counter(v["archetype"] for v in roles[role].values()) != dict.fromkeys(ARCHETYPES, 10):
            raise ValueError("evaluation must cover all nine archetypes equally")
        if len({v["prototype_id"] for v in roles[role].values()}) != 18:
            raise ValueError("evaluation requires 18 independent prototypes")
    names = list(roles)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            for key in (
                "prototype_id",
                "family_id",
                "behavior_signature_sha256",
                "instruction_sha256",
            ):
                overlap = {v[key] for v in roles[left].values()} & {
                    v[key] for v in roles[right].values()
                }
                if overlap:
                    raise ValueError(f"{key} leaks between {left} and {right}: {sorted(overlap)}")
    value = {
        "schema_version": "1",
        "prototype_specs_sha256": fingerprint(read(SPECS)),
        "roles": roles,
    }
    return {**value, "corpus_sha256": fingerprint(value)}


def check() -> dict[str, Any]:
    value = manifest(scenarios())
    if value != read(FROZEN):
        raise ValueError("Phase 12 corpus differs from frozen manifest")
    return value


def transport_manifest(frozen: dict[str, Any]) -> dict[str, Any]:
    """Retain final-test lineage and bounds, never answer-bearing target metadata."""
    allowed = {
        "family_id",
        "prototype_id",
        "archetype",
        "difficulty",
        "scenario_sha256",
        "behavior_signature_sha256",
        "instruction_sha256",
        "required_tool_calls",
    }
    return {
        **frozen,
        "redacted_roles": ["test"],
        "roles": {
            **frozen["roles"],
            "test": {
                key: {field: value for field, value in task.items() if field in allowed}
                for key, task in frozen["roles"]["test"].items()
            },
        },
    }


def representatives(meta: dict[str, Any]) -> list[str]:
    groups: dict[str, list[str]] = {}
    for key, task in meta.items():
        groups.setdefault(task["family_id"], []).append(key)
    return [sorted(keys)[0] for _, keys in sorted(groups.items())]


def render(output: Path, roles: tuple[str, ...]) -> dict[str, Any]:
    """Render only requested roles. A training payload never contains test scenarios."""
    if roles not in (("train", "dev"), ("test",), ("train", "dev", "test")):
        raise ValueError("unsupported corpus payload roles")
    frozen = check()
    output.mkdir(parents=True, exist_ok=False)
    data = scenarios()
    for role in roles:
        catalog = output / role / "scenarios"
        canonical = output / role / "canonical"
        for key, scenario in data[role].items():
            asyncio.run(render_harbor_task(scenario, canonical / key))
            source = catalog / key / "scenario.yaml"
            source.parent.mkdir(parents=True)
            source.write_bytes((canonical / key / "scenario.yaml").read_bytes())
        derive_execution_suite(
            output / role / "execution", catalog=catalog, canonical_tasks=canonical
        )
        write(output / role / "metadata.json", frozen["roles"][role])
    return frozen
