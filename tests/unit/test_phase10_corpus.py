"""CPU-only contract checks for the deterministic Phase 10 corpus."""

from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from agentops_demo.benchmark.catalog import render_catalog_sync
from agentops_demo.benchmark.phase10_acceptance import (
    _negative_kwargs,
    _proof_performed,
    render_smoke,
)
from agentops_demo.benchmark.phase10_variants import (
    PHASE10,
    FamilySpec,
    behavior_signature_sha256,
    build_corpus,
    check,
    load_families,
    load_phase10_corpus,
    make_variant,
)
from agentops_demo.harbor.phase10_negative_worker import run as run_negative_worker
from agentops_demo.taskify.integrity import harbor_task_sha256, scenario_sha256
from agentops_demo.validation.scenario import load_scenario
from rl.phase8c.execution_suite import derive_execution_suite


def _prototype(family: FamilySpec):
    return load_scenario(Path("benchmarks/billing/scenarios") / family.prototype / "scenario.yaml")


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "phase10"
    family_root = root / "families"
    family_root.mkdir(parents=True)
    selected = {
        family.id: family
        for family in load_families()
        if family.id in {"normal-paid-direct", "readonly-paid-status"}
    }
    for family in selected.values():
        payload = family.model_dump(mode="json")
        payload["variant_count"] = 2
        (family_root / f"{family.id}.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    config = {
        "schema_version": "1",
        "generator_version": "billing-phase10-v1",
        "global_seed": 20260925,
        "family_count": 2,
        "task_count": 4,
        "split_counts": {"train": 2, "dev": 2, "holdout": 0},
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    split = {
        "schema_version": "1",
        "families": {
            "train": ["normal-paid-direct"],
            "dev": ["readonly-paid-status"],
            "holdout": [],
        },
    }
    (root / "split.yaml").write_text(yaml.safe_dump(split, sort_keys=False))
    return root


def test_snapshot_and_split_lineage_are_current() -> None:
    result = check()
    assert result["status"] == "CURRENT"
    assert result["task_count"] == 200
    train = load_phase10_corpus("train")
    dev = load_phase10_corpus("dev")
    holdout = load_phase10_corpus("holdout")
    assert (len(train["task_ids"]), len(dev["task_ids"]), len(holdout["task_ids"])) == (
        160,
        20,
        20,
    )
    assert train["corpus_sha256"] == dev["corpus_sha256"] == holdout["corpus_sha256"]
    assert not (set(train["family_ids"]) & set(holdout["family_ids"]))
    assert not (set(train["prototype_ids"]) & set(dev["prototype_ids"]))
    assert not (set(train["prototype_ids"]) & set(holdout["prototype_ids"]))
    assert not (set(dev["prototype_ids"]) & set(holdout["prototype_ids"]))
    assert (
        len(train["prototype_ids"]),
        len(dev["prototype_ids"]),
        len(holdout["prototype_ids"]),
    ) == (16, 4, 3)
    assert len(train["behavior_signatures"]) == 160
    stats = json.loads((PHASE10 / "generated" / "corpus-stats.json").read_text())
    assert stats["behavior_signature_collisions"]["cross_split"] == 0
    assert stats["behavior_signature_collisions"]["within_split"] > 0


def test_anchor_serialization_omits_optional_generation() -> None:
    anchor = load_scenario("benchmarks/billing/scenarios/paid-refund-direct/scenario.yaml")
    assert anchor.benchmark is not None
    assert anchor.benchmark.generation is None
    assert "generation" not in anchor.benchmark.model_dump(mode="json")


def test_typed_transform_updates_all_invoice_references() -> None:
    family = next(item for item in load_families() if item.id == "multi-paid-disputed-order")
    prototype = _prototype(family)
    variant = make_variant(family, prototype, 0)
    known = {invoice.id for invoice in variant.initial_state.invoices}
    assert {item.invoice_id for item in variant.expected_invariants} <= known
    assert set(variant.benchmark.mutation_policy.allowed_targets) <= known
    assert set(variant.benchmark.trajectory.required_inspections) <= known
    assert all(
        invoice.customer_id in {item.id for item in variant.initial_state.customers}
        for invoice in variant.initial_state.invoices
    )
    assert variant.benchmark.generation.prototype_sha256 == scenario_sha256(prototype)
    assert variant.id == make_variant(family, prototype, 0).id
    unrelated = next(item for item in load_families() if item.id == "wrong-similar-id")
    make_variant(unrelated, _prototype(unrelated), 0)
    assert scenario_sha256(variant) == scenario_sha256(make_variant(family, prototype, 0))
    changed = variant.model_dump(mode="json")
    changed["benchmark"]["generation"]["prototype_sha256"] = "a" * 64
    assert scenario_sha256(variant) != scenario_sha256(type(variant).model_validate(changed))


def test_missing_invoice_is_the_only_valid_absent_inspection() -> None:
    family = next(item for item in load_families() if item.id == "missing-urgent-request")
    scenario = make_variant(family, _prototype(family), 0)
    known = {item.id for item in scenario.initial_state.invoices}
    assert len(set(scenario.benchmark.trajectory.required_inspections) - known) == 1
    assert all(item.invoice_id in known for item in scenario.expected_invariants)


@pytest.mark.parametrize(
    ("family_id", "record_kind"),
    [
        ("refunded-direct-repeat", "refunds"),
        ("escalated-direct-repeat", "escalations"),
    ],
)
def test_terminal_state_records_follow_renamed_invoices(family_id: str, record_kind: str) -> None:
    family = next(item for item in load_families() if item.id == family_id)
    scenario = make_variant(family, _prototype(family), 0)
    known = {invoice.id for invoice in scenario.initial_state.invoices}
    records = getattr(scenario.initial_state, record_kind)
    assert records and all(record.invoice_id in known for record in records)
    assert all(
        record.id.startswith("refund-") or record.id.startswith("escalation-") for record in records
    )
    assert all(item.type == "unchanged" for item in scenario.expected_invariants)


def test_family_lint_rejects_unknown_placeholder_and_strategy() -> None:
    payload = load_families()[0].model_dump(mode="json")
    bad = copy.deepcopy(payload)
    bad["instruction_templates"][0] = "Refund {unknown}."
    with pytest.raises(ValueError, match="placeholder"):
        FamilySpec.model_validate(bad)
    bad = copy.deepcopy(payload)
    bad["negative_strategy"] = "guess"
    with pytest.raises(ValueError, match="negative strategy"):
        FamilySpec.model_validate(bad)
    bad = copy.deepcopy(payload)
    bad["dimensions"]["target_amount"] = []
    with pytest.raises(ValueError, match="nonempty"):
        FamilySpec.model_validate(bad)


def test_four_task_fixture_flows_to_harbor_execution_suite(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    manifest = build_corpus(root, root / "generated")
    assert manifest["task_count"] == 4
    assert check(root)["status"] == "CURRENT"
    split = load_phase10_corpus("train", root)
    assert len(split["task_ids"]) == 2
    assert render_smoke(root)["task_count"] == 4
    tasks = tmp_path / "tasks"
    entries = render_catalog_sync(root / "generated" / "scenarios", tasks)
    assert len(entries) == 4
    suite = derive_execution_suite(
        tmp_path / "execution",
        catalog=root / "generated" / "scenarios",
        canonical_tasks=tasks,
    )
    assert suite.manifest["task_count"] == 4
    assert suite.manifest["execution_profile"] == "trl-harbor-shared-verifier"


def test_split_rejects_duplicate_family(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    split_path = root / "split.yaml"
    split = yaml.safe_load(split_path.read_text())
    split["families"]["dev"].append("normal-paid-direct")
    split_path.write_text(yaml.safe_dump(split))
    with pytest.raises(ValueError, match="missing, duplicate, or unknown"):
        build_corpus(root, root / "generated")


def test_prototype_overlap_rejected_with_families_and_splits(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    extra = next(f for f in load_families() if f.id == "normal-paid-exclusion")
    payload = extra.model_dump(mode="json")
    payload["variant_count"] = 2
    (root / "families" / f"{extra.id}.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    config_path = root / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["family_count"] = 3
    config["task_count"] = 6
    config["split_counts"]["holdout"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    split_path = root / "split.yaml"
    split = yaml.safe_load(split_path.read_text())
    split["families"]["holdout"] = [extra.id]
    split_path.write_text(yaml.safe_dump(split, sort_keys=False))
    with pytest.raises(
        ValueError,
        match=(
            r"prototype paid-refund-direct crosses splits: "
            r"holdout: normal-paid-exclusion; train: normal-paid-direct"
        ),
    ):
        build_corpus(root, root / "generated")


def test_same_split_prototype_reuse_is_allowed(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    extra = next(f for f in load_families() if f.id == "normal-paid-exclusion")
    payload = extra.model_dump(mode="json")
    payload["variant_count"] = 2
    (root / "families" / f"{extra.id}.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    config_path = root / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["family_count"] = 3
    config["task_count"] = 6
    config["split_counts"]["train"] = 4
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    split_path = root / "split.yaml"
    split = yaml.safe_load(split_path.read_text())
    split["families"]["train"].append(extra.id)
    split_path.write_text(yaml.safe_dump(split, sort_keys=False))
    manifest = build_corpus(root, root / "generated")
    assert manifest["task_count"] == 6
    stats = json.loads((root / "generated" / "corpus-stats.json").read_text())
    assert stats["prototype_counts_by_split"]["train"] == 1


def test_split_only_changes_lineage_not_scenarios(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    first = build_corpus(root, root / "first")
    before = {
        p.relative_to(root / "first" / "scenarios"): p.read_bytes()
        for p in (root / "first" / "scenarios").rglob("*.yaml")
    }
    split_path = root / "split.yaml"
    split = yaml.safe_load(split_path.read_text())
    split["families"]["train"], split["families"]["dev"] = (
        split["families"]["dev"],
        split["families"]["train"],
    )
    split_path.write_text(yaml.safe_dump(split, sort_keys=False))
    second = build_corpus(root, root / "second")
    after = {
        p.relative_to(root / "second" / "scenarios"): p.read_bytes()
        for p in (root / "second" / "scenarios").rglob("*.yaml")
    }
    assert before == after
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert first["family_specs_sha256"] == second["family_specs_sha256"]
    assert (root / "first" / "split-manifest.json").read_bytes() != (
        root / "second" / "split-manifest.json"
    ).read_bytes()
    rendered_hashes = []
    suite_hashes = []
    for name in ("first", "second"):
        rendered = tmp_path / f"{name}-rendered"
        entries = render_catalog_sync(root / name / "scenarios", rendered)
        rendered_hashes.append(
            {
                entry.scenario.id: harbor_task_sha256(rendered / entry.scenario.id)
                for entry in entries
            }
        )
        suite = derive_execution_suite(
            tmp_path / f"{name}-execution",
            catalog=root / name / "scenarios",
            canonical_tasks=rendered,
        )
        suite_hashes.append(suite.execution_suite_sha256)
    assert rendered_hashes[0] == rendered_hashes[1]
    assert suite_hashes[0] == suite_hashes[1]


def test_behavior_signature_ignores_cosmetics_but_tracks_policy() -> None:
    family = next(f for f in load_families() if f.id == "normal-paid-direct")
    original = make_variant(family, _prototype(family), 0)
    baseline = behavior_signature_sha256(original)
    cosmetic = original.model_dump(mode="json")
    cosmetic["instruction"] = "Another wording for the same billing request."
    cosmetic["initial_state"]["invoices"][0]["amount"] = "249.95"
    cosmetic["initial_state"]["invoices"].reverse()
    cosmetic["benchmark"]["generation"]["variant_index"] = 999
    customer_ids = {
        customer["id"]: f"customer-renamed-{index}"
        for index, customer in enumerate(cosmetic["initial_state"]["customers"])
    }
    invoice_ids = {
        invoice["id"]: f"invoice-renamed-{index}"
        for index, invoice in enumerate(cosmetic["initial_state"]["invoices"])
    }
    for customer in cosmetic["initial_state"]["customers"]:
        customer["id"] = customer_ids[customer["id"]]
    for invoice in cosmetic["initial_state"]["invoices"]:
        invoice["id"] = invoice_ids[invoice["id"]]
        invoice["customer_id"] = customer_ids[invoice["customer_id"]]
    for kind in ("refunds", "escalations"):
        for record in cosmetic["initial_state"][kind]:
            record["invoice_id"] = invoice_ids[record["invoice_id"]]
    for invariant in cosmetic["expected_invariants"]:
        invariant["invoice_id"] = invoice_ids[invariant["invoice_id"]]
    policy = cosmetic["benchmark"]["mutation_policy"]
    policy["allowed_targets"] = [invoice_ids[item] for item in policy["allowed_targets"]]
    trajectory = cosmetic["benchmark"]["trajectory"]
    trajectory["required_inspections"] = [
        invoice_ids[item] for item in trajectory["required_inspections"]
    ]
    assert behavior_signature_sha256(type(original).model_validate(cosmetic)) == baseline
    for change in (
        "status",
        "allowed",
        "required",
        "invariant",
        "trajectory",
        "refund_record",
        "escalation_record",
        "archetype",
        "customer_status",
        "currency",
    ):
        semantic = original.model_dump(mode="json")
        invoice_id = semantic["initial_state"]["invoices"][0]["id"]
        if change == "status":
            semantic["initial_state"]["invoices"][0]["status"] = "open"
        elif change == "allowed":
            semantic["benchmark"]["mutation_policy"]["allowed_targets"] = []
        elif change == "required":
            semantic["benchmark"]["trajectory"]["required_inspections"] = []
        elif change == "invariant":
            semantic["expected_invariants"].append(
                {"type": "no_escalation", "invoice_id": invoice_id}
            )
        elif change == "refund_record":
            semantic["initial_state"]["refunds"].append(
                {"id": "refund-new", "invoice_id": invoice_id, "reason": "existing"}
            )
        elif change == "escalation_record":
            semantic["initial_state"]["escalations"].append(
                {"id": "escalation-new", "invoice_id": invoice_id, "reason": "existing"}
            )
        elif change == "archetype":
            semantic["benchmark"]["archetype"] = "read-only"
        elif change == "customer_status":
            semantic["initial_state"]["customers"][0]["status"] = "inactive"
        elif change == "currency":
            semantic["initial_state"]["invoices"][0]["currency"] = "EUR"
        else:
            semantic["benchmark"]["trajectory"]["require_inspection_before_mutation"] = False
        assert behavior_signature_sha256(type(original).model_validate(semantic)) != baseline


def test_cross_split_behavior_signature_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_root(tmp_path)
    monkeypatch.setattr(
        "agentops_demo.benchmark.phase10_variants.behavior_signature_sha256",
        lambda _scenario: "a" * 64,
    )
    with pytest.raises(
        ValueError,
        match=r"behavior signature a+ crosses splits:.*task=.*family=.*prototype=.*split=dev",
    ):
        build_corpus(root, root / "generated")


def test_cross_split_instruction_leakage_is_rejected(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    normal = yaml.safe_load((root / "families" / "normal-paid-direct.yaml").read_text())
    readonly_path = root / "families" / "readonly-paid-status.yaml"
    readonly = yaml.safe_load(readonly_path.read_text())
    readonly["instruction_templates"] = normal["instruction_templates"]
    readonly_path.write_text(yaml.safe_dump(readonly, sort_keys=False))
    with pytest.raises(ValueError, match="instruction leaks"):
        build_corpus(root, root / "generated")


def test_family_diversity_rejects_cosmetic_templates(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    family_path = root / "families" / "normal-paid-direct.yaml"
    family = yaml.safe_load(family_path.read_text())
    for key, pool in family["dimensions"].items():
        family["dimensions"][key] = pool[:1]
    family_path.write_text(yaml.safe_dump(family, sort_keys=False))
    with pytest.raises(ValueError, match="diversity gate"):
        build_corpus(root, root / "generated")


def test_unknown_prototype_is_rejected(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    family_path = root / "families" / "normal-paid-direct.yaml"
    family = yaml.safe_load(family_path.read_text())
    family["prototype"] = "unknown-prototype"
    family_path.write_text(yaml.safe_dump(family, sort_keys=False))
    with pytest.raises(ValueError, match="unknown prototype"):
        build_corpus(root, root / "generated")


def test_negative_strategy_targets_protected_invoice() -> None:
    task_id = "wrong-cross-customer-v00"
    kwargs = _negative_kwargs(task_id, "refund_wrong_target", PHASE10)
    scenario = load_scenario(PHASE10 / "generated" / "scenarios" / task_id / "scenario.yaml")
    assert kwargs["wrong_invoice"] not in scenario.benchmark.mutation_policy.allowed_targets
    assert kwargs["target_invoice"] in scenario.benchmark.mutation_policy.allowed_targets
    assert json.loads(kwargs["inspection_ids"]) == [kwargs["target_invoice"]]


def test_negative_worker_records_duplicate_attempt(tmp_path: Path) -> None:
    database = tmp_path / "billing.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE invoices (id TEXT PRIMARY KEY, status TEXT);"
            "CREATE TABLE refunds (id TEXT PRIMARY KEY, invoice_id TEXT UNIQUE, reason TEXT);"
            "CREATE TABLE escalations (id TEXT PRIMARY KEY, invoice_id TEXT UNIQUE, reason TEXT);"
            "INSERT INTO invoices VALUES ('inv-1', 'paid');"
        )
    result = run_negative_worker(
        {
            "strategy": "duplicate_refund",
            "target_invoice": "inv-1",
            "wrong_invoice": "",
            "inspection_ids": ["inv-1"],
        },
        database_path=database,
        artifact_path=tmp_path / "agent-run.json",
        evidence_path=tmp_path / "negative-execution.json",
    )
    assert [call["name"] for call in result["tool_calls"]] == [
        "get_invoice",
        "refund_invoice",
        "refund_invoice",
    ]
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM refunds").fetchone() == (1,)
    assert json.loads((tmp_path / "negative-execution.json").read_text()) == result


def test_negative_proof_requires_the_declared_wrong_action() -> None:
    kwargs = {
        "strategy": "refund_wrong_target",
        "target_invoice": "inv-1",
        "wrong_invoice": "inv-2",
        "inspection_ids": '["inv-1"]',
    }
    correct_proof = {
        "strategy": "refund_wrong_target",
        "tool_calls": [
            {"name": "get_invoice", "arguments": {"invoice_id": "inv-1"}},
            {"name": "refund_invoice", "arguments": {"invoice_id": "inv-2"}},
        ],
    }
    assert _proof_performed(correct_proof, kwargs)
    correct_proof["tool_calls"].pop()
    assert not _proof_performed(correct_proof, kwargs)
