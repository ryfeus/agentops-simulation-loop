from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agentops_demo.benchmark.catalog import CatalogRenderError, load_catalog, render_catalog_sync
from agentops_demo.taskify.integrity import harbor_task_sha256, validate_harbor_task
from scripts.benchmark_billing import _render_manifest, main

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "benchmarks" / "billing" / "scenarios"
REVISION = "a" * 40


def _copy_scenario(catalog: Path, scenario_id: str) -> Path:
    destination = catalog / scenario_id
    shutil.copytree(SCENARIOS / scenario_id, destination)
    return destination / "scenario.yaml"


def _manifest(catalog: Path, tasks: Path) -> dict[str, object]:
    return _render_manifest(load_catalog(catalog), tasks, REVISION)


def test_billing_catalog_is_declarative_and_renderable(tmp_path: Path) -> None:
    entries = load_catalog()
    expected_ids = {entry.scenario.id for entry in entries}

    assert {
        entry.scenario.benchmark.archetype for entry in entries if entry.scenario.benchmark
    } >= {
        "disputed-refund",
        "normal-refund",
        "read-only",
        "missing-invoice",
        "multi-invoice",
        "wrong-target",
        "duplicate-action",
    }
    assert all(entry.scenario.observed_failure is None for entry in entries)
    assert all(
        entry.scenario.benchmark is not None
        and entry.scenario.benchmark.trajectory.required_inspections
        for entry in entries
    )

    tasks = tmp_path / "tasks"
    rendered = render_catalog_sync(SCENARIOS, tasks)

    assert {entry.scenario.id for entry in rendered} == expected_ids
    assert {task.name for task in tasks.iterdir() if task.is_dir()} == expected_ids
    for task_id in expected_ids:
        validate_harbor_task(tasks / task_id)


def test_rendered_task_validation_rejects_invalid_instruction_and_toml(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    render_catalog_sync(SCENARIOS, tasks)
    task = tasks / load_catalog()[0].scenario.id

    (task / "instruction.md").write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"instruction\.md"):
        validate_harbor_task(task)

    (task / "instruction.md").write_text("Restored instruction\n", encoding="utf-8")
    (task / "task.toml").write_text("[task\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"task\.toml"):
        validate_harbor_task(task)


def test_catalog_render_and_manifest_identities_are_deterministic(tmp_path: Path) -> None:
    first_tasks = tmp_path / "first" / "tasks"
    second_tasks = tmp_path / "second" / "tasks"
    render_catalog_sync(SCENARIOS, first_tasks)
    render_catalog_sync(SCENARIOS, second_tasks)

    first = _manifest(SCENARIOS, first_tasks)
    second = _manifest(SCENARIOS, second_tasks)

    assert first["tasks"] == second["tasks"]
    assert first["suite_sha256"] == second["suite_sha256"]


def test_source_change_updates_scenario_task_and_suite_identities(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    scenario_path = _copy_scenario(catalog, "paid-refund-direct")
    first_tasks = tmp_path / "first" / "tasks"
    render_catalog_sync(catalog, first_tasks)
    first = _manifest(catalog, first_tasks)

    original = scenario_path.read_text(encoding="utf-8")
    scenario_path.write_text(
        original.replace(
            "Refund invoice inv-201 because I was charged incorrectly.",
            "Issue a refund for inv-201.",
        ),
        encoding="utf-8",
    )
    second_tasks = tmp_path / "second" / "tasks"
    render_catalog_sync(catalog, second_tasks)
    second = _manifest(catalog, second_tasks)

    task_id = "paid-refund-direct"
    first_task = first["tasks"][task_id]  # type: ignore[index]
    second_task = second["tasks"][task_id]  # type: ignore[index]
    assert first_task["scenario_sha256"] != second_task["scenario_sha256"]  # type: ignore[index]
    assert first_task["harbor_task_sha256"] != second_task["harbor_task_sha256"]  # type: ignore[index]
    assert first["suite_sha256"] != second["suite_sha256"]


def test_rerender_removes_stale_tasks_and_invalid_catalog_preserves_previous_tree(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog"
    _copy_scenario(catalog, "paid-refund-direct")
    stale_path = _copy_scenario(catalog, "open-refund-direct")
    tasks = tmp_path / "output" / "tasks"
    render_catalog_sync(catalog, tasks)

    shutil.rmtree(stale_path.parent)
    render_catalog_sync(catalog, tasks)
    assert {task.name for task in tasks.iterdir()} == {"paid-refund-direct"}
    previous_digest = harbor_task_sha256(tasks / "paid-refund-direct")

    invalid_path = _copy_scenario(catalog, "open-refund-direct")
    invalid_path.write_text("not: a scenario\n", encoding="utf-8")
    with pytest.raises(CatalogRenderError) as error:
        render_catalog_sync(catalog, tasks)

    assert error.value.catalog_count == 2
    assert error.value.rendered_count == 0
    assert harbor_task_sha256(tasks / "paid-refund-direct") == previous_digest

    assert main(["--catalog", str(catalog), "--output", str(tasks.parent), "--render-only"]) == 1
    summary = json.loads((tasks.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"
    assert summary["catalog_count"] == 2
    assert summary["rendered_count"] == 0
    assert summary["failed_count"] == 1
    assert summary["output"] == str(tasks.parent)
