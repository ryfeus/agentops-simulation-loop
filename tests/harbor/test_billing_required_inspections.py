from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
from pathlib import Path
from types import ModuleType

from agentops_demo.benchmark.catalog import load_catalog
from agentops_demo.taskify.harbor_renderer import render_harbor_task

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "benchmarks" / "billing" / "scenarios"


def _task(tmp_path: Path, scenario_id: str) -> Path:
    scenario = next(
        entry.scenario for entry in load_catalog(SCENARIOS) if entry.scenario.id == scenario_id
    )
    return asyncio.run(render_harbor_task(scenario, tmp_path / scenario_id))


def _verifier(task: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"verifier_{task.name}", task / "tests" / "verify.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evaluate(
    task: Path,
    tmp_path: Path,
    calls: list[dict[str, object]],
    mutations: list[str] | None = None,
) -> dict[str, float]:
    database = tmp_path / f"billing-{len(list(tmp_path.glob('billing-*.db')))}.db"
    with sqlite3.connect(database) as connection:
        connection.executescript((task / "environment" / "seed.sql").read_text(encoding="utf-8"))
        for mutation in mutations or []:
            connection.execute(mutation)
    result = tmp_path / "agent-run.json"
    result.write_text(json.dumps({"tool_calls": calls}), encoding="utf-8")
    verifier = _verifier(task)
    verifier.AGENT_RESULT_PATH = result
    rewards, _ = verifier.evaluate_database(database)
    return rewards


def test_readonly_task_requires_inspection(tmp_path: Path) -> None:
    task = _task(tmp_path, "status-paid-readonly")

    assert _evaluate(task, tmp_path, [])["reward"] == 0.0
    assert (
        _evaluate(
            task,
            tmp_path,
            [{"name": "get_invoice", "arguments": {"invoice_id": "inv-201"}}],
        )["reward"]
        == 1.0
    )


def test_already_refunded_task_requires_inspection(tmp_path: Path) -> None:
    task = _task(tmp_path, "already-refunded-direct")

    assert _evaluate(task, tmp_path, [])["reward"] == 0.0
    assert (
        _evaluate(
            task,
            tmp_path,
            [{"name": "get_invoice", "arguments": {"invoice_id": "inv-301"}}],
        )["reward"]
        == 1.0
    )


def test_missing_invoice_lookup_counts_as_required_inspection(tmp_path: Path) -> None:
    task = _task(tmp_path, "missing-refund")

    assert _evaluate(task, tmp_path, [])["reward"] == 0.0
    assert (
        _evaluate(
            task,
            tmp_path,
            [{"name": "get_invoice", "arguments": {"invoice_id": "inv-999"}}],
        )["reward"]
        == 1.0
    )


def test_mutation_tasks_still_require_inspection_before_the_write(tmp_path: Path) -> None:
    task = _task(tmp_path, "paid-refund-direct")
    mutations = [
        "INSERT INTO refunds VALUES ('refund-inv-201', 'inv-201', 'test')",
        "UPDATE invoices SET status = 'refunded' WHERE id = 'inv-201'",
    ]
    refund = {"name": "refund_invoice", "arguments": {"invoice_id": "inv-201"}}
    lookup = {"name": "get_invoice", "arguments": {"invoice_id": "inv-201"}}

    assert _evaluate(task, tmp_path, [refund], mutations)["reward"] == 0.0
    assert _evaluate(task, tmp_path, [lookup, refund], mutations)["reward"] == 1.0
