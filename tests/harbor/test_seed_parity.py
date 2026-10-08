from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path

from agentops_demo.validation.scenario import load_scenario
from scripts.generate_harbor_seed import render_seed

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "scenarios/disputed-refund/scenario.yaml"
TASK = ROOT / "benchmarks/disputed-refund"
SEED = TASK / "environment/seed.sql"


async def test_committed_seed_matches_generator() -> None:
    assert SEED.read_text() == await render_seed(SCENARIO)


def test_seed_semantically_matches_canonical_scenario(tmp_path: Path) -> None:
    scenario = load_scenario(SCENARIO)
    database = tmp_path / "billing.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(SEED.read_text())
        customer_rows = connection.execute("SELECT id, status FROM customers").fetchall()
        invoice_rows = connection.execute(
            "SELECT id, customer_id, amount, currency, status FROM invoices"
        ).fetchall()
        refund_rows = connection.execute("SELECT id FROM refunds").fetchall()
        escalation_rows = connection.execute("SELECT id FROM escalations").fetchall()
        refund_indexes = connection.execute("PRAGMA index_list(refunds)").fetchall()
        escalation_indexes = connection.execute("PRAGMA index_list(escalations)").fetchall()

    invoice = scenario.initial_state.invoices[0]
    assert customer_rows == [("customer-42", "active")]
    assert invoice_rows == [
        (invoice.id, invoice.customer_id, str(Decimal("89.00")), invoice.currency, invoice.status)
    ]
    assert refund_rows == []
    assert escalation_rows == []
    assert "refunds_invoice_id_unique" in {row[1] for row in refund_indexes}
    assert "escalations_invoice_id_unique" in {row[1] for row in escalation_indexes}


def test_instruction_matches_canonical_scenario() -> None:
    scenario = load_scenario(SCENARIO)
    assert (TASK / "instruction.md").read_text().strip() == scenario.instruction
