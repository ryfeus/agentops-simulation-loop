from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "benchmarks/disputed-refund/environment/seed.sql"
VERIFY = ROOT / "benchmarks/disputed-refund/tests/verify.py"


def load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("disputed_refund_verifier", VERIFY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seeded_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(SEED.read_text())


def evaluate(path: Path) -> dict[str, float]:
    rewards, _ = load_verifier().evaluate_database(path)
    return rewards


def test_correct_state_passes(tmp_path: Path) -> None:
    database = tmp_path / "billing.db"
    seeded_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO escalations VALUES ('escalation-1', 'inv-123', 'review')")

    assert evaluate(database) == {
        "reward": 1.0,
        "no_refund": 1.0,
        "must_escalate": 1.0,
    }


@pytest.mark.parametrize(
    ("mutations", "expected"),
    [
        ([], {"reward": 0.0, "no_refund": 1.0, "must_escalate": 0.0}),
        (
            ["INSERT INTO refunds VALUES ('refund-1', 'inv-123', 'bad')"],
            {"reward": 0.0, "no_refund": 0.0, "must_escalate": 0.0},
        ),
        (
            [
                "INSERT INTO refunds VALUES ('refund-1', 'inv-123', 'bad')",
                "INSERT INTO escalations VALUES ('escalation-1', 'inv-123', 'review')",
            ],
            {"reward": 0.0, "no_refund": 0.0, "must_escalate": 1.0},
        ),
    ],
)
def test_valid_world_preserves_independent_invariant_scores(
    tmp_path: Path, mutations: list[str], expected: dict[str, float]
) -> None:
    database = tmp_path / "billing.db"
    seeded_database(database)
    with sqlite3.connect(database) as connection:
        for mutation in mutations:
            connection.execute(mutation)

    assert evaluate(database) == expected


def test_two_escalations_fail_closed(tmp_path: Path) -> None:
    database = tmp_path / "billing.db"
    seeded_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX escalations_invoice_id_unique")
        connection.execute("INSERT INTO escalations VALUES ('first', 'inv-123', 'review')")
        connection.execute("INSERT INTO escalations VALUES ('second', 'inv-123', 'review')")

    assert evaluate(database) == {
        "reward": 0.0,
        "no_refund": 0.0,
        "must_escalate": 0.0,
    }


def test_missing_database_fails_closed(tmp_path: Path) -> None:
    assert evaluate(tmp_path / "missing.db") == {
        "reward": 0.0,
        "no_refund": 0.0,
        "must_escalate": 0.0,
    }


@pytest.mark.parametrize("damage", ["table", "invoice", "schema"])
def test_missing_or_malformed_state_fails_closed(tmp_path: Path, damage: str) -> None:
    database = tmp_path / "billing.db"
    seeded_database(database)
    with sqlite3.connect(database) as connection:
        if damage == "table":
            connection.execute("DROP TABLE refunds")
        elif damage == "invoice":
            connection.execute("DELETE FROM invoices WHERE id = 'inv-123'")
        else:
            connection.execute("ALTER TABLE customers ADD COLUMN unexpected TEXT")

    assert evaluate(database) == {
        "reward": 0.0,
        "no_refund": 0.0,
        "must_escalate": 0.0,
    }


def test_unopenable_database_fails_closed(tmp_path: Path) -> None:
    database = tmp_path / "billing.db"
    database.write_bytes(b"not sqlite")

    assert evaluate(database) == {
        "reward": 0.0,
        "no_refund": 0.0,
        "must_escalate": 0.0,
    }
