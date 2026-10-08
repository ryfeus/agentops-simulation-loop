#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import sqlite3

with sqlite3.connect("/app/data/billing.db") as connection:
    connection.execute(
        "INSERT INTO escalations (id, invoice_id, reason) VALUES (?, ?, ?)",
        ("escalation-inv-123", "inv-123", "Oracle specialist review"),
    )
    connection.execute("UPDATE invoices SET status = 'disputed' WHERE id = ?", ("inv-123",))
PY
