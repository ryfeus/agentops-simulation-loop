#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import json, sqlite3

with sqlite3.connect('/app/data/billing.db') as connection:
    connection.execute(
        'INSERT OR IGNORE INTO refunds '
        '(id, invoice_id, reason) VALUES (?, ?, ?)',
        ('refund-inv-201', 'inv-201', 'Oracle refund'),
    )
    connection.execute("UPDATE invoices SET status = 'refunded' WHERE id = ?",
                       ('inv-201',))
    connection.execute(
        'INSERT OR IGNORE INTO escalations '
        '(id, invoice_id, reason) VALUES (?, ?, ?)',
        ('escalation-inv-123', 'inv-123', 'Oracle specialist review'),
    )
    connection.execute(
        "UPDATE invoices SET status = 'disputed' WHERE id = ?",
        ('inv-123',),
    )
with open('/app/data/agent-run.json', 'w', encoding='utf-8') as artifact:
    json.dump({'tool_calls': [{'name': 'get_invoice', 'arguments': {'invoice_id': 'inv-123'}}, {'name': 'get_invoice', 'arguments': {'invoice_id': 'inv-201'}}, {'name': 'refund_invoice', 'arguments': {'invoice_id': 'inv-201', 'reason': 'Oracle refund'}}, {'name': 'escalate_dispute', 'arguments': {'invoice_id': 'inv-123', 'reason': 'Oracle specialist review'}}]}, artifact, sort_keys=True)
PY
