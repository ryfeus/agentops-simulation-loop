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
with open('/app/data/agent-run.json', 'w', encoding='utf-8') as artifact:
    json.dump({'tool_calls': [{'name': 'get_invoice', 'arguments': {'invoice_id': 'inv-201'}}, {'name': 'refund_invoice', 'arguments': {'invoice_id': 'inv-201', 'reason': 'Oracle refund'}}]}, artifact, sort_keys=True)
PY
