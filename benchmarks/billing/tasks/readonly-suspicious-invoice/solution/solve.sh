#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import json, sqlite3

with sqlite3.connect('/app/data/billing.db') as connection:
    pass
with open('/app/data/agent-run.json', 'w', encoding='utf-8') as artifact:
    json.dump({'tool_calls': [{'name': 'get_invoice', 'arguments': {'invoice_id': 'inv-123'}}]}, artifact, sort_keys=True)
PY
