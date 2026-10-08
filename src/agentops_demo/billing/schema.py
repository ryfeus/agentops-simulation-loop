"""SQLite schema private to the Phase 1 repository implementation."""

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS invoices (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    amount TEXT NOT NULL,
    currency TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS refunds (
    id TEXT PRIMARY KEY,
    invoice_id TEXT NOT NULL REFERENCES invoices(id),
    reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    invoice_id TEXT NOT NULL REFERENCES invoices(id),
    reason TEXT NOT NULL
);
"""

CARDINALITY_INDEX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS refunds_invoice_id_unique ON refunds(invoice_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS escalations_invoice_id_unique ON escalations(invoice_id)",
)
