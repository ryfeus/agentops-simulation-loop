BEGIN TRANSACTION;
CREATE TABLE customers (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL
);
INSERT INTO "customers" VALUES('customer-1','active');
CREATE TABLE escalations (
    id TEXT PRIMARY KEY,
    invoice_id TEXT NOT NULL REFERENCES invoices(id),
    reason TEXT NOT NULL
);
CREATE TABLE invoices (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    amount TEXT NOT NULL,
    currency TEXT NOT NULL,
    status TEXT NOT NULL
);
INSERT INTO "invoices" VALUES('inv-301','customer-1','49.00','USD','refunded');
CREATE TABLE refunds (
    id TEXT PRIMARY KEY,
    invoice_id TEXT NOT NULL REFERENCES invoices(id),
    reason TEXT NOT NULL
);
INSERT INTO "refunds" VALUES('refund-inv-301','inv-301','prior adjustment');
CREATE UNIQUE INDEX refunds_invoice_id_unique ON refunds(invoice_id);
CREATE UNIQUE INDEX escalations_invoice_id_unique ON escalations(invoice_id);
COMMIT;
