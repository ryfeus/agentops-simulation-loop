"""Aurora DSQL schema for the billing repository."""

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS customers (
        id VARCHAR(128) PRIMARY KEY,
        status VARCHAR(32) NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS invoices (
        id VARCHAR(128) PRIMARY KEY,
        customer_id VARCHAR(128) NOT NULL,
        amount NUMERIC(12, 2) NOT NULL,
        currency VARCHAR(3) NOT NULL,
        status VARCHAR(32) NOT NULL,
        CONSTRAINT invoices_customer_fk
            FOREIGN KEY (customer_id) REFERENCES customers(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS refunds (
        id VARCHAR(128) PRIMARY KEY,
        invoice_id VARCHAR(128) NOT NULL,
        reason TEXT NOT NULL,
        CONSTRAINT refunds_invoice_unique UNIQUE (invoice_id),
        CONSTRAINT refunds_invoice_fk
            FOREIGN KEY (invoice_id) REFERENCES invoices(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS escalations (
        id VARCHAR(128) PRIMARY KEY,
        invoice_id VARCHAR(128) NOT NULL,
        reason TEXT NOT NULL,
        CONSTRAINT escalations_invoice_unique UNIQUE (invoice_id),
        CONSTRAINT escalations_invoice_fk
            FOREIGN KEY (invoice_id) REFERENCES invoices(id)
    )
    """,
)

RUNTIME_GRANTS = (
    "GRANT SELECT ON customers, invoices, refunds, escalations TO billing_runtime",
    "GRANT INSERT ON refunds, escalations TO billing_runtime",
    "GRANT UPDATE ON invoices TO billing_runtime",
)
