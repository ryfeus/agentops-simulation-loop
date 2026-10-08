"""Administrative Aurora DSQL schema and canonical-state operations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentops_demo.billing.dsql_schema import RUNTIME_GRANTS, SCHEMA_STATEMENTS
from agentops_demo.contracts.scenario import InitialState
from agentops_demo.validation.scenario import load_scenario

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TERRAFORM_DIR = REPOSITORY_ROOT / "infra" / "terraform"
SCENARIO_PATH = REPOSITORY_ROOT / "scenarios" / "disputed-refund" / "scenario.yaml"
ROLE_NAME = "billing_runtime"
ROLE_ARN_PATTERN = re.compile(r"^arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+$")


def terraform_outputs() -> dict[str, Any]:
    result = subprocess.run(
        ("terraform", f"-chdir={TERRAFORM_DIR}", "output", "-json"),
        check=True,
        capture_output=True,
        text=True,
    )
    raw = json.loads(result.stdout)
    return {name: entry["value"] for name, entry in raw.items()}


async def connect_admin(endpoint: str, region: str):
    from scripts.aws_context import verified_session

    if region != "us-west-2":
        raise RuntimeError("DSQL is intentionally pinned to us-west-2")
    verified_session()
    import aurora_dsql_asyncpg

    return await aurora_dsql_asyncpg.connect(
        host=endpoint,
        region=region,
        profile=os.getenv("AWS_PROFILE", "default"),
        user="admin",
        database="postgres",
    )


async def bootstrap() -> None:
    outputs = terraform_outputs()
    endpoint = str(outputs["dsql_endpoint"])
    region = str(outputs["aws_region"])
    role_arn = str(outputs["agentcore_runtime_role_arn"])
    if region != "us-west-2":
        raise RuntimeError(f"refusing DSQL bootstrap outside us-west-2: {region}")
    if not ROLE_ARN_PATTERN.fullmatch(role_arn):
        raise RuntimeError("Terraform returned an invalid AgentCore role ARN")

    connection = await connect_admin(endpoint, region)
    try:
        for statement in SCHEMA_STATEMENTS:
            await connection.execute(statement)
        role = await connection.fetchrow("SELECT 1 FROM pg_roles WHERE rolname = $1", ROLE_NAME)
        if role is None:
            await connection.execute(f"CREATE ROLE {ROLE_NAME} WITH LOGIN")

        mappings = await connection.fetch(
            "SELECT arn FROM sys.iam_pg_role_mappings WHERE pg_role_name = $1", ROLE_NAME
        )
        mapped_arns = {str(row["arn"]) for row in mappings}
        unexpected = mapped_arns - {role_arn}
        if unexpected:
            raise RuntimeError(
                f"{ROLE_NAME} is already mapped to unexpected IAM roles: {sorted(unexpected)}"
            )
        if role_arn not in mapped_arns:
            await connection.execute(f"AWS IAM GRANT {ROLE_NAME} TO '{role_arn}'")
        for statement in RUNTIME_GRANTS:
            try:
                await connection.execute(statement)
            except Exception as exc:
                raise RuntimeError(f"failed DSQL grant: {statement}") from exc
    finally:
        await connection.close()

    await reset_demo()
    print("Schema: OK")
    print(f"{ROLE_NAME} mapping: OK")
    print("Scenario disputed-refund: seeded")


async def reset_state(connection: Any, initial_state: InitialState) -> None:
    transaction = connection.transaction()
    await transaction.start()
    try:
        for table in ("refunds", "escalations", "invoices", "customers"):
            await connection.execute(f"DELETE FROM {table}")
        await connection.executemany(
            "INSERT INTO customers (id, status) VALUES ($1, $2)",
            [(item.id, item.status) for item in initial_state.customers],
        )
        await connection.executemany(
            """
            INSERT INTO invoices (id, customer_id, amount, currency, status)
            VALUES ($1, $2, $3, $4, $5)
            """,
            [
                (item.id, item.customer_id, item.amount, item.currency, item.status)
                for item in initial_state.invoices
            ],
        )
        await connection.executemany(
            "INSERT INTO refunds (id, invoice_id, reason) VALUES ($1, $2, $3)",
            [(item.id, item.invoice_id, item.reason) for item in initial_state.refunds],
        )
        await connection.executemany(
            "INSERT INTO escalations (id, invoice_id, reason) VALUES ($1, $2, $3)",
            [(item.id, item.invoice_id, item.reason) for item in initial_state.escalations],
        )
        await transaction.commit()
    except Exception:
        await transaction.rollback()
        raise


async def reset_demo() -> None:
    outputs = terraform_outputs()
    scenario = load_scenario(SCENARIO_PATH)
    connection = await connect_admin(str(outputs["dsql_endpoint"]), str(outputs["aws_region"]))
    try:
        await reset_state(connection, scenario.initial_state)
    finally:
        await connection.close()
    print("Canonical DSQL demo state restored")


async def read_state() -> dict[str, Any]:
    outputs = terraform_outputs()
    connection = await connect_admin(str(outputs["dsql_endpoint"]), str(outputs["aws_region"]))
    try:
        invoice: Mapping[str, Any] | None = await connection.fetchrow(
            """
            SELECT id, customer_id, amount, currency, status
            FROM invoices WHERE id = $1
            """,
            "inv-123",
        )
        refunds = await connection.fetchval(
            "SELECT COUNT(*) FROM refunds WHERE invoice_id = $1", "inv-123"
        )
        escalations = await connection.fetchval(
            "SELECT COUNT(*) FROM escalations WHERE invoice_id = $1", "inv-123"
        )
    finally:
        await connection.close()
    return {
        "invoice": (
            {
                "id": invoice["id"],
                "customer_id": invoice["customer_id"],
                "amount": str(invoice["amount"]),
                "currency": invoice["currency"],
                "status": invoice["status"],
            }
            if invoice is not None
            else None
        ),
        "refunds": int(refunds),
        "escalations": int(escalations),
    }


async def show_state(output: Path | None = None) -> None:
    state = await read_state()
    serialized = json.dumps(state, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized)
    print(serialized, end="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("bootstrap", "reset", "show"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.operation == "bootstrap":
        asyncio.run(bootstrap())
    elif args.operation == "reset":
        asyncio.run(reset_demo())
    else:
        asyncio.run(show_state(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
