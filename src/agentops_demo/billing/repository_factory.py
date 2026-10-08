"""Configuration boundary for billing repository implementations."""

from __future__ import annotations

from pathlib import Path

from agentops_demo.billing.dsql_settings import DSQLSettings
from agentops_demo.billing.sqlite_repository import SQLiteBillingRepository
from agentops_demo.contracts.billing import BillingRepository, BillingWorldInspector


def create_billing_repository(
    *,
    backend: str,
    sqlite_path: str | Path = "data/billing.db",
    dsql_settings: DSQLSettings | None = None,
) -> BillingRepository:
    """Create the configured backend while keeping DSQL imports optional."""

    if backend == "sqlite":
        return SQLiteBillingRepository(sqlite_path)
    if backend == "dsql":
        from agentops_demo.billing.dsql_repository import DSQLBillingRepository

        return DSQLBillingRepository(dsql_settings or DSQLSettings.from_environment())
    raise ValueError(f"unsupported billing repository backend {backend!r}")


def create_billing_world_inspector(
    *,
    backend: str,
    sqlite_path: str | Path = "data/billing.db",
    dsql_settings: DSQLSettings | None = None,
) -> BillingWorldInspector:
    """Create only the runtime-internal inspection capability."""

    repository = create_billing_repository(
        backend=backend, sqlite_path=sqlite_path, dsql_settings=dsql_settings
    )
    if not isinstance(repository, BillingWorldInspector):
        raise TypeError(f"backend {backend!r} does not support world snapshots")
    return repository
