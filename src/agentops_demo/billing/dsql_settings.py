"""Strict, secret-free Aurora DSQL runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _required_environment(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be set and nonblank")
    return value


@dataclass(frozen=True)
class DSQLSettings:
    """Connection identity for IAM-authenticated Aurora DSQL access."""

    endpoint: str
    region: str
    user: str = "billing_runtime"
    database: str = "postgres"
    max_retries: int = 4

    def __post_init__(self) -> None:
        for name in ("endpoint", "region", "user", "database"):
            if not getattr(self, name).strip():
                raise ValueError(f"DSQL {name} must be nonblank")
        if not 0 <= self.max_retries <= 8:
            raise ValueError("DSQL max_retries must be between 0 and 8")

    @classmethod
    def from_environment(cls) -> DSQLSettings:
        try:
            max_retries = int(os.getenv("DSQL_MAX_RETRIES", "4"))
        except ValueError as exc:
            raise ValueError("DSQL_MAX_RETRIES must be an integer") from exc
        return cls(
            endpoint=_required_environment("DSQL_ENDPOINT"),
            region=_required_environment("DSQL_REGION"),
            user=os.getenv("DSQL_USER", "billing_runtime"),
            database=os.getenv("DSQL_DATABASE", "postgres"),
            max_retries=max_retries,
        )
