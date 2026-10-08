from pathlib import Path

import pytest

from agentops_demo.billing.repository_factory import create_billing_repository
from agentops_demo.billing.sqlite_repository import SQLiteBillingRepository


def test_repository_factory_defaults_to_sqlite_path(tmp_path: Path) -> None:
    path = tmp_path / "billing.db"
    repository = create_billing_repository(backend="sqlite", sqlite_path=path)
    assert isinstance(repository, SQLiteBillingRepository)
    assert repository.path == path


def test_repository_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported billing repository backend"):
        create_billing_repository(backend="other")
