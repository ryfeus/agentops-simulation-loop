from __future__ import annotations

import pytest

from agentops_demo.billing.dsql_settings import DSQLSettings


def test_dsql_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSQL_ENDPOINT", "cluster.dsql.us-west-2.on.aws")
    monkeypatch.setenv("DSQL_REGION", "us-west-2")
    settings = DSQLSettings.from_environment()
    assert settings.user == "billing_runtime"
    assert settings.database == "postgres"
    assert settings.max_retries == 4


@pytest.mark.parametrize("name", ["DSQL_ENDPOINT", "DSQL_REGION"])
def test_dsql_settings_require_environment(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv("DSQL_ENDPOINT", "endpoint")
    monkeypatch.setenv("DSQL_REGION", "us-west-2")
    monkeypatch.delenv(name)
    with pytest.raises(ValueError, match=name):
        DSQLSettings.from_environment()


@pytest.mark.parametrize("retries", [-1, 9])
def test_dsql_settings_bound_retries(retries: int) -> None:
    with pytest.raises(ValueError, match="between 0 and 8"):
        DSQLSettings(endpoint="endpoint", region="us-west-2", max_retries=retries)
