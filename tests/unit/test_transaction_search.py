from __future__ import annotations

import json
import stat
import sys
from types import SimpleNamespace

import pytest

from scripts.ensure_transaction_search import ensure, restore


class FakeXRay:
    def __init__(self, destination="XRay", indexing=0.0):
        self.destination = destination
        self.indexing = indexing
        self.destination_updates = 0
        self.index_updates = 0

    def get_trace_segment_destination(self):
        return {"Destination": self.destination, "Status": "ACTIVE"}

    def update_trace_segment_destination(self, *, Destination):
        self.destination = Destination
        self.destination_updates += 1

    def get_indexing_rules(self):
        return (
            [{"unused": True}]
            if False
            else {
                "IndexingRules": [
                    {
                        "Name": "Default",
                        "Rule": {"Probabilistic": {"DesiredSamplingPercentage": self.indexing}},
                    }
                ]
            }
        )

    def update_indexing_rule(self, *, Name, Rule):
        assert Name == "Default"
        self.indexing = Rule["Probabilistic"]["DesiredSamplingPercentage"]
        self.index_updates += 1


class FakeSession:
    def __init__(self, xray, account="123456789012", policy=True):
        self.xray = xray
        self.account = account
        self.policy = policy

    def client(self, name):
        if name == "sts":
            return SimpleNamespace(get_caller_identity=lambda: {"Account": self.account})
        if name == "logs":
            policies = [{"policyName": "agentops-demo-transaction-search"}] if self.policy else []
            return SimpleNamespace(
                describe_resource_policies=lambda: {"resourcePolicies": policies}
            )
        return self.xray


def install_boto3(monkeypatch, session):
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(Session=lambda **_kwargs: session))
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", "123456789012")


def test_transaction_search_updates_only_drift(monkeypatch, tmp_path) -> None:
    xray = FakeXRay()
    install_boto3(monkeypatch, FakeSession(xray))
    monkeypatch.setenv("AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE", "yes")
    result = ensure(apply=True, indexing_percentage=100.0, receipt_path=tmp_path / "receipt.json")
    assert result["destination"] == "CloudWatchLogs"
    assert xray.destination_updates == 1
    assert xray.index_updates == 1
    ensure(apply=True, indexing_percentage=100.0, receipt_path=tmp_path / "second.json")
    assert xray.destination_updates == 1
    assert xray.index_updates == 1


def test_transaction_search_check_is_read_only(monkeypatch) -> None:
    install_boto3(monkeypatch, FakeSession(FakeXRay()))
    with pytest.raises(RuntimeError, match="destination"):
        ensure(check_only=True)


def test_transaction_search_timeout_remains_overrideable(monkeypatch, tmp_path) -> None:
    class PendingXRay(FakeXRay):
        def get_trace_segment_destination(self):
            return {"Destination": "CloudWatchLogs", "Status": "PENDING"}

    install_boto3(monkeypatch, FakeSession(PendingXRay()))
    monkeypatch.setenv("AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE", "yes")
    with pytest.raises(RuntimeError, match="ACTIVE"):
        ensure(
            apply=True, timeout=0, indexing_percentage=100, receipt_path=tmp_path / "receipt.json"
        )


def test_transaction_search_rejects_wrong_account_or_missing_policy(monkeypatch) -> None:
    install_boto3(monkeypatch, FakeSession(FakeXRay(), account="000000000000"))
    with pytest.raises(RuntimeError, match="refusing AWS"):
        ensure()
    install_boto3(monkeypatch, FakeSession(FakeXRay(), policy=False))
    monkeypatch.setenv("AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE", "yes")
    with pytest.raises(RuntimeError, match="resource policy"):
        ensure(apply=True, indexing_percentage=100)


def test_default_preview_does_not_mutate_even_with_percentage(monkeypatch):
    xray = FakeXRay()
    install_boto3(monkeypatch, FakeSession(xray, policy=False))
    result = ensure(indexing_percentage=100)
    assert result["indexing_percentage"] == 0
    assert result["resource_policy_present"] is False
    assert xray.destination_updates == xray.index_updates == 0


def test_apply_requires_acknowledgement_before_session(monkeypatch):
    monkeypatch.delenv("AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE", raising=False)
    monkeypatch.setitem(
        sys.modules, "boto3", SimpleNamespace(Session=lambda **kwargs: pytest.fail("AWS called"))
    )
    with pytest.raises(RuntimeError, match="ACCOUNT_WIDE"):
        ensure(apply=True, indexing_percentage=100)


def test_receipt_and_idempotent_restore(monkeypatch, tmp_path):
    xray = FakeXRay(indexing=7)
    install_boto3(monkeypatch, FakeSession(xray))
    monkeypatch.setenv("AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE", "yes")
    path = tmp_path / "receipt.json"
    ensure(apply=True, indexing_percentage=100, receipt_path=path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    receipt = json.loads(path.read_text())
    assert receipt["before"] == {"destination": "XRay", "indexing_percentage": 7}
    restore(path)
    assert xray.destination == "XRay" and xray.indexing == 7
    updates = (xray.destination_updates, xray.index_updates)
    restore(path)
    assert (xray.destination_updates, xray.index_updates) == updates
    xray.indexing = 8
    with pytest.raises(RuntimeError, match="concurrently"):
        restore(path)


def test_partial_failure_can_restore_journaled_state(monkeypatch, tmp_path):
    class FailingXRay(FakeXRay):
        def update_indexing_rule(self, **kwargs):
            super().update_indexing_rule(**kwargs)
            raise RuntimeError("connection dropped after update")

    xray = FailingXRay()
    install_boto3(monkeypatch, FakeSession(xray))
    monkeypatch.setenv("AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE", "yes")
    path = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError, match="connection dropped"):
        ensure(apply=True, indexing_percentage=100, receipt_path=path)
    xray.update_indexing_rule = lambda **kwargs: FakeXRay.update_indexing_rule(xray, **kwargs)
    restore(path)
    assert xray.destination == "XRay" and xray.indexing == 0


def test_restore_refuses_other_account_and_symlink(monkeypatch, tmp_path):
    xray = FakeXRay()
    install_boto3(monkeypatch, FakeSession(xray))
    monkeypatch.setenv("AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE", "yes")
    path = tmp_path / "receipt.json"
    ensure(apply=True, indexing_percentage=100, receipt_path=path)
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", "210987654321")
    with pytest.raises(RuntimeError, match="different account"):
        restore(path)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(RuntimeError, match="symlink"):
        restore(link)


def test_restore_rejects_readable_receipt(monkeypatch, tmp_path):
    install_boto3(monkeypatch, FakeSession(FakeXRay()))
    monkeypatch.setenv("AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE", "yes")
    path = tmp_path / "receipt.json"
    ensure(apply=True, indexing_percentage=100, receipt_path=path)
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="0600"):
        restore(path)


def test_receipt_creation_failure_prevents_mutation(monkeypatch, tmp_path):
    xray = FakeXRay()
    install_boto3(monkeypatch, FakeSession(xray))
    monkeypatch.setenv("AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE", "yes")
    path = tmp_path / "receipt.json"
    path.write_text("existing receipt")
    with pytest.raises(FileExistsError):
        ensure(apply=True, indexing_percentage=100, receipt_path=path)
    assert xray.destination_updates == xray.index_updates == 0
    assert path.read_text() == "existing receipt"
